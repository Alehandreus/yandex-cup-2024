# can contain code from ktoshik's CV course


# set some constants
from setup import *

import tqdm
import hydra
import torch
from torch import nn

from myutils import *
from dataset import cover_dataloader
from models_arc import ArcModel, ArcFaceLoss


class TrainModule:
    def __init__(self, cfg):
        self.cfg = cfg

        # inflate number of classes
        self.model = ArcModel(num_classes=cfg.num_classes).cuda()

        self.arcface_loss = ArcFaceLoss(m=0.5, s=45)

        self.ema = MetricLogger(alpha=0.99)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
        )

        self.t_loader = cover_dataloader(data_split="train", **cfg)
        self.v_loader = cover_dataloader(data_split="val", **cfg)
        self.test_loader = cover_dataloader(data_split="test", **cfg)

    def train(self):
        # self.validate()

        for epoch_i in range(self.cfg.epochs):
            self.model.unfreeze(epoch_i)
            self.optimizer.zero_grad()

            self.pbar = tqdm(enumerate(self.t_loader), total=len(self.t_loader))    
            for step, batch in self.pbar:       
                loss = self.training_step(step, batch)        
                self.ema.update(loss)    
                self.pbar.set_description(f"loss_ema: {self.ema.exp:.3f}")

            self.validate()

        torch.save(self.model.state_dict(), "arc.pt")

    def training_step(self, step, batch):
        self.model.train()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            emb = self.model.forward(batch["anchor"].cuda())
            loss = self.arcface_loss(
                self.model.arcface_head(emb),
                batch["anchor_label"].long().cuda(),
            )

        loss.backward()
        if step % (100 // self.cfg.batch_size) == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        clique_ids = []
        embeddings = []
        for batch in tqdm(self.v_loader):
            anchor_ids = batch["anchor_id"]
            anchors = batch["anchor"].cuda()
            embs = self.model.forward(anchors)
            for anchor_id, emb in zip(anchor_ids, embs):
                clique_id = self.v_loader.dataset.version2clique.loc[int(anchor_id), 'clique']
                clique_ids.append(clique_id)
                embeddings.append(emb)

        preds = torch.stack(embeddings)
        clique_ids = torch.tensor(clique_ids)

        metrics = calculate_ranking_metrics_torch(embeddings=preds, cliques=clique_ids, rerank=self.cfg.rerank, device='cpu')

        print(
            f"\n{' Validation Results ':=^50}\n"
            + "\n".join([f'"{key}": {value.mean()}' for key, value in metrics.items()])
            + f"\n{' End of Validation ':=^50}\n"
        )
    
    @torch.no_grad()
    def test(self):
        self.model.eval()

        trackids = []
        embeddings = []
        for batch in tqdm(self.test_loader):
            anchor_ids = batch["anchor_id"]
            anchors = batch["anchor"].cuda()
            embs = self.model.forward(anchors)
            for anchor_id, emb in zip(anchor_ids, embs):
                embeddings.append(emb)
                trackids.append(anchor_id)

        embeddings = torch.stack(embeddings)

        s_distances = pairwise_distances_torch(embeddings.cuda(), 100, chunk_size=64, rerank=self.cfg.rerank).cpu()

        trackids_expanded = torch.tensor(trackids).expand(len(trackids), -1)
        ranked_lists = torch.gather(trackids_expanded, 1, s_distances[:, 1:])
        ranked_lists = ranked_lists.numpy().tolist()
        predictions = list(zip(trackids, ranked_lists))

        save_test_predictions(predictions, output_dir='.')


@hydra.main(version_base=None, config_path="config", config_name="arc")
def main(cfg):
    module = TrainModule(cfg)

    if cfg.mae_path:
        module.model.mae.load_state_dict(torch.load(cfg.mae_path, weights_only=True))

    if cfg.checkpoint_path:
        module.model.load_state_dict(torch.load(cfg.checkpoint_path, weights_only=True), strict=False)

    if cfg.test_only:
        module.test()
        return

    module.train()


if __name__ == "__main__":
    main()
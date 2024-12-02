import os
from typing import Literal
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class CoverDataset(Dataset):
    def __init__(
        self,
        data_path,
        data_split: Literal["train", "val", "test"],
        min_clique_size = 0,
    ):
        super().__init__()
        self.data_path = data_path
        self.file_ext = 'npy'
        self.dataset_path = self.data_path + ("test" if data_split == "test" else "train")
        self.data_split = data_split
        self.debug = False
        self.max_len = 50
        self.min_clique_size = min_clique_size
        self._load_data()
        self.rnd_indices = np.random.permutation(len(self.track_ids))
        self.current_index = 0
        if self.data_split != "test":
            self.num_classes = len(self.cliques_subset)

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, index):
        track_id = self.track_ids[index]
        anchor_cqt = self._load_cqt(track_id)
        anchor_cqt_tr = anchor_cqt
        
        if self.data_split == "train":
            clique_id = self.version2clique.loc[track_id, 'clique']
            pos_id, neg_id = self._triplet_sampling(track_id, clique_id)
            positive_cqt = self._load_cqt(pos_id)
            negative_cqt = self._load_cqt(neg_id)

            # augmentations for anchor cqt spectrogram
            if torch.rand(1) > 0.5:
                anchor_cqt_tr = torch.flip(anchor_cqt_tr, [1])
            if torch.rand(1) > 0.5:
                anchor_cqt_tr = torch.roll(anchor_cqt_tr, int(torch.randint(-10, 10, (1,))), 1)
        else:
            clique_id = -1
            pos_id = -1
            positive_cqt = torch.zeros_like(anchor_cqt)
            neg_id = -1
            negative_cqt = torch.zeros_like(anchor_cqt)

        return dict(
            anchor_id=track_id,
            anchor=anchor_cqt,
            anchor_tr=anchor_cqt_tr,
            anchor_label=torch.tensor(clique_id, dtype=torch.float),
            positive_id=pos_id,
            positive=positive_cqt,
            negative_id=neg_id,
            negative=negative_cqt,
        )

    def _make_file_path(self, track_id, file_ext):
        a = track_id % 10
        b = track_id // 10 % 10
        c = track_id // 100 % 10
        return os.path.join(str(c), str(b), str(a), f'{track_id}.{file_ext}')

    def _triplet_sampling(self, track_id, clique_id):
        versions = self.versions.loc[clique_id, "versions"]
        pos_list = np.setdiff1d(versions, track_id)
        pos_id = np.random.choice(pos_list, 1)[0]
        if self.current_index >= len(self.rnd_indices):
            self.current_index = 0
            self.rnd_indices = np.random.permutation(len(self.track_ids))
        neg_id = self.track_ids[self.rnd_indices[self.current_index]]
        self.current_index += 1
        while neg_id in versions:
            if self.current_index >= len(self.rnd_indices):
                self.current_index = 0
                self.rnd_indices = np.random.permutation(len(self.track_ids))
            neg_id = self.track_ids[self.rnd_indices[self.current_index]]
            self.current_index += 1
        return (pos_id, neg_id)

    def _load_data(self):
        if self.data_split in ['train', 'val']:
            cliques_subset = np.load(os.path.join(self.data_path, "splits", "{}_cliques.npy".format(self.data_split)))
            self.cliques_subset = cliques_subset.tolist()
            self.versions = pd.read_csv(
                os.path.join(self.data_path, "cliques2versions.tsv"), sep='\t', converters={"versions": eval}
            )
            if self.data_split == "train":
                self.versions = self.versions[self.versions["versions"].map(len) >= self.min_clique_size]
                cliques_subset = set(cliques_subset) & set(self.versions["clique"].values)
                self.cliques_subset = list(cliques_subset)

            self.cluques2versions = {
                row["clique"]: np.array(row["versions"])
                for _, row in self.versions.iterrows()
                if row["clique"] in cliques_subset
            }
            self.versions = self.versions[self.versions["clique"].isin(set(cliques_subset))]
            mapping = {}
            for k, clique in enumerate(sorted(cliques_subset)):
                mapping[clique] = k
            self.versions["clique"] = self.versions["clique"].map(lambda x: mapping[x])
            self.versions.set_index("clique", inplace=True)
            self.version2clique = pd.DataFrame(
                [{'version': version, 'clique': clique} for clique, row in self.versions.iterrows() for version in row['versions']]
            ).set_index('version')
            self.track_ids = self.version2clique.index.to_list()
            self.id_to_idx = {track_id: idx for idx, track_id in enumerate(self.track_ids)}
        else:
            self.track_ids = np.load(os.path.join(self.data_path, "splits", "{}_ids.npy".format(self.data_split)))

    def _load_cqt(self, track_id) :
        filename = os.path.join(self.dataset_path, self._make_file_path(track_id, self.file_ext))
        cqt_spectrogram = np.load(filename)
        return torch.from_numpy(cqt_spectrogram)


def cover_dataloader(
    data_path,
    data_split: Literal["train", "val", "test"],
    batch_size,
    min_clique_size=0,
    num_workers=5,
    **kwargs,
):
    return DataLoader(
        CoverDataset(data_path, data_split, min_clique_size),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=data_split == "train",
        drop_last=False,
    )
    
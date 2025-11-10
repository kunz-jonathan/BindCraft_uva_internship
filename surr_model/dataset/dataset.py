from pathlib import Path
from typing import List, Tuple
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxtyping import Array, Int
from torch.utils.data import DataLoader
import esm2quinox

def test():
    print('second')

class Dataset:
    def __init__(self, transform=None, columns=List, data_path=Path) -> None:
        self.data = pd.read_csv(data_path)

        self.sequences_prot = self.data[columns[0]].tolist()
        self.sequences_prot = [x.split("-")[0] for x in self.sequences_prot]
        self.sequences_pept = self.data[columns[0]].tolist()
        self.sequences_pept = [x.split("-")[0] for x in self.sequences_pept]
        self.aff_Vals = -1 * np.array(self.data[columns[1]])
        self.transform = transform
        self.max_seq_prot = 0
        self.max_seq_pept = 0

        for seq in self.sequences_prot:
            if self.max_seq_prot < len(seq):
                self.max_seq_prot = len(seq)

        for seq in self.sequences_pept:
            if self.max_seq_pept < len(seq):
                self.max_seq_pept = len(seq)

    def __len__(self):
        return len(self.sequences_pept)

    def __getitem__(self, idx):
        affVal = self.aff_Vals[idx]

        # protein
        seq = self.sequences_prot[idx]
        if self.transform:
            padd_len = self.max_seq_prot - len(seq)
            seq = self.transform([seq + "." * padd_len])[0]

            seq_prot = np.array(seq)

        # peptide
        seq = self.sequences_pept[idx]
        if self.transform:
            padd_len = self.max_seq_pept - len(seq)
            seq = self.transform([seq + "." * padd_len])[0]

            seq_pept = np.array(seq)

        return seq_prot, seq_pept, affVal


def initialize_datasets(paths: List, batches=List):
    Dataset_training = Dataset(
        data_path=Path(paths[0]),
        transform=esm2quinox.tokenise,
        columns=["Sequence", "delta_G"],
    )
    Dataset_validation = Dataset(
        data_path=Path(paths[1]),
        transform=esm2quinox.tokenise,
        columns=["Sequence", "delta_G"],
    )
    Dataset_test = Dataset(
        data_path=Path(paths[2]),
        transform=esm2quinox.tokenise,
        columns=["Sequence", "delta_G"],
    )

    scaling_factor = max(np.array(Dataset_training.aff_Vals).reshape(-1, 1))

    ds_training_scaled = (
        np.array(Dataset_training.aff_Vals).reshape(-1, 1) / scaling_factor
    )
    Dataset_training.aff_Vals = np.array(ds_training_scaled)
    training_DataLoader = DataLoader(Dataset_training, batch_size=batches[0])

    ds_validation_scaled = (
        np.array(Dataset_training.aff_Vals).reshape(-1, 1) / scaling_factor
    )
    Dataset_validation.aff_Vals = np.array(ds_validation_scaled)
    validation_DataLoader = DataLoader(Dataset_validation, batch_size=batches[1])

    ds_test_scaled = np.array(Dataset_test.aff_Vals).reshape(-1, 1) / scaling_factor
    Dataset_test.aff_Vals = np.array(ds_test_scaled)
    test_DataLoader = DataLoader(Dataset_test, batch_size=batches[2])

    return training_DataLoader, validation_DataLoader, test_DataLoader

class Dataset_PEPBI:
    def __init__(self, transform=None, columns=List, data_path=Path) -> None:
        """
        columns (List):
            0 ~ sequence_prot
            1 ~ sequence_pept
            2 ~ aff_Vals - alr. scaled
        """
        self.data = pd.read_csv(data_path)

        self.sequences_prot = self.data[columns[0]].tolist()
        self.sequences_prot = [x.strip() for x in self.sequences_prot]
        #self.sequences_prot = [x.split("-")[0] for x in self.sequences_prot]
        self.sequences_pept = self.data[columns[1]].tolist()
        self.sequences_pept = [x.strip() for x in self.sequences_pept]
        self.aff_Vals = -1* np.array(self.data[columns[2]])
        self.transform = transform
        self.max_seq_prot = 0
        self.max_seq_pept = 0

        for seq in self.sequences_prot:
            if self.max_seq_prot < len(seq):
                self.max_seq_prot = len(seq)

        for seq in self.sequences_pept:
            if self.max_seq_pept < len(seq):
                self.max_seq_pept = len(seq)

    def __len__(self):
        return len(self.sequences_pept)

    def __getitem__(self, idx):
        affVal = self.aff_Vals[idx]

        # protein
        seq = self.sequences_prot[idx]
        if self.transform:
            padd_len = self.max_seq_prot - len(seq)
            seq = self.transform([seq + "." * padd_len])[0]

            seq_prot = np.array(seq)

        # peptide
        seq = self.sequences_pept[idx]
        if self.transform:
            padd_len = self.max_seq_pept - len(seq)
            seq = self.transform([seq + "." * padd_len])[0]

            seq_pept = np.array(seq)

        return seq_prot, seq_pept, affVal

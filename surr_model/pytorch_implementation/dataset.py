import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

class Dataset_PEPBI:
    def __init__(self, transform, columns, data_path) -> None:
        """
        columns (List):
            0 ~ sequence_prot
            1 ~ sequence_pept
            2 ~ aff_Vals - alr. scaled
        """
        self.transform = transform
        self.data = pd.read_csv(data_path)

        self.sequences_prot = self.data[columns[0]].astype(str).str.strip().tolist()
        self.sequences_pept = self.data[columns[1]].astype(str).str.strip().tolist()

        # Tokenize all sequences at once to preserve order and be more efficient
        prot_enc = transform(self.sequences_prot, padding=True, return_tensors="pt")
        pept_enc = transform(self.sequences_pept, padding=True, return_tensors="pt")

        # Store each sample as a small dict (keeps same format as per-item tokenization)
        self.sequences_prot = [
            {"input_ids": prot_enc["input_ids"][i].unsqueeze(0), "attention_mask": prot_enc["attention_mask"][i].unsqueeze(0)}
            for i in range(prot_enc["input_ids"].size(0))
        ]
        self.sequences_pept = [
            {"input_ids": pept_enc["input_ids"][i].unsqueeze(0), "attention_mask": pept_enc["attention_mask"][i].unsqueeze(0)}
            for i in range(pept_enc["input_ids"].size(0))
        ]

        self.aff_Vals = np.array(self.data[columns[2]], dtype="f")
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
        # if self.transform:
        #     padd_len = self.max_seq_prot - len(seq)
        #     seq = self.transform(seq + "." * padd_len, return_tensors='pt')

            
        seq_prot = seq

        # peptide
        seq = self.sequences_pept[idx]
        # if self.transform:
        #     padd_len = self.max_seq_pept - len(seq)
        #     seq = self.transform(seq + "." * padd_len, return_tensors='pt')

        #     # seq_pept_id = seq['input_ids']
        #     # seq_pept_att_mask = seq['attention_mask']
            
        seq_pept = seq

        return seq_prot, seq_pept, affVal

def no_tensor_collate(batch):
    return batch 

def load_datasets(base_path='/home/kunzj'):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D",local_files_only=True)
    
    data_train = Dataset_PEPBI(
        columns=["Prot_Seq", "Pept_Seq", "Energy"],
        data_path=f"{base_path}/BindCraft_uva_internship/data/ppi_affinity_dataset/cosine_sim_scaled/three_way/train.csv",
        transform=tokenizer,
    )
    train_loader = DataLoader(data_train, batch_size=1,collate_fn=no_tensor_collate)



    data_validation = Dataset_PEPBI(
        columns=["Prot_Seq", "Pept_Seq", "Energy"],
        data_path=f"{base_path}/BindCraft_uva_internship/data/ppi_affinity_dataset/cosine_sim_scaled/three_way/validation.csv",
        transform=tokenizer,
    )
    val_loader = DataLoader(data_validation, batch_size=1,collate_fn=no_tensor_collate)

    data_test = Dataset_PEPBI(
        columns=["Prot_Seq", "Pept_Seq", "Energy"],
        data_path=f"{base_path}/BindCraft_uva_internship/data/ppi_affinity_dataset/cosine_sim_scaled/three_way/test.csv",
        transform=tokenizer,
    )
    test_loader = DataLoader(data_test, batch_size=1,collate_fn=no_tensor_collate)
    
    return train_loader, val_loader, test_loader


def load_datasets_our_data(base_path='/home/kunzj'):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D",local_files_only=True)
    
    data_train = Dataset_PEPBI(
        columns=["seq_prot", "seq_pept", "final_score"],
        data_path=f"{base_path}/BindCraft_uva_internship/data/mmpbsa/three_way_split/finished_data_cosine_sim_train.csv",
        transform=tokenizer,
    )
    train_loader = DataLoader(data_train, batch_size=1,collate_fn=no_tensor_collate)



    data_validation = Dataset_PEPBI(
        columns=["seq_prot", "seq_pept", "final_score"],
        data_path=f"{base_path}/BindCraft_uva_internship/data/mmpbsa/three_way_split/finished_data_cosine_sim_validation.csv",
        transform=tokenizer,
    )
    val_loader = DataLoader(data_validation, batch_size=1,collate_fn=no_tensor_collate)

    
    return train_loader, val_loader

def load_dataset_two_way_mmpbsa(base_path='/home/kunzj'):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D",local_files_only=True)
    
    data_train = Dataset_PEPBI(
        columns=["seq_prot", "seq_pept", "final_score"],
        data_path=f"{base_path}/BindCraft_uva_internship/data/mmpbsa/two_way_split/finished_data_cosine_sim_train.csv",
        transform=tokenizer,
    )
    train_loader = DataLoader(data_train, batch_size=1,collate_fn=no_tensor_collate)



    data_validation = Dataset_PEPBI(
        columns=["seq_prot", "seq_pept", "final_score"],
        data_path=f"{base_path}/BindCraft_uva_internship/data/mmpbsa/two_way_split/finished_data_cosine_sim_validation.csv",
        transform=tokenizer,
    )
    val_loader = DataLoader(data_validation, batch_size=1,collate_fn=no_tensor_collate)

    
    return train_loader, val_loader

def load_dataset_one_set_mmpbsa(base_path='/home/kunzj'):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D",local_files_only=True)
    
    data_train = Dataset_PEPBI(
        columns=["seq_prot", "seq_pept", "final_score"],
        data_path=f"{base_path}/BindCraft_uva_internship/data/mmpbsa/finished_data_cosine_sim.csv",
        transform=tokenizer,
    )
    train_loader = DataLoader(data_train, batch_size=1,collate_fn=no_tensor_collate)

    
    return train_loader


def load_dataset_second_iteraion(base_path='/home/kunzj'):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D",local_files_only=True)
    
    data_train = Dataset_PEPBI(
        columns=["seq_prot", "seq_pept", "final_score"],
        data_path=f"{base_path}/BindCraft_uva_internship/data/mmpbsa/second_iteration/finished_data_cosine_sim_second_iteration.csv",
        transform=tokenizer,
    )
    train_loader = DataLoader(data_train, batch_size=1,collate_fn=no_tensor_collate)

    
    return train_loader
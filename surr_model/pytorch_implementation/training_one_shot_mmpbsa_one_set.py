import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from dataset import load_dataset_one_set_mmpbsa
from model_torch import StrippedPredictor
import esm
import pickle

# add that it maps as well the step losses and not just the epochs


def one_step(data_loader, optimizer, model, loss_fn,aux):
    running_loss = 0.0

    for i, data in enumerate(data_loader):
        
        train_prot, train_pept, train_aff_val = data[0]
        
        id_prot = train_prot['input_ids'].to('cuda')
        att_prot = train_prot['attention_mask'].to('cuda')    

        id_pept = train_pept['input_ids'].to('cuda')
        att_pept = train_pept['attention_mask'].to('cuda') 
            
        optimizer.zero_grad()
        outputs = model(id_prot,att_prot,id_pept,att_pept)

        loss = loss_fn(outputs, torch.tensor(train_aff_val).to('cuda'))
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0:
            aux['train_loss_steps'].append(loss.item())
            print(f'Step Loss - {i}: {loss.item()}')
            
    total_loss = running_loss / (i + 1)

    return total_loss, aux


def train():

    
    model_esm2_prot = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D",local_files_only=True)
    model_esm2_pept = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D",local_files_only=True)
    model = StrippedPredictor(model_esm2_prot,model_esm2_pept)
    # tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)

    train_loader = load_dataset_one_set_mmpbsa()

    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.MSELoss()

    EPOCHS = 10

    aux = {"train_loss_epoch": [], 'train_loss_steps':[]}

    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch + 1}:")

        model.train()
        avg_trloss, aux = one_step(train_loader, optimizer, model, loss_fn,aux)

        model.eval()


        aux["train_loss_epoch"].append(avg_trloss)

        print("Loss train {} ".format(avg_trloss))

        model_path = f"model_{epoch + 1}"
        torch.save(
            model.state_dict(),
            f"/home/kunzj/BindCraft_uva_internship/surr_model/pytorch_implementation/models/one_shot_mmpbsa_one_dataset/{model_path}.pt",
        )
    
    with open("/home/kunzj/BindCraft_uva_internship/surr_model/pytorch_implementation/models/one_shot_mmpbsa_one_dataset/model_performance.pkl", "wb") as f:
        pickle.dump(aux, f)




# def test(model_path):
#     torch.cuda.empty_cache()
#     model_esm2 = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
#     model = StrippedPredictor(model_esm2)
#     model.load_state_dict(torch.load(f'/home/kunzj/BindCraft_uva_internship/surr_model/pytorch_implementation/{model_path}.pt',weights_only=True))



if __name__ == '__main__':
    train()
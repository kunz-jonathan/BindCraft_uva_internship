import torch.nn as nn
import torch.nn.functional as F
from peft import LoKrConfig,get_peft_model



class StrippedPredictor(nn.Module):
    def __init__(self, model_prot,model_pept):
        super().__init__()
        peft_config = LoKrConfig(
        r=16,
        alpha=32, 
        target_modules=['key','query','value']
        )
        model_peft_prot = get_peft_model(model_prot, peft_config)
        model_peft_prot.print_trainable_parameters()
        
        self.esm2_prot = model_peft_prot  # PyTorch ESM2 model
        
        model_peft_pept = get_peft_model(model_pept, peft_config)
        model_peft_pept.print_trainable_parameters()
            
        self.esm2_pept = model_peft_pept
        self.prot_dropout = nn.Dropout(p=0.2)
        self.pept_dropout = nn.Dropout(p=0.2)

        self.prot_projection = nn.Linear(640, 320)
        self.pept_projection = nn.Linear(640, 320)

    def forward(self, id_prot,att_prot, id_pept, att_pept ):
        ### PROTEIN ###
        out_prot = self.esm2_prot(id_prot,att_prot)
        emb_prot = out_prot["pooler_output"]


        x_prot = emb_prot.squeeze()
        x_prot = self.prot_dropout(x_prot)
        x_prot = self.prot_projection(x_prot)

        ### PEPTIDE ###
        out_pept = self.esm2_pept(id_pept, att_pept)
        emb_pept = out_pept["pooler_output"]


        x_pept = emb_pept.squeeze()
        x_pept = self.pept_dropout(x_pept)
        x_pept = self.pept_projection(x_pept)

        ### COSINE SIMILARITY HEAD ###
        pred_aff = F.cosine_similarity(x_prot, x_pept,dim=0)

        return pred_aff

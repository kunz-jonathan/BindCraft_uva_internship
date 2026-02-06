import equinox as eqx
import esm
import esm2quinox
import jax.numpy as jnp
import jax.random as jr
import torch
from model_equinox import jax_predictor
from model_torch import StrippedPredictor
from transformers import AutoModel


best_model = 'model_5'
training ='second_iteration_training'
## initializing models
# pytorch
model_esm2_prot_hugging_face = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D",local_files_only=True)
model_esm2_pept_hugging_face = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D",local_files_only=True)
model_torch = StrippedPredictor(
    model_esm2_prot_hugging_face, model_esm2_pept_hugging_face
)

model_torch.load_state_dict(
    torch.load(
        f"/home/kunzj/BindCraft_uva_internship/surr_model/pytorch_implementation/models/{training}/{best_model}.pt",
        weights_only=True,
    )
)
# needs to be switched to before model loading if second training is done
model_torch.esm2_prot = model_torch.esm2_prot.merge_and_unload()
model_torch.esm2_pept = model_torch.esm2_pept.merge_and_unload()

# equinox
model_key = jr.PRNGKey(0)
model_esm2_torch_prot, _ = esm.pretrained.esm2_t30_150M_UR50D()
model_esm2_torch_pept, _ = esm.pretrained.esm2_t30_150M_UR50D()
model_esm2_prot = esm2quinox.from_torch(model_esm2_torch_prot)
model_esm2_pept = esm2quinox.from_torch(model_esm2_torch_pept)
model_aff = jax_predictor(
    model_prot=model_esm2_prot, model_pept=model_esm2_pept, key=model_key
)


### loading the torch weights in equinox ###

## esm2 models
print("Loading ESM-2 Attention QKV Weights using JAX array slicing...")

# --- PROTEIN MODEL (ESM2_PROT) ---

# We will update the weights and biases using eqx.tree_at by setting the slice for layer i.
# JAX arrays are immutable, so we must use tree_at to create a new model_aff instance
# for each layer update.

current_model_aff = model_aff

for i in range(30):
    print(f"Updating ESM2_PROT Layer {i+1}/30...")
    
    # Prot_1_W. Query Weights (Weight requires transpose )
    q_weight_torch = model_torch.esm2_prot.encoder.layer[i].attention.self.query.weight.detach()
    current_model_aff = eqx.tree_at(
        # The function finds the weight array (which contains all 30 layers)
        lambda m: m.esm2_prot.layers.attn.query_proj.weight, 
        current_model_aff,
        # The setter uses .at[i].set() on the existing JAX array
        replace_fn=lambda x: x.at[i].set(jnp.array(q_weight_torch))
    )
    
    # Prot_1_B. Query Bias (Bias requires no transpose)
    q_bias_torch = model_torch.esm2_prot.encoder.layer[i].attention.self.query.bias.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_prot.layers.attn.query_proj.bias, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(q_bias_torch))
    )

    # 3. Key Weights (Weight requires transpose )
    k_weight_torch = model_torch.esm2_prot.encoder.layer[i].attention.self.key.weight.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_prot.layers.attn.key_proj.weight, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(k_weight_torch))
    )
    
    # 4. Key Bias (Bias requires no transpose)
    k_bias_torch = model_torch.esm2_prot.encoder.layer[i].attention.self.key.bias.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_prot.layers.attn.key_proj.bias, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(k_bias_torch))
    )
    
    # 5. Value Weights (Weight requires transpose )
    v_weight_torch = model_torch.esm2_prot.encoder.layer[i].attention.self.value.weight.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_prot.layers.attn.value_proj.weight, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(v_weight_torch))
    )
    
    # 6. Value Bias (Bias requires no transpose)
    v_bias_torch = model_torch.esm2_prot.encoder.layer[i].attention.self.value.bias.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_prot.layers.attn.value_proj.bias, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(v_bias_torch))
    )
    
    # 7. Output Weights 
    out_weight_torch = model_torch.esm2_prot.encoder.layer[i].attention.output.dense.weight.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_prot.layers.attn.output_proj.weight, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(out_weight_torch))
    )
    
    # 8. Output Bias 
    out_bias_torch = model_torch.esm2_prot.encoder.layer[i].attention.output.dense.bias.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_prot.layers.attn.output_proj.bias, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(out_bias_torch))
    )
    
    # 9. Intermediate In Layer Weight 
    inter_in_weight_torch = model_torch.esm2_prot.encoder.layer[i].intermediate.dense.weight.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_prot.layers.linear1.weight, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(inter_in_weight_torch))
    )
    
    # 10. Intermediate In Layer Bias 
    inter_in_bias_torch = model_torch.esm2_prot.encoder.layer[i].intermediate.dense.bias.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_prot.layers.linear1.bias, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(inter_in_bias_torch))
    )
    
    # 10. Intermediate Out Layer Weight 
    inter_out_weight_torch = model_torch.esm2_prot.encoder.layer[i].output.dense.weight.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_prot.layers.linear2.weight, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(inter_out_weight_torch))
    )
    
    # 11. Intermediate Out Layer Bias 
    inter_out_bias_torch = model_torch.esm2_prot.encoder.layer[i].output.dense.bias.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_prot.layers.linear2.bias, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(inter_out_bias_torch))
    )
    
    
    
    # --- PEPTIDE MODEL (ESM2_PEPT) ---
    
    # 7. Query Weights (Weight requires transpose )
    q_weight_torch_pept = model_torch.esm2_pept.encoder.layer[i].attention.self.query.weight.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_pept.layers.attn.query_proj.weight, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(q_weight_torch_pept))
    )
    
    # 8. Query Bias (Bias requires no transpose)
    q_bias_torch_pept = model_torch.esm2_pept.encoder.layer[i].attention.self.query.bias.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_pept.layers.attn.query_proj.bias, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(q_bias_torch_pept))
    )

    # 9. Key Weights (Weight requires transpose )
    k_weight_torch_pept = model_torch.esm2_pept.encoder.layer[i].attention.self.key.weight.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_pept.layers.attn.key_proj.weight, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(k_weight_torch_pept))
    )
    
    # 10. Key Bias (Bias requires no transpose)
    k_bias_torch_pept = model_torch.esm2_pept.encoder.layer[i].attention.self.key.bias.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_pept.layers.attn.key_proj.bias, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(k_bias_torch_pept))
    )

    # 11. Value Weights (Weight requires transpose )
    v_weight_torch_pept = model_torch.esm2_pept.encoder.layer[i].attention.self.value.weight.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_pept.layers.attn.value_proj.weight, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(v_weight_torch_pept))
    )
    
    # 12. Value Bias (Bias requires no transpose)
    v_bias_torch_pept = model_torch.esm2_pept.encoder.layer[i].attention.self.value.bias.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_pept.layers.attn.value_proj.bias, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(v_bias_torch_pept))
    )
    

    # 7. Output Weights 
    out_weight_torch = model_torch.esm2_pept.encoder.layer[i].attention.output.dense.weight.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_pept.layers.attn.output_proj.weight, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(out_weight_torch))
    )
    
    # 8. Output Bias 
    out_bias_torch = model_torch.esm2_pept.encoder.layer[i].attention.output.dense.bias.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_pept.layers.attn.output_proj.bias, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(out_bias_torch))
    )
    
    # 9. Intermediate In Layer Weight 
    inter_in_weight_torch = model_torch.esm2_pept.encoder.layer[i].intermediate.dense.weight.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_pept.layers.linear1.weight, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(inter_in_weight_torch))
    )
    
    # 10. Intermediate In Layer Bias 
    inter_in_bias_torch = model_torch.esm2_pept.encoder.layer[i].intermediate.dense.bias.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_pept.layers.linear1.bias, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(inter_in_bias_torch))
    )
    
    # 10. Intermediate Out Layer Weight 
    inter_out_weight_torch = model_torch.esm2_pept.encoder.layer[i].output.dense.weight.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_pept.layers.linear2.weight, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(inter_out_weight_torch))
    )
    
    # 11. Intermediate Out Layer Bias 
    inter_out_bias_torch = model_torch.esm2_pept.encoder.layer[i].output.dense.bias.detach()
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_pept.layers.linear2.bias, 
        current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(inter_out_bias_torch))
    )

# Assign the final updated model back to model_aff


# --- Inside the Transformer Loop ---
for i in range(30):
    print(f"Updating Layers (Prot & Pept) {i+1}/30...")
    
    # 1. Access the specific layers for convenience
    torch_layer_prot = model_torch.esm2_prot.encoder.layer[i]
    torch_layer_pept = model_torch.esm2_pept.encoder.layer[i]

    # --- ESM2_PROT BLOCK MAPPING ---
    
    # [Your existing Attention and Linear logic here...]
    # Adding missing LayerNorms for Prot:
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_prot.layers.layer_norm1.weight, current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(torch_layer_prot.attention.LayerNorm.weight.detach()))
    )
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_prot.layers.layer_norm1.bias, current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(torch_layer_prot.attention.LayerNorm.bias.detach()))
    )
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_prot.layers.layer_norm2.weight, current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(torch_layer_prot.LayerNorm.weight.detach()))
    )
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_prot.layers.layer_norm2.bias, current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(torch_layer_prot.LayerNorm.bias.detach()))
    )

    # --- ESM2_PEPT BLOCK MAPPING ---

    # [Your existing Attention and Linear logic here...]
    # Adding missing LayerNorms for Pept:
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_pept.layers.layer_norm1.weight, current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(torch_layer_pept.attention.LayerNorm.weight.detach()))
    )
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_pept.layers.layer_norm1.bias, current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(torch_layer_pept.attention.LayerNorm.bias.detach()))
    )
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_pept.layers.layer_norm2.weight, current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(torch_layer_pept.LayerNorm.weight.detach()))
    )
    current_model_aff = eqx.tree_at(
        lambda m: m.esm2_pept.layers.layer_norm2.bias, current_model_aff,
        replace_fn=lambda x: x.at[i].set(jnp.array(torch_layer_pept.LayerNorm.bias.detach()))
    )


print("Conversion Complete.")
# Assign the final updated model back to model_aff
model_aff = current_model_aff
print("ESM-2 Attention QKV weights successfully updated using array slicing and transpose.")

#=========================================================================================#

## Pooler and Projection layers
print("Loading Pooler Weights...")
model_aff = eqx.tree_at(
    lambda m: m.pooler_layer_prot.weight, 
    model_aff,
    # Access the weight, detach, transpose, convert to JAX array
    jnp.array(model_torch.esm2_prot.pooler.dense.weight.detach()) 
)
model_aff = eqx.tree_at(
    lambda m: m.pooler_layer_prot.bias, 
    model_aff,
    jnp.array(model_torch.esm2_prot.pooler.dense.bias.detach())
)

# 2. Peptide Pooler Layer
model_aff = eqx.tree_at(
    lambda m: m.pooler_layer_pept.weight, 
    model_aff,
    jnp.array(model_torch.esm2_pept.pooler.dense.weight.detach())
)
model_aff = eqx.tree_at(
    lambda m: m.pooler_layer_pept.bias, 
    model_aff,
    jnp.array(model_torch.esm2_pept.pooler.dense.bias.detach())
)


#=========================================================================================#

## PROJECTION LAYERS
print("Loading Projection Weights...")

# 3. Protein Projection Layer
model_aff = eqx.tree_at(
    lambda m: m.prot_projection.weight, 
    model_aff,
    jnp.array(model_torch.prot_projection.weight.detach())
)
model_aff = eqx.tree_at(
    lambda m: m.prot_projection.bias, 
    model_aff,
    jnp.array(model_torch.prot_projection.bias.detach())
)

# 4. Peptide Projection Layer
model_aff = eqx.tree_at(
    lambda m: m.pept_projection.weight, 
    model_aff,
    jnp.array(model_torch.pept_projection.weight.detach())
)
model_aff = eqx.tree_at(
    lambda m: m.pept_projection.bias, 
    model_aff,
    jnp.array(model_torch.pept_projection.bias.detach())
)

print("Pooler and Projection weights conversion complete.")

## saving model
inference_model = eqx.nn.inference_mode(model_aff)
#inference_model = eqx.Partial(inference_model, state=model_state)
eqx.tree_serialise_leaves("./models/equinox/second_iteration/model_affinity_final.eqx", inference_model)

import jax
import jax.numpy as jnp
import equinox as eqx
import esm2quinox
import esm
from colabdesign.af.alphafold.common import residue_constants

def add_ste_seq_loss(self, loss_weight: float) -> None:
    key = jax.random.PRNGKey(42)
    
    # load models
    model_torch, _ = esm.pretrained.esm2_t6_8M_UR50D()
    model = esm2quinox.from_torch(model_torch)
    
    # get the correct embedding
    full_embed_weights = model.logit_head.linear2.weight
    colab_aa_dict = residue_constants.restype_order
    bindcraft_order = "".join(list(colab_aa_dict.keys()))
    esm_indices = jnp.array([esm2quinox._esm2._alphabet[aa] for aa in bindcraft_order])
    aligned_weights = full_embed_weights[esm_indices, :] 
    cls_embed = full_embed_weights[esm2quinox._esm2._alphabet["^"], :]
    eos_embed = full_embed_weights[esm2quinox._esm2._alphabet["$"], :]
    
    
    
    def loss_fn(opt, aux: dict) -> float:
        seq = aux['seq']['hard']
        
        seq_embeds = jnp.matmul(seq, aligned_weights)
        
        cls_expanded = jnp.expand_dims(cls_embed, (0,1))
        eos_expanded = jnp.expand_dims(eos_embed, (0,1))
        inputs_embeds = jnp.concatenate(
            [cls_expanded, seq_embeds, eos_expanded], axis=1
        )  
        inputs_embeds = inputs_embeds.squeeze(0)  
        print("inputs_embeds shape:", inputs_embeds.shape)  # Debugging line
        
        is_pad = jnp.zeros(
        (inputs_embeds.shape[0],), dtype=jnp.bool_)
        
        dynamic_layers, static_layer = eqx.partition(model.layers, eqx.is_array)

        def f(x_carry, dynamic_layer):
            layer = eqx.combine(dynamic_layer, static_layer)
            x_out = layer(x_carry, is_pad=is_pad)  # Must feed is_pad array
            return x_out, None

        x, _ = jax.lax.scan(f, inputs_embeds, xs=dynamic_layers)

        # G. Apply original vmapped norms and prediction steps
        hidden = jax.vmap(model.layer_norm)(x)
        plm_logits = jax.vmap(model.logit_head)(hidden)  # Shape: [12, 33]

        # H. Strip context frames and isolate optimization target loss
        seq_plm_logits = plm_logits[1:-1, :]
        
    
        loss_term = - jnp.mean(seq_plm_logits) * opt.get('hard',0)
        
        return {"seq_loss": loss_term}
    
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["seq_loss"] = loss_weight
import jax
import jax.numpy as jnp
from .surr_model import init_surr_model




def add_seq_loss(self, loss_weight: float) -> None:
    """
    wrapper around loss function that registers seq_loss in af_model

    Args:
        self (alf_model)
        loss_weight (flaot)

    Returns:
        None

    """
    model, params_restored, rng = init_surr_model()
    

    def loss_fn( aux: dict)->float:
        """
        seq.-loss function
        
        Args:
            aux (dict): af2 dictionary containing all relevant information
            
        Returns:
            seq_loss (float): loss value
        
        """
        
        #jax.debug.print("aux['seq']['pseudo'] = {}",aux['seq']['pseudo'])
        # seq = jax.device_get(aux['seq']['pseudo'])
        

        pred_aff = model.apply( params_restored,rng,aux['seq']['pseudo'])
        #pred_aff = surr_model.apply(seq)
 
        
        seq_loss = pred_aff[0,0] # actually prob. not relevant having an additional relu fnc. here as it should be in surrogate model right?
        return {"seq_loss": seq_loss}
        

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["seq_loss"] = loss_weight

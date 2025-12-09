from ..blockBase import Block
from typing import override
from ..config import Config
from ..utils.debug import check_tensor_debug
from ..utils.cu_kulfan_airfoil import cuKulfanAirfoil, get_aero_from_kulfan_parameters_cuda
from typing import Dict, Any
import torch
from math import sqrt
from functorch import jacrev, vmap


class NeuralFoilSampling(Block):
    @override
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.run.device)
        nfConfig = self.config.neuralFoilSampling

        n_1d = int(sqrt(nfConfig.n_samples))
        aoa_1d = torch.linspace(nfConfig.AoA_min, nfConfig.AoA_max, n_1d, device=self.device)
        re_1d  = torch.linspace(nfConfig.Re_min, nfConfig.Re_max, n_1d, device=self.device)

        aoa, re = torch.meshgrid(aoa_1d, re_1d, indexing="ij")
        self.alpha_batch = aoa.reshape(-1)
        self.Re_batch = re.reshape(-1)

        self.last_airfoil = None

        def aero_fn(upper, lower, LE, TE, alpha, Re):
            B = 1
            kulfan_batch = {
                "upper_weights_cuda": upper.unsqueeze(0).expand(B, -1),
                "lower_weights_cuda": lower.unsqueeze(0).expand(B, -1),
                "leading_edge_weight_cuda": LE.expand(B),
                "TE_thickness_cuda": TE.expand(B),
            }
            aero = get_aero_from_kulfan_parameters_cuda(
                kulfan_batch, alpha.unsqueeze(0), Re.unsqueeze(0),
                device=upper.device,
                model_size=self.config.neuralFoilSampling.neuralFoil_size
            )

            return torch.stack([aero["CL"], aero["CD"], aero["CM"]], dim=1)

        self._jac_fn = vmap(
            jacrev(aero_fn, argnums=(0, 1, 2, 3)),
            in_dims=(None, None, None, None, 0, 0) # 0 means that it's batched
        )

    @override
    def forward(self, downstream_info: Dict[str, Any]) -> Dict[str, Any]:
        self.last_airfoil: cuKulfanAirfoil = downstream_info["airfoil"]
        B = self.alpha_batch.shape[0]

        self.upper = self.last_airfoil.upper_weights_cuda
        self.lower = self.last_airfoil.lower_weights_cuda
        self.LE    = self.last_airfoil.leading_edge_weight_cuda
        self.TE    = self.last_airfoil.TE_thickness_cuda

        upper_batch = self.upper.unsqueeze(0).expand(B, -1).contiguous()
        lower_batch = self.lower.unsqueeze(0).expand(B, -1).contiguous()
        LE_batch    = self.LE.expand(B).contiguous()
        TE_batch    = self.TE.expand(B).contiguous()

        kulfan_batch = {
            "upper_weights_cuda": upper_batch,
            "lower_weights_cuda": lower_batch,
            "leading_edge_weight_cuda": LE_batch,
            "TE_thickness_cuda": TE_batch,
        }

        self._last_aero_coeff = get_aero_from_kulfan_parameters_cuda(
            kulfan_batch,
            self.alpha_batch,
            self.Re_batch,
            device=self.device,
            model_size=self.config.neuralFoilSampling.neuralFoil_size,
        )

        return {
            "alpha": self.alpha_batch,
            "Re": self.Re_batch,
            "CL": self._last_aero_coeff["CL"].detach(),
            "CD": self._last_aero_coeff["CD"].detach(),
            "CM": self._last_aero_coeff["CM"].detach(),
        }

    @override
    def backward(self, upstream_grads: Dict[str, Any]) -> torch.Tensor:
        dN_dupper, dN_dlower, dN_dLE, dN_dTE = self._jac_fn(
            self.upper,
            self.lower,
            self.LE,
            self.TE,
            self.alpha_batch,
            self.Re_batch
        )
        
        #check_tensor_debug(dN_dupper, "dN_dupper")
        #check_tensor_debug(dN_dlower, "dN_dlower")
        #check_tensor_debug(dN_dLE, "dN_dLE")
        #check_tensor_debug(dN_dTE, "dN_dTE")
                
        dN_dupper = dN_dupper.squeeze(1)
        dN_dlower = dN_dlower.squeeze(1)
        dN_dLE = dN_dLE.squeeze(1)
        dN_dTE = dN_dTE.squeeze(1)
                
                
        dN_dw = torch.cat([
            dN_dupper,
            dN_dlower,
            dN_dLE.unsqueeze(-1),
            dN_dTE.unsqueeze(-1)
        ], dim=-1)
        
        df_dphi = upstream_grads["training_df_dphi"]
        df_dot = torch.einsum('bcf,bcg->bfg', df_dphi, df_dphi)
        obj_hessian_phi = df_dot.sum(dim=0)
        
        dfdN_dot = torch.einsum('bcf,bcg->bfg', df_dphi, dN_dw)
        obj_dw_dphi = dfdN_dot.sum(0)
        
                
        dphi_dw = torch.linalg.solve(obj_hessian_phi, -obj_dw_dphi)
              
        stream_dJ_dphi = upstream_grads["stream_dJ_dphi"]

        dJ_dw = stream_dJ_dphi @ dphi_dw      
        
                  
        return {"dJ_dw":dJ_dw}

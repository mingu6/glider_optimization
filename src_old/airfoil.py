import time
from typing import Dict, Tuple, List
from dataclasses import dataclass

import aerosandbox as asb
import aerosandbox.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebvander2d


@dataclass
class AirfoilConfig:
    """Configuration for Kulfan airfoil parameters."""
    upper_weights: np.ndarray
    lower_weights: np.ndarray
    leading_edge_weight: float = 0.02
    te_thickness: float = 0.0
    n1: float = 0.5
    n2: float = 1.0


@dataclass
class FlowConditions:
    """Flow conditions for aerodynamic analysis."""
    rho: float = 1.225  # kg/m³
    mu: float = 1.81e-5  # Pa·s
    chord: float = 1.0  # m
    mach: float = 0.2


class AirfoilAnalyzer:
    """Analyzes airfoil aerodynamics using Chebyshev polynomial regression."""
    
    def __init__(
        self,
        airfoil_config: AirfoilConfig,
        flow_conditions: FlowConditions,
        chebyshev_degree: int = 17,
        lambda_reg: float = 0.5
    ):
        self.airfoil_config = airfoil_config
        self.flow_conditions = flow_conditions
        self.chebyshev_degree = chebyshev_degree
        self.lambda_reg = lambda_reg
        
        self.airfoil = self._create_airfoil()
        self.coeffs_cl = None
        self.coeffs_cd = None
        self.coeffs_cm = None
        self.v_min = None
        self.v_max = None
        self.aoa_min = None
        self.aoa_max = None
    
    def _create_airfoil(self) -> asb.KulfanAirfoil:
        """Create Kulfan airfoil from configuration."""
        cfg = self.airfoil_config
        return asb.KulfanAirfoil(
            "Supercool",
            upper_weights=cfg.upper_weights,
            lower_weights=cfg.lower_weights,
            leading_edge_weight=cfg.leading_edge_weight,
            TE_thickness=cfg.te_thickness,
            N1=cfg.n1,
            N2=cfg.n2
        )
    
    def visualize_airfoil(self, figsize: Tuple[int, int] = (8, 3)) -> None:
        """Plot airfoil geometry."""
        fig, ax = plt.subplots(figsize=figsize)
        self.airfoil.draw()
    
    def generate_dataset(
        self,
        velocities: np.ndarray,
        angles_of_attack: np.ndarray
    ) -> pd.DataFrame:
        """
        Generate aerodynamic coefficient dataset.
        
        Args:
            velocities: Array of velocities (m/s)
            angles_of_attack: Array of angles of attack (degrees)
            
        Returns:
            DataFrame with V, AoA, CL, CD, CM columns
        """
        dataset: List[Dict] = []
        fc = self.flow_conditions
        
        print(f"Generating dataset: {len(velocities)} velocities x "
              f"{len(angles_of_attack)} AoA = {len(velocities) * len(angles_of_attack)} points")
        
        for v in velocities:
            re = fc.rho * v * fc.chord / fc.mu
            
            for aoa in angles_of_attack:
                res = self.airfoil.get_aero_from_neuralfoil(
                    alpha=aoa,
                    Re=re,
                    mach=fc.mach,
                )
                
                dataset.append({
                    "V": float(v),
                    "AoA": float(aoa),
                    "CL": float(np.atleast_1d(res["CL"])[0]),
                    "CD": float(np.atleast_1d(res["CD"])[0]),
                    "CM": float(np.atleast_1d(res["CM"])[0])
                })
        
        return pd.DataFrame(dataset)
    
    @staticmethod
    def scale_to_chebyshev_domain(
        x: np.ndarray,
        x_min: float,
        x_max: float
    ) -> np.ndarray:
        """Scale input to [-1, 1] Chebyshev domain."""
        return 2 * (x - x_min) / (x_max - x_min) - 1
    
    @staticmethod
    def ridge_solve(X: np.ndarray, y: np.ndarray, lambda_reg: float) -> np.ndarray:
        """
        Solve ridge regression: minimize ||Xw - y||² + λ||w||².
        
        Args:
            X: Design matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            lambda_reg: Regularization parameter
            
        Returns:
            Coefficient vector (n_features,)
        """
        return np.linalg.solve(
            X.T @ X + lambda_reg * np.eye(X.shape[1]),
            X.T @ y
        )
    
    def fit(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Fit Chebyshev polynomial models to aerodynamic data.
        
        Args:
            df: DataFrame with V, AoA, CL, CD, CM columns
            
        Returns:
            Dictionary with training metrics
        """
        # Extract and convert data (ensure proper scalar conversion)
        v = df['V'].to_numpy(dtype=np.float64)
        aoa = df['AoA'].to_numpy(dtype=np.float64)
        cl = df['CL'].to_numpy(dtype=np.float64)
        cd = df['CD'].to_numpy(dtype=np.float64)
        cm = df['CM'].to_numpy(dtype=np.float64)
        
        # Store scaling parameters
        self.v_min, self.v_max = v.min(), v.max()
        self.aoa_min, self.aoa_max = aoa.min(), aoa.max()
        
        # Scale to Chebyshev domain [-1, 1]
        v_scaled = self.scale_to_chebyshev_domain(v, self.v_min, self.v_max)
        aoa_scaled = self.scale_to_chebyshev_domain(aoa, self.aoa_min, self.aoa_max)
        
        # Build Chebyshev design matrix
        deg = self.chebyshev_degree
        X_cheb = chebvander2d(v_scaled, aoa_scaled, [deg, deg])
        
        print(f"Chebyshev design matrix shape: {X_cheb.shape}")
        print(f"Polynomial degree: {deg}")
        print(f"Regularization λ: {self.lambda_reg}")
        
        # Train models
        start_time = time.time()
        self.coeffs_cl = self.ridge_solve(X_cheb, cl, self.lambda_reg)
        self.coeffs_cd = self.ridge_solve(X_cheb, cd, self.lambda_reg)
        self.coeffs_cm = self.ridge_solve(X_cheb, cm, self.lambda_reg)
        training_time = (time.time() - start_time) * 1000
        
        # Evaluate fit quality
        cl_pred = X_cheb @ self.coeffs_cl
        cd_pred = X_cheb @ self.coeffs_cd
        cm_pred = X_cheb @ self.coeffs_cm
        
        metrics = {
            'training_time_ms': training_time,
            'cl_mae': np.mean(np.abs(cl - cl_pred)),
            'cd_mae': np.mean(np.abs(cd - cd_pred)),
            'cm_mae': np.mean(np.abs(cm - cm_pred)),
            'cl_rmse': np.sqrt(np.mean((cl - cl_pred)**2)),
            'cd_rmse': np.sqrt(np.mean((cd - cd_pred)**2)),
            'cm_rmse': np.sqrt(np.mean((cm - cm_pred)**2)),
        }
        
        return metrics
    
    def predict(self, v: float, aoa: float) -> Dict[str, float]:
        """
        Predict aerodynamic coefficients at given conditions.
        
        Args:
            v: Velocity (m/s)
            aoa: Angle of attack (degrees)
            
        Returns:
            Dictionary with CL, CD, CM predictions
        """
        if self.coeffs_cl is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        v_scaled = float(self.scale_to_chebyshev_domain(
            np.array([v]), self.v_min, self.v_max
        )[0])
        aoa_scaled = float(self.scale_to_chebyshev_domain(
            np.array([aoa]), self.aoa_min, self.aoa_max
        )[0])
        
        # Build feature vector
        deg = self.chebyshev_degree
        X = chebvander2d(
            np.array([v_scaled]), 
            np.array([aoa_scaled]), 
            [deg, deg]
        )
        
        return {
            'CL': float((X @ self.coeffs_cl)[0]),
            'CD': float((X @ self.coeffs_cd)[0]),
            'CM': float((X @ self.coeffs_cm)[0])
        }


def main():
    airfoil_config = AirfoilConfig(
        upper_weights=np.array([0.1, 0.15, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01]),
        lower_weights=np.array([-0.05, -0.05, -0.04, -0.03, -0.02, -0.01, -0.005, 0.0]),
    )
    
    analyzer = AirfoilAnalyzer(
        airfoil_config=airfoil_config,
        flow_conditions=FlowConditions(),
        chebyshev_degree=17,
        lambda_reg=0.5
    )
    
    analyzer.visualize_airfoil()
    
    velocities = np.linspace(5, 15, 5)
    angles_of_attack = np.linspace(-5, 15, 11)
    df = analyzer.generate_dataset(velocities, angles_of_attack)
    
    print(f"\nDataset size: {len(df)} samples")
    print(f"Velocity range: [{velocities.min():.1f}, {velocities.max():.1f}] m/s")
    print(f"AoA range: [{angles_of_attack.min():.1f}, {angles_of_attack.max():.1f}]°")
    
    print("\n" + "="*60)
    print("Training Chebyshev Polynomial Models")
    print("="*60)
    metrics = analyzer.fit(df)
    
    print(f"\nTraining time: {metrics['training_time_ms']:.3f} ms")
    print("\nModel Performance (Mean Absolute Error):")
    print(f"  CL: {metrics['cl_mae']:.6f}")
    print(f"  CD: {metrics['cd_mae']:.6f}")
    print(f"  CM: {metrics['cm_mae']:.6f}")
    print("\nModel Performance (RMSE):")
    print(f"  CL: {metrics['cl_rmse']:.6f}")
    print(f"  CD: {metrics['cd_rmse']:.6f}")
    print(f"  CM: {metrics['cm_rmse']:.6f}")
    
    print("\n" + "="*60)
    print("Example Prediction")
    print("="*60)
    test_v, test_aoa = 10.0, 5.0
    pred = analyzer.predict(test_v, test_aoa)
    print(f"Conditions: V={test_v} m/s, AoA={test_aoa}°")
    print(f"Predictions: CL={pred['CL']:.4f}, CD={pred['CD']:.4f}, CM={pred['CM']:.4f}")


if __name__ == "__main__":
    main()
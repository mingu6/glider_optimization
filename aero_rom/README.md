# aero_rom

Reduced–order aerodynamic model generator for a small UAV wing + elevator.

This project:

- Builds a lifting–line (Weissinger–L) model for the wing and elevator.
- Uses NeuralFoil (via AeroSandbox) to get 2D airfoil polars.
- Sweeps angle of attack and airspeed over a prescribed grid.
- Produces aerodynamic coefficient surfaces (CL, CD, CM).
- Fits regular–grid interpolants (RGI) to those surfaces.
- Saves the resulting ROMs (`.npz`) for later use in simulations or control.

ex:
python run_from_config.py data/config.json
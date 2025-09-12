# Newton-Lightwheel 
This repository contains the scene setup, assets, and tools used for particle-based simulations with Newton. It includes USD scene files, textures, Houdini setups, and Python scripts to help load and run the environment.  

### Collisions
The collision geometry in the scene is currently divided into two parts:  
- One part is passed to the **Mujoco solver**.  
- The other part is used by the **MPM solver**.  
Collision updates are handled separately for these two parts.  

### Particles
The USD files stored in this repository contain around **1.5M particles**. Both the particle count and shapes can be adjusted in **Houdini**.  

### Assets
All assets (textures, USDA files) are located in the **Assets** folder. For now, only a preliminary organization has been done.  

### Newton
The Newton version currently in use:  
[https://github.com/gdaviet/newton/tree/feat/mpm_multi_mat](https://github.com/gdaviet/newton/tree/feat/mpm_multi_mat)

### Houdini
Current Houdini Version to load the Hip file is 20.5.487 and 
the Hip file are placed in the root directory of this repository:  
- The **hda** folder contains the required Houdini Digital Assets.  
- The **hip** files mainly process terrain, particles, and all Newton-related attributes and key values.  

### Scene Files
The scene file can be edited in any USD-compatible tool (**IsaacSim, Blender, etc.**).  

A Python file is also provided for loading the scene. It can be run in the corresponding Newton environment. The loading code is still at an early stage and will be updated further.  

### Usage
1. **Open the USD scene**  
   - Load the scene file directly in **IsaacSim**, **Blender**, or any other USD-supported tool for layout and inspection.  

2. **Run with Newton**  
   - Use the provided Python script to load the scene into the Newton environment.  
   - Example:  
     ```bash
     uv run mpm_test.py
     ```  
3. **Edit with Houdini**  
   - Use the `.hip` files to adjust terrain, particle counts, or Newton-related attributes.  
   - Export the modified assets (USD/USDA) back into the **Assets** folder for reuse.  

### `_newton__` Static Collision Fields

| Name | Type | Mapping to Newton |
|------|------|-------------|
| `NewtonCollisionAPI` | bool | Toggle collision on or off |
| `_newton__contact_ka` | float | ShapeConfig.ka |
| `_newton__contact_ke` | float | ShapeConfig.ke |
| `_newton__contact_kd` | float | ShapeConfig.kd |
| `_newton__contact_kf` | float | ShapeConfig.kf |
| `_newton__contact_thickness` | float | ShapeConfig.thickness |
| `_newton__material_staticFriction` | float | ShapeConfig.mu |
| `_newton__material_dynamicFriction` | float | ShapeConfig.mu |
| `_newton__material_restitution` | float | ShapeConfig.restitution |
| `_newton__material_density` | float | ShapeConfig.density |
| `_newton__collision_group` | int | ShapeConfig.collision_group |
| `_newton__collision_isVisible` | bool | ShapeConfig.is_visible |
| `_newton__collision_approximation` | bool | Toggle Mesh approximation on or off |
| `_newton__collision_approximationMethod` | string | Mesh approximation method name (e.g., “coacd”) |

### `_newton__` Particle Attribute Fields

| Name | Type | Mapping to Newton |
|------|------|-------------|
| `_newton__particle_qd` | vec3 | Model.particle_qd |
| `_newton__particle_mass` | float | Model.particle_mass |
| `_newton__particle_radius` | float | Model.particle_radius |
| `_newton__particle_flags` | float | Model.particle_flags |
| `_newton__particle_material_friction` | float | Model.particle_mu|
| `_newton__particle_material_youngModulus` | float | ImplicitMPMOptions.young_modulus|
| `_newton__particle_material_damping` | float | ImplicitMPMOptions.damping|
| `_newton__particle_material_poissonRatio` | float | ImplicitMPMOptions.poisson_ratio|
| `_newton__particle_material_hardening` | float | ImplicitMPMOptions.hardening|
| `_newton__particle_material_yieldPressure` | float | ImplicitMPMOptions.yield_pressure|
| `_newton__particle_material_yieldStress` | float | ImplicitMPMOptions.yield_stress|
| `_newton__particle_material_tensileYieldRatio` | float | ImplicitMPMOptions.tensile_yield_ratio |

### Future Work
- **Lighting and Materials**: Explore more realistic lighting models and textures. For presentation purposes, consider offline rendering as well.  
- **Performance Optimization**: Improve real-time performance. With reduced particle counts and higher-end hardware, the goal is to reach ~20 FPS.  
- **Particle-to-Mesh Conversion**: Investigate converting mud or snow particles into mesh representations.  
<!-- - **Additional Scene Elements**: Add details such as leaves, grass, or rocks to enhance realism.   -->
<!-- - **Rendering Enhancements**: Experiment with techniques like subsurface scattering (SSS) in RTX for better visual fidelity.   -->


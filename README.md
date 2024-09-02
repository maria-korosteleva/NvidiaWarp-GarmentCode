# NVIDIA Warp -- version for GarmentCodeData

This package is the fork of [NVIDIA Warp](https://github.com/NVIDIA/warp) based on Warp v.1.0.0-beta.6. We implemented the changes to the cloth simulation as introduced in the [GarmentCodeData](https://igl.ethz.ch/projects/GarmentCodeData/) project.

This simulator version is used by [PyGarment v2.0.0+](https://github.com/maria-korosteleva/GarmentCode) that implements [GarmentCode](https://igl.ethz.ch/projects/garmentcode/) and [GarmentCodeData](https://igl.ethz.ch/projects/GarmentCodeData/).

## Warp Overview

Warp is a Python framework for writing high-performance simulation and graphics code. Warp takes
regular Python functions and JIT compiles them to efficient kernel code that can run on the CPU or GPU.

Warp is designed for spatial computing and comes with a rich set of primitives that make it easy to write
programs for physics simulation, perception, robotics, and geometry processing. In addition, Warp kernels
are differentiable and can be used as part of machine-learning pipelines with frameworks such as PyTorch and JAX.

Please refer to the project [Documentation](https://nvidia.github.io/warp/) for API and language reference and [CHANGELOG.md](./CHANGELOG.md) for release history.

## Changes introduced for GarmentCodeData

All the changes are implemented for the XPBD solver. 

* **[Self-collisions]** Implemented point-triangle and edge-edge self-collisions detection and resolution for cloth simulation following the works of [Cincotti C.](https://carmencincotti.com/2022-11-21/cloth-self-collisions/) and [Lewin, C.](https://www.semanticscholar.org/paper/Cloth-Self-Collision-with-Predictive-Contacts-Lewin/211d9e302549c1d6ae645b99a70fd9dca417c85f).
* **[Attachment]** Added support for equality and inequality-based attachment constraints used for, e.g., fixing the skirt placement on the waist area
* **[Body-part drag for initial collision resolution]** Introduced body-part based collision resolution constraint that drags intersecting garment panels towards their corresponding body parts untill the collisions are resolved. This helps with resolving initialization issues.
* **[Body model collision constraint]** Introduces body collision constraints that pushed the cloth parts found inside the body model outside.
* **[Edge-based ray intersection]** Additional mesh intersection query: `mesh_query_edge()` and a placeholder for `adj_mesh_query_edge()` -- similar to `mesh_query_ray()` function, but with the length limit on the ray

### Fixes
* In `apply_particle_deltas()` in XPBD apply `max_velocity` constraint to the point location for greater stability. 

See [GarmentCodeData](https://igl.ethz.ch/projects/GarmentCodeData/) paper for more details.

## Installing

To use this version of NVIDIA Warp, please, follow the steps for manual installation below. The following tools are required:

* Microsoft Visual Studio 2019 upwards (Windows)
* GCC 7.2 upwards (Linux)
* CUDA Toolkit 11.5 or higher
* [Git LFS](https://git-lfs.github.com/) installed

After cloning the repository, users should run:

    python build_lib.py

This will generate the `warp.dll` / `warp.so` core library respectively. When building manually users should ensure that their `CUDA_PATH` environment variable is set, otherwise Warp will be built without CUDA support. Alternatively, the path to the CUDA toolkit can be passed to the build command as `--cuda_path="..."`. After building, the Warp package should be installed using:

    pip install -e .

This ensures that subsequent modifications to the library will be reflected in the Python package.

If you are cloning from Windows, please first ensure that you have enabled "Developer Mode" in Windows settings and symlinks in git:

    git config --global core.symlinks true

This will ensure symlinks inside ``exts/omni.warp.core`` work upon cloning.


## Learn More

Please see the following resources for additional background on Warp:

* [GTC 2022 Presentation](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41599)
* [GTC 2021 Presentation](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31838)
* [SIGGRAPH Asia 2021 Differentiable Simulation Course](https://dl.acm.org/doi/abs/10.1145/3476117.3483433)

The underlying technology in Warp has been used in a number of research projects at NVIDIA including the following publications:

* Accelerated Policy Learning with Parallel Differentiable Simulation - Xu, J., Makoviychuk, V., Narang, Y., Ramos, F., Matusik, W., Garg, A., & Macklin, M. [(2022)](https://short-horizon-actor-critic.github.io)
* DiSECt: Differentiable Simulator for Robotic Cutting - Heiden, E., Macklin, M., Narang, Y., Fox, D., Garg, A., & Ramos, F [(2021)](https://github.com/NVlabs/DiSECt)
* gradSim: Differentiable Simulation for System Identification and Visuomotor Control - Murthy, J. Krishna, Miles Macklin, Florian Golemo, Vikram Voleti, Linda Petrini, Martin Weiss, Breandan Considine et al. [(2021)](https://gradsim.github.io)

## Citing

If you use Warp in your research please use the following citation:

```bibtex
@misc{warp2022,
title= {Warp: A High-performance Python Framework for GPU Simulation and Graphics},
author = {Miles Macklin},
month = {March},
year = {2022},
note= {NVIDIA GPU Technology Conference (GTC)},
howpublished = {\url{https://github.com/nvidia/warp}}
}
```

If you use this version of framework, additionally cite GarmentCodeData: 

```bibtex
@inproceedings{GarmentCodeData:2024,
  author = {Korosteleva, Maria and Kesdogan, Timur Levent and Kemper, Fabian and Wenninger, Stephan and Koller, Jasmin and Zhang, Yuhan and Botsch, Mario and Sorkine-Hornung, Olga},
  title = {{GarmentCodeData}: A Dataset of 3{D} Made-to-Measure Garments With Sewing Patterns},
  booktitle={Computer Vision -- ECCV 2024},
  year = {2024},
  keywords = {sewing patterns, garment reconstruction, dataset},
}
```

## License

Warp is provided under the NVIDIA Source Code License (NVSCL), please see [LICENSE.md](./LICENSE.md) for full license text. Note that the license currently allows only non-commercial use of this code.

> Later revisions of the license expand the use of warp to commertial use. We aim to share the changes related to GarmentCodeData introduced in this repo, so we follow the NVIDIA licensing decision. 

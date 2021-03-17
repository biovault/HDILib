Linux & Macos (Travis) | Windows (Appveyor)
--- | ---
[![Build Status](https://travis-ci.com/biovault/HDILib.svg?branch=master)](https://travis-ci.com/biovault/HDILib) | [![Build status](https://ci.appveyor.com/api/projects/status/xtd9ee63fukd462p?svg=true)](https://ci.appveyor.com/project/bldrvnlw/hdilib)

Currently the following build matrix is performed

OS | Architecture | Compiler
--- | --- | ---
Windows | x64 | MSVC 2017
Linux | x86_64 | gcc 9
Macos | x86_64 | clang 10


[![DOI](https://zenodo.org/badge/100361974.svg)](https://zenodo.org/badge/latestdoi/100361974)


# HDILib: High Dimensional Inspector Library
HDILib is a library for the scalable analysis of large and high-dimensional data.
It contains scalable manifold-learning algorithms, visualizations and visual-analytics frameworks.
HDILib is implemented in C++, OpenGL and JavaScript.
It is developed within a joint collaboration between the [Computer Graphics & Visualization](https://graphics.tudelft.nl/) group at the [Delft University of Technology](https://www.tudelft.nl) and the [Division of Image Processing (LKEB)](https://www.lumc.nl/org/radiologie/research/LKEB/) at the [Leiden Medical Center](https://www.lumc.nl/).

## Authors
- [Nicola Pezzotti](http://nicola17.github.io/) initiated the HDI project, developed the A-tSNE and HSNE algorithms and implemented most of the visualizations and frameworks.
- [Thomas Höllt](https://www.thomashollt.com/) ported the library to MacOS.

## Used
HDI is used in the following projects:
- [Cytosplore](https://www.cytosplore.org/): interactive system for understanding how the immune system works
- [Brainscope](http://www.brainscope.nl/brainscope): web portal for fast,
interactive visual exploration of the [Allen Atlases](http://www.brain-map.org/) of the adult and developing human brain
transcriptome
- [DeepEyes](https://graphics.tudelft.nl/Publications-new/2018/PHVLEV18/): progressive analytics system for designing deep neural networks

## Reference
Reference to cite when you use HDI in a research paper:

```
@inproceedings{Pezzotti2016HSNE,
  title={Hierarchical stochastic neighbor embedding},
  author={Pezzotti, Nicola and H{\"o}llt, Thomas and Lelieveldt, Boudewijn PF and Eisemann, Elmar and Vilanova, Anna},
  journal={Computer Graphics Forum},
  volume={35},
  number={3},
  pages={21--30},
  year={2016}
}
@article{Pezzotti2017AtSNE,
  title={Approximated and user steerable tsne for progressive visual analytics},
  author={Pezzotti, Nicola and Lelieveldt, Boudewijn PF and van der Maaten, Laurens and H{\"o}llt, Thomas and Eisemann, Elmar and Vilanova, Anna},
  journal={IEEE transactions on visualization and computer graphics},
  volume={23},
  number={7},
  pages={1739--1752},
  year={2017}
}
```

## Building

### GIT Cloning 
When cloning the repo be aware that it includes submodules. With the latest git versions you shoule use the following command:

```
git clone --recurse-submodules https://github.com/biovault/HDILib.git
```

### Ubuntu

On **Ubuntu 16.04** you can build and install HDI by running the following commands

```bash
./scripts/install-dependencies.sh
mkdir build
cd build
cmake  -DCMAKE_BUILD_TYPE=Release ..
make -j 8
sudo make install
```

### Windows

On **Windows** use CMake

```
cmake .. -G "Visual Studio 15 2017 Win64" -DCMAKE_BUILD_TYPE=Release -DHDILIB_BUILD_WITH_CONAN=ON
```      
    (*Note: this assumes that the build dir is one level down from the project root.
    The default of HDILIB_BUILD_WITH_CONAN is OFF*)
 - If all goes well Conan will have installed the dependencies in its cache and 
 created the required defines for the Cmake configuration.
 Open the .sln in VisualStudio and build ALL_BUILD for Release or Debug matching the CMAKE_BUILD_TYPE.
     On Windows the result of the build are three *.lib files


## Applications

A suite of command line and visualization applications is available in the [original High Dimensional Inspector](https://github.com/biovault/High-Dimensional-Inspector) repository.

### CI/CD process

HDILib has all external dependencies as submodules. Conan is only used in the CI/CD process for upload to the artifactory. The conanfile uses the cmake tool as builder.

 - In a python (3.6 or 3.7) environment: 
``` 
pip install conan
```
 - Add the biovault conan remote (for prebuilt packages/upload):
```
conan remote add conan-biovault https://lkeb-artifactory.lumc.nl/artifactory/api/conan/conan-local
```
 - Make a build directory below the HDILib project root. 
    For example: *./_build_release* or *./_build_debug*
    (<u>when using conan the source directories are shared but 
    separate build directories should be used for release and debug.</u>)
 - In the python environment (with conan and cmake accessible) 
 cd to the build directory and issue the following (for VisualStudio 2017):

#### CI/CD note on https
Openssl in the python libraries does not have a recent list of CA-authorities, 
that includes the authority for lkeb-artifactory GEANT issued certificate.
Therefore it is essential to append the lkb-artifactory cert.pem to the
cert.pem file in the conan home directory for a successful https connection. 
See the CI scripts for details.


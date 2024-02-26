# HDILib: High Dimensional Inspector Library ![master ci status](https://github.com/biovault/HDILib/actions/workflows/build.yml/badge.svg)
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
With the latest git versions you shoule use the following command:
```bash
git clone --recurse-submodules https://github.com/biovault/HDILib.git
```

### Requirements

HDILib depends on [flann]((https://github.com/mariusmuja/flann)):

1. Windows build requires: flann >1.8.5
2. Ubuntu/Mac builds require: flann 1.8.4

Flann can be built from the source but we recommend vcpkg to install it, especially on Windows.

### Installing flann 

On Windows with vcpkg
```bash
.\vcpkg install flann:x64-windows-static
```
When configuring cmake make sure to setup vcpkg with CMAKE_TOOLCHAIN_FILE (`PATH_TO/vcpkg/scripts/buildsystems/vcpkg.cmake`) and VCPKG_TARGET_TRIPLET `x64-windows-static`

On Linux with e.g.
```bash
sudo apt-get -y install libflann-dev liblz4-dev
```

On Mac OS with
```
brew install flann lz4
```

### Generate the build files

**Windows**
This will produce a HDILib.sln file for VisualStudio. 
Open the .sln in VisualStudio and build ALL_BUILD for Release or Debug matching the CMAKE_BUILD_TYPE.
```cmd
cmake -S . -B build -G "Visual Studio 16 2019" -A "x64" -DCMAKE_TOOLCHAIN_FILE=.\build\conan_toolchain.cmake -DVCPKG_TARGET_TRIPLET=x64-windows-static -DCMAKE_INSTALL_PREFIX=install
```

**Linux**
This will produce a Makefile, other generators like ninja are also possible. Use the make commands e.g. `make -j 8 && make install` to build and install. 

```bash
cmake  -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install -DENABLE_PID=ON -G "Unix Makefiles"

// build and install the libary, independent of generator
cmake --build build --config Release --target install
```

**Macos**
Tested with Xcode 10.3 & apple-clang 10:
```bash
cmake  -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install
```

## Using the HDILib

The subdirectory test_package builds an exammple that links agains the HDILib binaries. Check the CMakeLists.txt this shows how to consume the HDILib Cmake package.

Find the package
```cmake
find_package(HDILib COMPONENTS hdiutils hdidata hdidimensionalityreduction PATHS ${HDILib_ROOT} CONFIG REQUIRED)
```

Consume the package and dependencies (Windows example)
```cmake
target_link_libraries(example PRIVATE HDI::hdidimensionalityreduction HDI::hdiutils HDI::hdidata ${CMAKE_DL_LIBS})
```

## Applications

A suite of command line and visualization applications is available in the [original High Dimensional Inspector](https://github.com/biovault/High-Dimensional-Inspector) repository.

## CI/CD process

Conan is used in the CI/CD process to retrieve a prebuilt flann from the lkeb-artifactory and to upload the completed HDILib to the artifactory. The conanfile uses the cmake tool as builder.


### CI/CD note on https
OpenSSL in the python libraries does not have a recent list of CA-authorities, that includes the authority for lkeb-artifactory GEANT issued certificate. Therefore it is essential to append the lkeb-artifactory cert.pem to the cert.pem file in the conan home directory for a successful https connection. See the CI scripts for details.

### Notes on Conan

#### CMake compatibility
The conan file uses (starting with conan 1.38) the conan CMakeDeps + CMakeToolchain logic to create a CMake compatible toolchain file. This is a .cmake file
containing all the necessary CMake variables for the build. 

These variables provided by the toolchain file allow the CMake file to locate the required packages that conan has downloaded.

#### Build bundle
The conan build creates three versions of the package, Release, Debug and 

### GitHub Actions status
![master ci status](https://github.com/biovault/HDILib/actions/workflows/build.yml/badge.svg)

Currently the following build matrix is performed:

| OS                   | Architecture | Compiler  |
| -------------------- | ------------ | --------- |
| Windows              | x64          | MSVC 2019 |
| Linux (ubuntu-22.04) | x86_64       | gcc 11    |
| Macos (12)           | x86_64         | clang 13  |

[![DOI](https://zenodo.org/badge/100361974.svg)](https://zenodo.org/badge/latestdoi/100361974)




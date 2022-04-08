###GitHub Actions status
![Branch release/1.2.6](https://github.com/biovault/HDILib/actions/workflows/build.yml/badge.svg?branch=release%2F1.2.6)

Currently the following build matrix is performed

| OS      | Architecture | Compiler  |
| ------- | ------------ | --------- |
| Windows | x64          | MSVC 2017 |
| Windows | x64          | MSVC 2019 |
| Linux   | x86_64       | gcc 8     |
| Linux   | x86_64       | gcc 9     |
| Macos   | x86_64       | clang 10  |
| Macos   | x86_64       | clang 12  |

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

### Requirements

HDILib has all external dependencies as submodules except for flann.

1. Windows build requires: flann 1.8.5
2. Ubuntu/Mac builds require: flann 1.8.4

Flann can be built from the source (https://github.com/mariusmuja/flann) or conan can be used to
installed using conan.

### Installing flann with conan

Assumes a python 3.6,3.7,3.8 or 3.9 environment

1. Install conan

 ```bash
 pip install conan==1.40.4
 ```

2. Add the lkeb artifactory

*Windows*

```cmd
.\tools\add_remote.bat
```

or

*Linux/Macos*
```cmd
./tools/add_remote.sh
```
3. Prepare the build directory

``` bash
mkdir build
cd build
```

4. Generate the build files

* 4a. **Windows**
This will produce a HDILib.sln file for VisualStudio
 Open the .sln in VisualStudio and build ALL_BUILD for Release or Debug matching the CMAKE_BUILD_TYPE.
     On Windows the result of the build are three *.lib files

Run cmake for VS 2017: 
```cmd
cmake .. -G "Visual Studio 15 2017 Win64" -DCMAKE_BUILD_TYPE=Release -DINSTALL_PREBUILT_DEPENDENCIES=ON -DCMAKE_INSTALL_PREFIX=install
```

or for newer VS 2019 (architecture is specified separately)

```cmd
 cmake .. -G "Visual Studio 16 2019" -A "x64" -DCMAKE_BUILD_TYPE=Release -DINSTALL_PREBUILT_DEPENDENCIES=ON -DCMAKE_INSTALL_PREFIX=install
```

In VisualStudio run *Build Solution* for Debug and Release this will produce an *install* directory under the *build* directory containing 
the final libraries and a CMake install file

It is also possible to install cmake itself with pip

```
pip install cmake
```

* 4b. *Linux*
This will produce a Makefile. Use the make command e.g. *make -j 8* to build
gcc8 or gcc9 should be used for prebuilt flann

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DINSTALL_PREBUILT_DEPENDENCIES=ON -DCMAKE_INSTALL_PREFIX=install
```

* 4c. *Macos*
Use Xcode 10.3 apple-clang 10 is supported
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DINSTALL_PREBUILT_DEPENDENCIES=ON -DCMAKE_INSTALL_PREFIX=install
```

## Using the HDILib

The subdirectory test_package builds an exammple that links agains the HDILib binaries. Check the CMakeLists.txt this shows how to consume the HDILib Cmake package.

Find the package
```cmake
find_package(HDILib COMPONENTS hdiutils hdidata hdidimensionalityreduction PATHS ${HDILib_ROOT} CONFIG REQUIRED)
```

Consume the package and dependencies (Windows example)
```cmake
target_link_libraries(example PRIVATE HDI::hdidimensionalityreduction HDI::hdiutils HDI::hdidata flann lz4::lz4 OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})
```


## Applications

A suite of command line and visualization applications is available in the [original High Dimensional Inspector](https://github.com/biovault/High-Dimensional-Inspector) repository.

### CI/CD process

Conan is used in the CI/CD process to retrieve a prebuilt flann from the lkeb-artifactory and to upload the completed HDILib to the artifactory. The conanfile uses the cmake tool as builder.


#### CI/CD note on https
OpenSSL in the python libraries does not have a recent list of CA-authorities, that includes the authority for lkeb-artifactory GEANT issued certificate. Therefore it is essential to append the lkeb-artifactory cert.pem to the cert.pem file in the conan home directory for a successful https connection. See the CI scripts for details.

#### Notes on Conan

##### CMake compatibility
The conan file uses (starting with conan 1.38) the conan CMakeDeps + CMakeToolchain logic to create a CMake compatible toolchain file. This is a .cmake file
containing all the necessary CMake variables for the build. 

These variables provided by the toolchain file allow the CMake file to locate the required packages that conan has downloaded.

##### Build bundle
The conan build creates three versions of the package, Release, Debug and 




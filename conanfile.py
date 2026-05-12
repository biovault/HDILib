# -*- coding: utf-8 -*-

from conans import ConanFile, tools
from conan.tools.cmake import CMakeDeps, CMake, CMakeToolchain
from conans.tools import os_info, SystemPackageTool
import os
import sys
from packaging import version
from pathlib import Path
import subprocess
import json

required_conan_version = "~=1.66.0"


class HDILibConan(ConanFile):
    name = "HDILib"
    description = (
        "HDILib is a library for the scalable analysis of large and high-dimensional"
        " data. "
    )
    topics = ("embedding", "analysis", "n-dimensional", "tSNE")
    url = "https://github.com/biovault/HDILib"
    author = "B. van Lew <b.van_lew@lumc.nl>"  # conanfile author
    license = (  # License for packaged library; please use SPDX Identifiers https://spdx.org/licenses/
        "MIT"
    )
    license = (  # License for packaged library; please use SPDX Identifiers https://spdx.org/licenses/
        "MIT"
    )
    default_user = "lkeb"
    default_channel = "stable"

    # Options may need to change depending on the packaged library
    settings = "os", "compiler", "arch", "build_type"
    # Note : This should only be built with: shared=False, fPIC=True
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    # scm = {
    #    "type": "git",dir
    #    "url": "https://github.com/biovault/HDILib.git",
    #    "submodule": "recursive"
    # }
    # , "conanbuildinfo.txt", "conanbuildinfo_debug.cmake", "conanbuildinfo_release.cmake", "conanbuildinfo_multi.cmake"
    exports = (
        "hdi*",
        "cmake*",
        "CMakeLists.txt",
        "LICENSE",
        "vcpkg.json",
        "vcpkg-overlays*",
        "examples*",
    )

    def _get_python_cmake(self):
        if None is not os.environ.get("APPVEYOR", None):
            pypath = Path(sys.executable)
            cmakePath = Path(pypath.parents[0], "Scripts/cmake.exe")
            return cmakePath
        return "cmake"

    def requirements(self): 
        pass # handled via vcpkg

    def system_requirements(self):
        if os_info.is_macos:
            installer = SystemPackageTool()
            installer.install("libomp")

    def configure(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def _vcpkg_triplet(self):
        # Derive the triplet from conan settings
        arch = "x64" if self.settings.arch == "x86_64" else "x86"
        if self.settings.os == "Windows":
            linkage = (
                "static" if self.options.get_safe("shared") == False else "dynamic"
            )
            return (
                f"{arch}-windows-static-md"
                if linkage == "static"
                else f"{arch}-windows"
            )
        return f"{arch}-linux"  # T.B.D. macos

    def _get_vcpkg_root(self):
        vcpkg_root = os.getenv("VCPKG_DIR", None)  # VCPKG_INSTALLATION_ROOT is the default vcpkg on github ci runners
        print(f"Conan: vcpkg_root is {vcpkg_root}")
        if vcpkg_root is None:
            raise RuntimeError(
                "Expected a preinstalled vcpkg and the environment variable"
                " VCPKG_DIR to be available"
            )
        return vcpkg_root

    def _get_vcpkg_toolchain(self):

        vcpkg_tc_path = Path(
            self._get_vcpkg_root(), "scripts", "buildsystems", "vcpkg.cmake"
        )
        print(f"Conan: vcpkg_tc_path is {vcpkg_tc_path}")
        if not vcpkg_tc_path.exists():
            raise RuntimeError(
                f"Expected vcpkg toolchain not found at {vcpkg_tc_path.absolute()}"
            )
        return vcpkg_tc_path

    def _inject_vcpkg_in_cmake_presets(self):
        # The VCPKG toolchain becomes the primary toolchain,
        # this gives automatic "vcpkg install"without an explicit call.
        # The conan toolchain is "CHAINLOADED" by vcpkg preserving conan functionality
        #
        # In conan v1 the CMake class reads these settings from the presets file
        # and used them to create the cmake command line.
        #
        # T.B.D. check conan v2 mechanism
        conan_toolchain = Path(self.generators_folder, "conan_toolchain.cmake")
        conan_presets = Path(self.generators_folder, "CMakePresets.json")
        with open(conan_presets) as f:
            conan_preset_data = json.load(f)
        for preset in conan_preset_data.get("configurePresets", []):
            preset["cacheVariables"]["VCPKG_CHAINLOAD_TOOLCHAIN_FILE"] = str(
                conan_toolchain.absolute()
            )
            preset["toolchainFile"] = str(self._get_vcpkg_toolchain().absolute())
        print(f"Modifying the presets file: {conan_presets}")
        with open(conan_presets, "w") as f:
            json.dump(conan_preset_data, f, indent=2)

    def generate(self):
        print("Conan: generate CMake toolchain")
        generator = None
        if self.settings.os == "Macos":
            generator = "Xcode"

        if self.settings.os == "Linux":
            generator = "Ninja Multi-Config"

        # A toolchain file can be used to modify CMake variables
        tc = CMakeToolchain(self, generator=generator)
        if self.settings.os == "Windows":
            tc.variables["CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS"] = True

        if self.build_folder is not None:
            tc.variables["CMAKE_INSTALL_PREFIX"] = str(
                Path(self.build_folder, "install").as_posix()
            )
        else:
            tc.variables["CMAKE_INSTALL_PREFIX"] = "${CMAKE_BINARY_DIR}"
        
        tc.variables["CMAKE_VERBOSE_MAKEFILE"] = "ON"

        if os_info.is_macos:
            proc = subprocess.run(
                "brew --prefix libomp", shell=True, capture_output=True
            )
            tc.variables["OpenMP_ROOT"] = f"{proc.stdout.decode('UTF-8').strip()}"

        tc.cache_variables["HDILib_BUILD_EXAMPLE"] = True

        tc.generate()
        self._inject_vcpkg_in_cmake_presets()

    def _configure_cmake(self):
        cmake = CMake(self)
        cmake.configure()
        return cmake

    def build(self):
        print(
            f"Build folder {self.build_folder} \n Package folder"
            f" {self.package_folder}\n Source folder {self.source_folder}"
        )
        install_dir = Path(self.build_folder).joinpath("install")
        install_dir.mkdir(exist_ok=True)

        print(f"Conan: Build Debug")
        cmake_debug = self._configure_cmake()
        cmake_debug.build(build_type="Debug")
        cmake_debug.install(build_type="Debug")

        print(f"Conan: Build Release")
        cmake_release = self._configure_cmake()
        cmake_release.build(build_type="Release")
        cmake_release.install(build_type="Release")

        #print(f"Conan: Build RelWithDebInfo")
        #cmake_release = self._configure_cmake()
        #cmake_release.build(build_type="RelWithDebInfo")
        #cmake_release.install(build_type="RelWithDebInfo")

    def package_id(self):
        # The package contains both Debug and Release build types
        del self.info.settings.build_type
        if self.settings.compiler == "Visual Studio":
            del self.info.settings.compiler.runtime

    def package_info(self):
        print("In packageinfo")
        self.cpp_info.libs = tools.collect_libs(self)
        self.cpp_info.set_property("skip_deps_file", True)
        self.cpp_info.set_property("cmake_config_file", True)

        # Also package the dependencies from vcpkg
        self.cpp_info.components["kompute"].libs = ["kompute"]
        self.cpp_info.components["kompute"].includedirs = ["include"]
        self.cpp_info.components["kompute"].libdirs = ["lib"]

        self.cpp_info.components["fmt"].libs = ["fmt"]
        self.cpp_info.components["fmt"].includedirs = ["include"]
        self.cpp_info.components["fmt"].libdirs = ["lib"]

        # If your main lib depends on these components
        # self.cpp_info.components["hdilib"].libs = ["HDILib"]
        # self.cpp_info.components["hdilib"].requires = ["kompute", "fmt"]

    def package(self):
        # Additional vcpkg build package components are copied into
        # install in the main CMakeLists.txt
        install_dir = Path(self.build_folder).joinpath("install")
        self.copy(pattern="*", src=str(install_dir))
        # Add the debug support files to the package
        # (*.pdb) if building the Visual Studio version
        if self.settings.compiler == "Visual Studio":
            self.copy("*.pdb", dst="lib/Debug", keep_path=False)
            self.copy("*.pdb", dst="lib/RelWithDebInfo", keep_path=False)

# -*- coding: utf-8 -*-

from conans import ConanFile, tools
from conan.tools.cmake import CMakeDeps, CMake, CMakeToolchain
import os
import sys
from packaging import version
from pathlib import Path

required_conan_version = ">=1.38.0"


class HDILibConan(ConanFile):
    name = "HDILib"
    version = "1.2.5"
    description = "HDILib is a library for the scalable analysis of large and high-dimensional data. "
    topics = ("embedding", "analysis", "n-dimensional", "tSNE")
    url = "https://github.com/biovault/HDILib"
    author = "B. van Lew <b.van_lew@lumc.nl>"  # conanfile author
    license = "MIT"  # License for packaged library; please use SPDX Identifiers https://spdx.org/licenses/
    default_user = "lkeb"
    default_channel = "stable"

    generators = "CMakeDeps"

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
    exports = "hdi*", "external*", "cmake*", "CMakeLists.txt", "LICENSE"

    def _get_python_cmake(self):
        if None is not os.environ.get("APPVEYOR", None):
            pypath = Path(sys.executable)
            cmakePath = Path(pypath.parents[0], "Scripts/cmake.exe")
            return cmakePath
        return "cmake"

    def system_requirements(self):
        if tools.os_info.is_macos:
            target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "10.13")
            if version.parse(target) > version.parse("10.12"):
                installer = tools.SystemPackageTool()
                installer.install("libomp")

    # Flann builds are bit complex and certain versions fail with
    # certain platform, and compiler combinations. Hence use
    # either self built 1.8.5 for Windows or system supplied
    # 1.8.4 on Linux and Macos
    def requirements(self):
        if self.settings.build_type == "None":
            print("Skip root package requirements for build_type NONE")
            return
        if self.settings.os == "Windows":
            self.requires("flann/1.8.5@lkeb/testing")
        else:
            # Macos and linux use 1.8.4
            self.requires("flann/1.8.4@lkeb/testing")
        # self.requires.add("lz4/1.9.2")

    def configure(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def fix_config_packages(self):
        """Iterate the dependencies and add the package root where
        it is marked as a "cmake_config_file" and when "skip_deps_file" is
        enabled. Permits using a package local cmake config-file.
        (the flann package is build with skip-deps)
        """
        package_names = {r.ref.name for r in self.dependencies.host.values()}
        for package_name in package_names:
            cpp_info = None
            package_props = self.dependencies[f"{package_name}"]
            # add fallback to old cpp_info name
            if hasattr(package_props, "cpp_info"):
                cpp_info = package_props.cpp_info
            else:
                cpp_info = package_props.new_cpp_info
            if cpp_info.get_property("skip_deps_file", CMakeDeps):
                print(f"No deps generated for {package_name} - skip_deps_file found")
            if cpp_info.get_property(
                "skip_deps_file", CMakeDeps
            ) and cpp_info.get_property("cmake_config_file", CMakeDeps):
                package_root = Path(package_props.package_folder)
                with open("conan_toolchain.cmake", "a") as toolchain:
                    toolchain.write(
                        fr"""
set(CMAKE_MODULE_PATH "{package_root.as_posix()}" ${{CMAKE_MODULE_PATH}})
set(CMAKE_PREFIX_PATH "{package_root.as_posix()}" ${{CMAKE_PREFIX_PATH}})
                    """
                    )
                with open("installed_packages.cmake", "a") as installed_packages:
                    installed_packages.write(
                        fr"""
set(CMAKE_MODULE_PATH "{package_root.as_posix()}" ${{CMAKE_MODULE_PATH}})
set(CMAKE_PREFIX_PATH "{package_root.as_posix()}" ${{CMAKE_PREFIX_PATH}})
                    """
                    )

    def generate(self):
        print("In generate")
        generator = None
        if self.settings.os == "Macos":
            generator = "Xcode"

        if self.settings.os == "Linux":
            generator = "Ninja Multi-Config"

        tc = CMakeToolchain(self, generator=generator)
        if self.settings.os == "Windows":
            tc.variables["CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS"] = True
        if self.settings.os == "Linux" or self.settings.os == "Macos":
            tc.variables["CMAKE_CXX_STANDARD"] = 14
            tc.variables["CMAKE_CXX_STANDARD_REQUIRED"] = "ON"
        tc.variables["HDI_EXTERNAL_FLANN_INCLUDE_DIR"] = "${CONAN_INCLUDE_DIRS_FLANN}"
        tc.variables["HDILib_VERSION"] = self.version
        if self.build_folder is not None:
            tc.variables["CMAKE_INSTALL_PREFIX"] = str(
                Path(self.build_folder, "install").as_posix()
            )
        else:
            tc.variables["CMAKE_INSTALL_PREFIX"] = "${CMAKE_BINARY_DIR}"
        tc.variables["ENABLE_CODE_ANALYSIS"] = "ON"
        tc.variables["CMAKE_VERBOSE_MAKEFILE"] = "ON"
        if os.getenv("Analysis", None) is None:
            tc.variables["ENABLE_CODE_ANALYSIS"] = "OFF"
        print("Call toolchain generate")
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()
        self.fix_config_packages()

    def _configure_cmake(self):
        cmake = CMake(self)
        print(f"Set version to {self.version}")
        cmake.configure()
        return cmake

    def build(self):
        print(
            f"Build folder {self.build_folder} \n Package folder {self.package_folder}\n Source folder {self.source_folder}"
        )
        install_dir = Path(self.build_folder).joinpath("install")
        install_dir.mkdir(exist_ok=True)

        cmake_debug = self._configure_cmake()
        cmake_debug.install(build_type="Debug")

        if os.getenv("Analysis", None) is None:
            # Disable code analysis in Release mode
            cmake_release = self._configure_cmake()
            cmake_release.install(build_type="Release")

    def package_id(self):
        # The package contains both Debug and Release build types
        del self.info.settings.build_type
        if self.settings.compiler == "Visual Studio":
            del self.info.settings.compiler.runtime

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
        self.cpp_info.set_property("skip_deps_file", True)
        self.cpp_info.set_property("cmake_config_file", True)

    def package(self):
        install_dir = Path(self.build_folder).joinpath("install")
        self.copy(pattern="*", src=str(install_dir))
        # Add the debug support files to the package
        # (*.pdb) if building the Visual Studio version
        if self.settings.compiler == "Visual Studio":
            self.copy("*.pdb", dst="lib/Debug", keep_path=False)

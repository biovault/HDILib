import os
import platform
from pathlib import Path
from conans import ConanFile, tools
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps
from conans.tools import os_info
import shutil
import subprocess
import json

required_conan_version = "~=1.66.0"

# This is a "hello world" type test that checks that the conan package can be consumed
# i.e. that that cmake support works, consumption of HDILib headers (compiler) and lib (linker) works
# and nothing else. It is is not a full unit or regression test for HDILib.


class HDILibTestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    # cmake_paths locates the HDILib built with build_type None
    
    def _get_vcpkg_root(self): 
        vcpkg_root = os.getenv("VCPKG_INSTALLATION_ROOT", None)
        if vcpkg_root is None:
            raise RuntimeError("Expected a preinstalled vcpkg and the environment variable VCPKG_INSTALLATION_ROOT to be available")
        return vcpkg_root
    
    def _get_vcpkg_toolchain(self):
        vcpkg_tc_path = Path(self._get_vcpkg_root(), "scripts", "buildsystems", "vcpkg.cmake")
        if not vcpkg_tc_path.exists():
            raise RuntimeError(f"Expected vcpkg toolchain not found at {vcpkg_tc_path.absolute()}")
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
            preset["cacheVariables"]["VCPKG_CHAINLOAD_TOOLCHAIN_FILE"] = str(conan_toolchain.absolute())
            preset["toolchainFile"] = str(self._get_vcpkg_toolchain().absolute())
        print(f"Modifying the presets file: {conan_presets}")
        with open(conan_presets, "w") as f:
          json.dump(conan_preset_data, f, indent=2)

    def generate(self):

        # deps = CMakeDeps(self)
        # deps.generate()

        if os.getenv("Analysis", None) is not None:
            return
        tc = CMakeToolchain(self)
        tc.variables["HDILib_ROOT"] = Path(
            self.deps_cpp_info["HDILib"].rootpath
        ).as_posix()
        # These vulkan related dependencies are bundled with HDILib
        tc.variables["kompute_ROOT"] = Path(
            self.deps_cpp_info["HDILib"].rootpath
        ).as_posix()
        tc.variables["fmt_ROOT"] = Path(
            self.deps_cpp_info["HDILib"].rootpath
        ).as_posix()
        tc.variables["glfw3_ROOT"] = Path(
            self.deps_cpp_info["HDILib"].rootpath
        ).as_posix()
        # Use the cmake export in the flann package
        tc.variables["flann_ROOT"] = Path(
            self.deps_cpp_info["flann"].rootpath, "lib", "cmake"
        ).as_posix()
        # Use the cmake export in the lz4 package
        tc.variables["lz4_ROOT"] = Path(
            self.deps_cpp_info["lz4"].rootpath, "lib", "cmake"
        ).as_posix()


        if os_info.is_macos:
            proc = subprocess.run(
                "brew --prefix libomp", shell=True, capture_output=True
            )
            omp_prefix_path = f"{proc.stdout.decode('UTF-8').strip()}"
            tc.variables["OpenMP_ROOT"] = omp_prefix_path

        tc.generate()

        self._inject_vcpkg_in_cmake_presets()

    def requirements(self):
        if os.getenv("Analysis", None) is not None:
            return
        print("In requirements")
        if self.settings.build_type == "None":
            print("Skip test_package requirements for build_type NONE")
            return
        else:
            self.requires("flann/1.9.2@lkeb/%s" % self.channel)

    def system_requirements(self):
        if os.getenv("Analysis", None) is not None:
            return
        if tools.os_info.is_linux:
            installer = tools.SystemPackageTool()
            # installer.install("libomp5")
            # installer.install("libomp-dev")

    def build(self):
        if os.getenv("Analysis", None) is not None:
            return
        
        cmake = CMake(self)
        cmake.configure()
        cmake.build()


    def test(self):
        if os.getenv("Analysis", None) is not None:
            return
        if not tools.cross_building(self.settings):
            os.chdir("bin")
            if platform.system() == "Windows":
                shutil.copy(
                    Path(
                        self.deps_cpp_info["lz4"].rootpath, "bin", "Release", "lz4.dll"
                    ),
                    Path("./", str(self.build_folder), "bin"),
                )
                examplePath = Path("./", str(self.build_folder), "bin", "example.exe")
                self.run(f"{str(examplePath)}")
            elif platform.system() == "Darwin":
                shutil.copy(
                    Path(
                        self.deps_cpp_info["flann"].rootpath,
                        "lib",
                        "Release",
                        "libflann_cpp.1.9.dylib",
                    ),
                    Path("./"),
                )
            else:
                self.run(".%sexample" % os.sep)

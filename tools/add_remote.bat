for /f "delims=" %%i in ('conan config home') do set FOUND_CONAN_HOME="%%i"
type %~dp0\cert.pem >> %FOUND_CONAN_HOME%\cacert.pem
conan remote add conan-biovault https://lkeb-artifactory.lumc.nl/artifactory/api/conan/conan-local
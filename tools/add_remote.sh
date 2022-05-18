FOUND_CONAN_HOME=`conan config home`
SCRIPT_PATH=`realpath $0`
BASE_PATH="${SCRIPT_PATH%/*}"
cat "$BASE_PATH"/../cert.pem >> "$FOUND_CONAN_HOME"/cacert.pem
conan remote add conan-biovault https://lkeb-artifactory.lumc.nl/artifactory/api/conan/conan-local
#!/usr/bin/env bash

set -ex

if [[ "$(uname -s)" == 'Darwin' ]]; then
    brew outdated python || brew upgrade python
    brew outdated pyenv || brew upgrade pyenv
    brew install pyenv-virtualenv
    brew install cmake || brew upgrade cmake || true

    if which pyenv > /dev/null; then
        eval "$(pyenv init -)"
    fi

    pyenv install 3.7.1
    pyenv virtualenv 3.7.1 conan
    pyenv rehash
    pyenv activate conan
fi

pip install conan --upgrade
pip install conan_package_tools bincrafters_package_tools packaging

conan user
echo Extend conan cacert.pem
conanhome=`conan config home`
cat cert.pem >> $conanhome/cacert.pem
conan remote remove conan-center
conan remote add center-artifactory https://conan.bintray.com

if [[ "$(uname -s)" == 'Darwin' ]]; then
    conan install lz4/1.9.2@ -r center-artifactory -s compiler=apple-clang -s arch=x86_64 -s build_type=Release -s compiler.version=10.0 -s os=Macos -o fPIC=True -o shared=False
fi
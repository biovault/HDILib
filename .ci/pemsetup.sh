#!/usr/bin/env bash

echo Extend conan cacert.pem
conanhome=`conan config home`
cat cert.pem >> $conanhome/cacert.pem
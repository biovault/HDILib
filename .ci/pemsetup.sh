#!/usr/bin/env bash

echo Extend conan cacert.pem
if [ $# -eq 0 ] ; then
    conanhome=`conan config home`
else
    conanhome=$1
fi
echo Conan home: $conanhome/cacert.pem
echo Lines in cacert.pem before update: $(wc -l $conanhome/cacert.pem)
cat cert.pem >> $conanhome/cacert.pem
echo Lines in cacert.pem after update : $(wc -l $conanhome/cacert.pem)
#!/bin/bash

make cleanreally

if [ $# -gt 0 ]; then
	VERSION=$1
else
	VERSION=g++
fi

distcc-pump make -j38 CXX="distcc $VERSION"

if [ $? -gt 0 ]; then
	distcc-pump make -j38 CXX="distcc $VERSION"
fi

exit

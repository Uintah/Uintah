#!/bin/bash

make cleanreally

if [ $# -gt 0 ]; then
	VERSION=$1
else
	VERSION=g++
fi

pump make -j38 CXX="distcc $VERSION"

if [ $? -gt 0 ]; then
	pump make -j38 CXX="distcc $VERSION"
fi

exit

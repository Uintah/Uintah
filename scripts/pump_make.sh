#!/bin/bash

make cleanreally

pump make -j38 CXX="distcc g++"

exit

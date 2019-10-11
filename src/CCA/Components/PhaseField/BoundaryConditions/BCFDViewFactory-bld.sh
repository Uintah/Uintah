#!/bin/bash
#
#  The MIT License
#
#  Copyright (c) 1997-2018 The University of Utah
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#  IN THE SOFTWARE.
#

VARs=(CC NC)
STNs=(P5 P7)
DIMs=(2 3)
PPPs=(
  "PureMetalProblem;HeatProblem"
  "PureMetalProblem;HeatProblem"
)
NFFs=(
  "4;1"
  "4;1"
)
DIRs=(x y z)
SIGNs=(minus plus)
BCs=(Dirichlet Neumann)

SS=""
VV=""
PP=""

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -v|--var)
    VARs=("$2")
    VV="$2"
    shift # past argument
    shift # past value
    ;;
    -s|--stn)
    SS="$2"
    shift # past argument
    shift # past value
    ;;
    -p|--pb)
    PP="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

if [ -n "${SS}" ]; then
  FOUND=0
  for ((s=0; s<${#STNs[@]}; s++)); do
    if [ "${SS}" == "${STNs[s]}" ]; then
      STNs=("${STNs[s]}");
      DIMs=("${DIMs[s]}");
      PPPs=("${PPPs[s]}");
      NFFs=("${NFFs[s]}");
      FOUND=1
    fi
  done
  if [[ $FOUND -eq 0 ]]; then
    >&2 echo "cannot find the stencil"
    exit
  fi
fi

if [ -n "${PP}" ]; then
  for ((pp=0; pp<${#PPPs[@]}; pp++)); do
    FOUND=0
    IFS=';' read -r -a PPs <<< "${PPPs[pp]}"
    IFS=';' read -r -a NFs <<< "${NFFs[pp]}"
    for ((p=0; p<${#PPs[@]}; p++)); do
      if [ "${PP}" == "${PPs[p]}" ]; then
        PPPs[pp]="${PPs[p]}";
        NFFs[pp]="${NFs[p]}";
        FOUND=1
      fi
    done
    if [[ $FOUND -eq 0 ]]; then
      PPPs[pp]="";
      NFFs[pp]="";
    fi
  done
fi

PBS=()
for ((pp=0; pp<${#PPPs[@]}; pp++)); do
  IFS=';' read -r -a PPs <<< "${PPPs[pp]}"
  for ((p=0; p<${#PPs[@]}; p++)); do
    FOUND=0
    for ((q=0; q<${#PBS[@]}; q++)); do
      if [ "${PPs[p]}" == "${PBS[q]}" ]; then
        FOUND=1
      fi
    done
    if [[ $FOUND -eq 0 ]]; then
      PBS+=("${PPs[p]}")
    fi
  done
done

SRC=${0%-bld.sh}${PP}${VV}${SS}-bld.cc

echo "generating $SRC"

echo '/*' > $SRC
echo ' * The MIT License' >> $SRC
echo ' *' >> $SRC
echo ' * Copyright (c) 1997-2018 The University of Utah' >> $SRC
echo ' *' >> $SRC
echo ' * Permission is hereby granted, free of charge, to any person obtaining a copy' >> $SRC
echo ' * of this software and associated documentation files (the "Software"), to' >> $SRC
echo ' * deal in the Software without restriction, including without limitation the' >> $SRC
echo ' * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or' >> $SRC
echo ' * sell copies of the Software, and to permit persons to whom the Software is' >> $SRC
echo ' * furnished to do so, subject to the following conditions:' >> $SRC
echo ' *' >> $SRC
echo ' * The above copyright notice and this permission notice shall be included in' >> $SRC
echo ' * all copies or substantial portions of the Software.' >> $SRC
echo ' *' >> $SRC
echo ' * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR' >> $SRC
echo ' * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,' >> $SRC
echo ' * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE' >> $SRC
echo ' * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER' >> $SRC
echo ' * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING' >> $SRC
echo ' * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS' >> $SRC
echo ' * IN THE SOFTWARE.' >> $SRC
echo ' */' >> $SRC
echo '' >> $SRC
for PB in ${PBS[@]}; do
  echo '#include <CCA/Components/PhaseField/DataTypes/'$PB'.h>' >> $SRC
done
echo '#include <CCA/Components/PhaseField/BoundaryConditions/BCFDView.h>' >> $SRC
echo '#include <CCA/Components/PhaseField/BoundaryConditions/BCFDViewFactory.h>' >> $SRC
echo '' >> $SRC
echo 'namespace Uintah {' >> $SRC
echo 'namespace PhaseField {' >> $SRC
echo '' >> $SRC

Fs=(
  "ScalarField<const double>"
  "VectorField<const double, 1>"
  "VectorField<const double, 3>"
)
STNs=(
  "P3 P5 P7"
  "P5"
  "P7"
)
VARs=(CC NC)

for ((f=0; f<${#Fs[@]}; f++)); do
  F="${Fs[f]}";
  for STN in ${STNs[f]}; do
    echo "template<> BCFactoryFDView < $F, $STN >::FactoryMap BCFactoryFDView < $F, $STN >::RegisteredNames = {};" >> $SRC
  done
  echo "" >> $SRC
done

echo '} // namespace Uintah' >> $SRC
echo '} // namespace PhaseField' >> $SRC

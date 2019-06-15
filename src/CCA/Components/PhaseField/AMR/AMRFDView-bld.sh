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
  "PureMetalProblem"
  "PureMetalProblem"
)
NFFs=(
  "4"
  "4"
)
DIRs=(x y z)
SIGNs=(minus plus)
BCs=(Dirichlet Neumann)
C2Fs=(FC0 FC1 FCSimple FCLinear FCBilinear)

SS=""
VV=""
CC=""
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
    -c|--c2f)
    C2Fs=("$2")
    CC="$2"
    shift # past argument
    shift # past value
    ;;
    -B|--no-bc)
    BCs=()
    shift # past argument
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

SRC=${0%-bld.sh}${PP}${VV}${SS}${CC}-bld.cc

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
echo '' >> $SRC
echo 'namespace Uintah {' >> $SRC
echo 'namespace PhaseField {' >> $SRC
echo '' >> $SRC

for VAR in ${VARs[@]}; do
  for ((s=0; s<${#STNs[@]}; s++)); do
    STN="${STNs[s]}"
    DIM="${DIMs[s]}"
    IFS=';' read -r -a PPs <<< "${PPPs[s]}"
    IFS=';' read -r -a NFs <<< "${NFFs[s]}"
    for ((n=0; n<${#PPs[@]}; n++)); do
      PP="${PPs[n]}"
      NF="${NFs[n]}"
      PB="$PP<$VAR, $STN>"
      for ((f=0; f<$NF; f++)); do
        I=$f

        for ((d0=0; d0<$DIM; d0++)); do
          DIR0="${DIRs[d0]}"
          for SIGN0 in "${SIGNs[@]}"; do
            F0=$DIR0$SIGN0
            for C2F in "${C2Fs[@]}"; do
              P0="Patch::$F0 | BC::FineCoarseInterface | FC::$C2F"
              echo "template<> const std::string BCFDView < $PB, $I, $P0 >::Name = \"$PP|$I|$VAR|$F0|$C2F|\";" >> $SRC

              for ((d1=d0+1; d1<$DIM; d1++)); do
                DIR1="${DIRs[d1]}"
                for SIGN1 in "${SIGNs[@]}"; do
                  F1=$DIR1$SIGN1
                  P1="Patch::$F1 | BC::FineCoarseInterface | FC::$C2F"
                  echo "template<> const std::string BCFDView < $PB, $I, $P0, $P1 >::Name = \"$PP|$I|$VAR|$F0|$C2F|$F1|$C2F|\";" >> $SRC

                  for ((d2=d1+1; d2<$DIM; d2++)); do
                    DIR2="${DIRs[d2]}"
                    for SIGN2 in "${SIGNs[@]}"; do
                      F2=$DIR2$SIGN2
                      P2="Patch::$F2 | BC::FineCoarseInterface | FC::$C2F"
                      echo "template<> const std::string BCFDView < $PB, $I, $P0, $P1, $P2 >::Name = \"$PP|$I|$VAR|$F0|$C2F|$F1|$C2F|$F2|$C2F|\";" >> $SRC

                      for BC2 in "${BCs[@]}"; do
                        P2="Patch::$F2 | BC::$BC2"
                        echo "template<> const std::string BCFDView < $PB, $I, $P0, $P1, $P2 >::Name = \"$PP|$I|$VAR|$F0|$C2F|$F1|$C2F|$F2|$BC2|\";" >> $SRC

                      done
                    done
                  done

                  for BC1 in "${BCs[@]}"; do
                    P1="Patch::$F1 | BC::$BC1"
                    echo "template<> const std::string BCFDView < $PB, $I, $P0, $P1 >::Name = \"$PP|$I|$VAR|$F0|$C2F|$F1|$BC1|\";" >> $SRC

                    for ((d2=d1+1; d2<$DIM; d2++)); do
                      DIR2="${DIRs[d2]}"
                      for SIGN2 in "${SIGNs[@]}"; do
                        F2=$DIR2$SIGN2
                        P2="Patch::$F2 | BC::FineCoarseInterface | FC::$C2F"
                        echo "template<> const std::string BCFDView < $PB, $I, $P0, $P1, $P2 >::Name = \"$PP|$I|$VAR|$F0|$C2F|$F1|$BC1|$F2|$C2F|\";" >> $SRC

                        for BC2 in "${BCs[@]}"; do
                          P2="Patch::$F2 | BC::$BC2"
                          echo "template<> const std::string BCFDView < $PB, $I, $P0, $P1, $P2 >::Name = \"$PP|$I|$VAR|$F0|$C2F|$F1|$BC1|$F2|$BC2|\";" >> $SRC

                        done
                      done
                    done

                  done
                done
              done

            done
            for BC0 in "${BCs[@]}"; do
              P0="Patch::$F0 | BC::$BC0"

              for ((d1=d0+1; d1<$DIM; d1++)); do
                DIR1="${DIRs[d1]}"
                for SIGN1 in "${SIGNs[@]}"; do
                  F1=$DIR1$SIGN1
                  for C2F in "${C2Fs[@]}"; do
                    P1="Patch::$F1 | BC::FineCoarseInterface | FC::$C2F"
                    echo "template<> const std::string BCFDView < $PB, $I, $P0, $P1 >::Name = \"$PP|$I|$VAR|$F0|$BC0|$F1|$C2F|\";" >> $SRC

                    for ((d2=d1+1; d2<$DIM; d2++)); do
                      DIR2="${DIRs[d2]}"
                      for SIGN2 in "${SIGNs[@]}"; do
                        F2=$DIR2$SIGN2
                        P2="Patch::$F2 | BC::FineCoarseInterface | FC::$C2F"
                        echo "template<> const std::string BCFDView < $PB, $I, $P0, $P1, $P2 >::Name = \"$PP|$I|$VAR|$F0|$BC0|$F1|$C2F|$F2|$C2F|\";" >> $SRC

                        for BC2 in "${BCs[@]}"; do
                          P2="Patch::$F2 | BC::$BC2"
                          echo "template<> const std::string BCFDView < $PB, $I, $P0, $P1, $P2 >::Name = \"$PP|$I|$VAR|$F0|$BC0|$F1|$C2F|$F2|$BC2|\";" >> $SRC

                        done
                      done
                    done

                  done
                  for BC1 in "${BCs[@]}"; do
                    P1="Patch::$F1 | BC::$BC1"

                    for ((d2=d1+1; d2<$DIM; d2++)); do
                      DIR2="${DIRs[d2]}"
                      for SIGN2 in "${SIGNs[@]}"; do
                        F2=$DIR2$SIGN2
                        for C2F in "${C2Fs[@]}"; do
                          P2="Patch::$F2 | BC::FineCoarseInterface | FC::$C2F"
                          echo "template<> const std::string BCFDView < $PB, $I, $P0, $P1, $P2 >::Name = \"$PP|$I|$VAR|$F0|$BC0|$F1|$BC1|$F2|$C2F|\";" >> $SRC

                        done
                      done
                    done

                  done
                done
              done

            done
          done
        done

      done
    done
  done
done

echo "" >> $SRC

for VAR in ${VARs[@]}; do
  for ((s=0; s<${#STNs[@]}; s++)); do
    STN="${STNs[s]}"
    DIM="${DIMs[s]}"
    IFS=';' read -r -a PPs <<< "${PPPs[s]}"
    IFS=';' read -r -a NFs <<< "${NFFs[s]}"
    for ((n=0; n<${#PPs[@]}; n++)); do
      PB="${PPs[n]}<$VAR, $STN>"
      NF="${NFs[n]}"
      for ((f=0; f<$NF; f++)); do
        I=$f

        for ((d0=0; d0<$DIM; d0++)); do
          DIR0="${DIRs[d0]}"
          for SIGN0 in "${SIGNs[@]}"; do
            F0=$DIR0$SIGN0
            for C2F in "${C2Fs[@]}"; do
              P0="Patch::$F0 | BC::FineCoarseInterface | FC::$C2F"
              echo "template class BCFDView < $PB, $I, $P0 >;" >> $SRC

              for ((d1=d0+1; d1<$DIM; d1++)); do
                DIR1="${DIRs[d1]}"
                for SIGN1 in "${SIGNs[@]}"; do
                  F1=$DIR1$SIGN1
                  P1="Patch::$F1 | BC::FineCoarseInterface | FC::$C2F"
                  echo "template class BCFDView < $PB, $I, $P0, $P1 >;" >> $SRC

                  for ((d2=d1+1; d2<$DIM; d2++)); do
                    DIR2="${DIRs[d2]}"
                    for SIGN2 in "${SIGNs[@]}"; do
                      F2=$DIR2$SIGN2
                      P2="Patch::$F2 | BC::FineCoarseInterface | FC::$C2F"
                      echo "template class BCFDView < $PB, $I, $P0, $P1, $P2 >;" >> $SRC

                      for BC2 in "${BCs[@]}"; do
                        P2="Patch::$F2 | BC::$BC2"
                        echo "template class BCFDView < $PB, $I, $P0, $P1, $P2 >;" >> $SRC

                      done
                    done
                  done

                  for BC1 in "${BCs[@]}"; do
                    P1="Patch::$F1 | BC::$BC1"
                    echo "template class BCFDView < $PB, $I, $P0, $P1 >;" >> $SRC

                    for ((d2=d1+1; d2<$DIM; d2++)); do
                      DIR2="${DIRs[d2]}"
                      for SIGN2 in "${SIGNs[@]}"; do
                        F2=$DIR2$SIGN2
                        P2="Patch::$F2 | BC::FineCoarseInterface | FC::$C2F"
                        echo "template class BCFDView < $PB, $I, $P0, $P1, $P2 >;" >> $SRC

                        for BC2 in "${BCs[@]}"; do
                          P2="Patch::$F2 | BC::$BC2"
                          echo "template class BCFDView < $PB, $I, $P0, $P1, $P2 >;" >> $SRC

                        done
                      done
                    done

                  done
                done
              done

            done
            for BC0 in "${BCs[@]}"; do
              P0="Patch::$F0 | BC::$BC0"

              for ((d1=d0+1; d1<$DIM; d1++)); do
                DIR1="${DIRs[d1]}"
                for SIGN1 in "${SIGNs[@]}"; do
                  F1=$DIR1$SIGN1
                  for C2F in "${C2Fs[@]}"; do
                    P1="Patch::$F1 | BC::FineCoarseInterface | FC::$C2F"
                    echo "template class BCFDView < $PB, $I, $P0, $P1 >;" >> $SRC

                    for ((d2=d1+1; d2<$DIM; d2++)); do
                      DIR2="${DIRs[d2]}"
                      for SIGN2 in "${SIGNs[@]}"; do
                        F2=$DIR2$SIGN2
                        P2="Patch::$F2 | BC::FineCoarseInterface | FC::$C2F"
                        echo "template class BCFDView < $PB, $I, $P0, $P1, $P2 >;" >> $SRC

                        for BC2 in "${BCs[@]}"; do
                          P2="Patch::$F2 | BC::$BC2"
                          echo "template class BCFDView < $PB, $I, $P0, $P1, $P2 >;" >> $SRC

                        done
                      done
                    done

                  done
                  for BC1 in "${BCs[@]}"; do
                    P1="Patch::$F1 | BC::$BC1"

                    for ((d2=d1+1; d2<$DIM; d2++)); do
                      DIR2="${DIRs[d2]}"
                      for SIGN2 in "${SIGNs[@]}"; do
                        F2=$DIR2$SIGN2
                        for C2F in "${C2Fs[@]}"; do
                          P2="Patch::$F2 | BC::FineCoarseInterface | FC::$C2F"
                          echo "template class BCFDView < $PB, $I, $P0, $P1, $P2 >;" >> $SRC

                        done
                      done
                    done

                  done
                done
              done

            done
          done
        done

      done
    done
  done
done

echo "" >> $SRC

echo '} // namespace Uintah' >> $SRC
echo '} // namespace PhaseField' >> $SRC

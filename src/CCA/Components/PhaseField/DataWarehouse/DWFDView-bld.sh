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

SRC=${0%.sh}.cc
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
echo '#include <CCA/Components/PhaseField/DataWarehouse/DWFDView.h>' >> $SRC
echo '#include <CCA/Components/PhaseField/DataWarehouse/DWFDViewFactory.h>' >> $SRC
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
    echo "template<> DWFactoryFDView < $F, $STN >::FactoryMap DWFactoryFDView < $F, $STN >::RegisteredNames = {};" >> $SRC
  done
  echo "" >> $SRC
done

for ((f=0; f<${#Fs[@]}; f++)); do
  F="${Fs[f]}";
  for STN in ${STNs[f]}; do
    for VAR in ${VARs[@]}; do
      echo "template<> const std::string DWFDView < $F, $STN, $VAR >::Name = \"$VAR|$STN\";" >> $SRC
    done
  done
  echo "" >> $SRC
done

for ((f=0; f<${#Fs[@]}; f++)); do
  F="${Fs[f]}";
  for STN in ${STNs[f]}; do
    for VAR in ${VARs[@]}; do
      echo "template class DWFDView < $F, $STN, $VAR >;" >> $SRC
    done
  done
  echo "" >> $SRC
done

echo '} // namespace Uintah' >> $SRC
echo '} // namespace PhaseField' >> $SRC

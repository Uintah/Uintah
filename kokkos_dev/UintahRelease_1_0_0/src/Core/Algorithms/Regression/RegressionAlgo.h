/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


#ifndef CORE_ALGORITHMS_REGRESSION_REGRESSIONALGO_H
#define CORE_ALGORITHMS_REGRESSION_REGRESSIONALGO_H 1

#include <Core/Algorithms/Util/AlgoLibrary.h>
#include <Core/Algorithms/Regression/share.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class RegressionAlgo : public AlgoLibrary {

  public:
    // Constructor
    // If no error reporting is required, initialize with a zero pointer
    RegressionAlgo(ProgressReporter* pr); 

    // Regression Testing depends on comparing with known outputs
    // The following functions are for comparing
    
    // The functions return an error if the compare function failed or if
    // the fields/matrices/nrrds/strings/bundles are not equal.
    
    bool CompareFields(FieldHandle& field1, FieldHandle& field2);
    bool CompareMatrices(MatrixHandle& matrix1, MatrixHandle& matrix2);
    bool CompareNrrds(NrrdDataHandle& nrrd1, NrrdDataHandle& nrrd2);
    bool CompareStrings(StringHandle& string1, StringHandle& string2);
    bool CompareBundles(BundleHandle& bundle1, BundleHandle& bundle2);

};

} // SCIRunAlgo

#endif

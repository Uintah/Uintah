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


#ifndef CORE_ALGORITHMS_MATH_MATHALGO_H
#define CORE_ALGORITHMS_MATH_MATHALGO_H 1

#include <Core/Algorithms/Util/AlgoLibrary.h>

#include <Core/Bundle/Bundle.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <Dataflow/Network/Module.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <sgi_stl_warnings_on.h>

#include <Core/Algorithms/Math/share.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class SparseElement; 
typedef std::vector<SparseElement> SparseElementVector;

class SCISHARE MathAlgo : public AlgoLibrary {

  public:
    MathAlgo(ProgressReporter* pr); // normal case

    // Build the FEMMatrix using a variable number of processes
    bool BuildFEMatrix(FieldHandle field, MatrixHandle& matrix, int num_proc, MatrixHandle ConductivityTable = 0, MatrixHandle GeomToComp = 0, MatrixHandle CompToGeom = 0);
    
    // CreateFEDirichletBC()
    bool CreateFEDirichletBC(MatrixHandle FEin, MatrixHandle RHSin, MatrixHandle BC, MatrixHandle& FEout, MatrixHandle& RHSout);
    
    // Resize a matrix, Dense or Sparse
    bool ResizeMatrix(MatrixHandle input, MatrixHandle& output, int m, int n);
    
    // Build a sparse matrix based on coordinates of elements
    bool CreateSparseMatrix(SparseElementVector& input, MatrixHandle& output, int m, int n);

    // Recursive CutHill-mcKee
    bool ReverseCuthillmcKee(MatrixHandle input,MatrixHandle& output,MatrixHandle& mapping,bool calcmapping = true);

    // CutHill-mcKee
    bool CuthillmcKee(MatrixHandle input,MatrixHandle& output,MatrixHandle& mapping,bool calcmapping = true);
    
    // Apply an operation on a row by row basis
    bool ApplyRowOperation(MatrixHandle input, MatrixHandle& output, std::string operation); 

    // Apply an operation on a column by column basis
    bool ApplyColumnOperation(MatrixHandle input, MatrixHandle& output, std::string operation); 
    
};


// helper classes

class SparseElement {
public:
  int     row;
  int     col;
  double  val;
};

inline bool operator==(const SparseElement& s1,const SparseElement& s2)
{
  if ((s1.row == s2.row)&&(s1.col == s2.col)) return (true);
  return (false);
}    

inline bool operator<(const SparseElement& s1, const SparseElement& s2)
{
  if (s1.row < s2.row) return(true);
  if (s1.row == s2.row) if (s1.col < s2.col) return(true);
  return (false);
}


} // end SCIRun namespace
#endif

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


//    File   : LinearAlgebra.h
//    Author : Michael Callahan
//    Date   : June 2002

#if !defined(LinearAlgebra_h)
#define LinearAlgebra_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/DenseColMajMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/MatrixOperations.h>

namespace SCIRun {

class LinearAlgebraAlgo : public DynamicAlgoBase
{
public:
  virtual void user_function(MatrixHandle &o1,
			     MatrixHandle &o2,
			     MatrixHandle &o3,
			     MatrixHandle &o4,
			     MatrixHandle &o5,
			     const MatrixHandle &i1,
			     const MatrixHandle &i2,
			     const MatrixHandle &i3,
			     const MatrixHandle &i4,
			     const MatrixHandle &i5) = 0;

  virtual string identify() = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(int matrixcount,
					    const string &function,
					    int hashoffset);
};


} // end namespace SCIRun

#endif // LinearAlgebra_h

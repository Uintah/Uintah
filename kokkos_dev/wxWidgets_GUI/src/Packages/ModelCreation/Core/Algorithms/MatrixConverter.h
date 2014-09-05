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


#ifndef MODELCREATION_CORE_ALGORITHMS_MATRIXCONVERTER_H
#define MODELCREATION_CORE_ALGORITHMS_MATRIXCONVERTER_H 1

#include <Packages/ModelCreation/Core/Util/AlgoLibrary.h>

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Transform.h>

namespace ModelCreation {

using namespace SCIRun;

class MatrixConverter : public AlgoLibrary {

public:
  MatrixConverter(ProgressReporter* pr);
  
  bool MatrixToDouble(MatrixHandle matrix, double &val);
  bool MatrixToInt(MatrixHandle matrix, int &val);
  bool MatrixToVector(MatrixHandle matrix, Vector& vec);
  bool MatrixToTensor(MatrixHandle matrix, Tensor& ten);
  bool MatrixToTransform(MatrixHandle matrix, Transform& trans);
  
  bool DoubleToMatrix(double val, MatrixHandle& matrix);
  bool IntToMatrix(int val, MatrixHandle& matrix);
  bool VectorToMatrix(Vector& vec, MatrixHandle& matrix);
  bool TensorToMatrix(Tensor& ten, MatrixHandle matrix);
  bool TransformToMatrix(Transform& trans, MatrixHandle& matrix);

};

} // end namespace

#endif

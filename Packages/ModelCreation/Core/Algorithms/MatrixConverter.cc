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

#include <Packages/ModelCreation/Core/Algorithms/MatrixConverter.h>

namespace ModelCreation {

using namespace SCIRun;

MatrixConverter::MatrixConverter(ProgressReporter* pr) :
  AlgoLibrary(pr)
{
}

bool MatrixConverter::MatrixToDouble(MatrixHandle matrix, double &val)
{
  if ((matrix->nrows() * matrix->ncols()) != 1)
  {
    error("MatrixToDouble: Matrix has not dimensions 1 x 1");
    return (false);
  }
  
  MatrixHandle mat = dynamic_cast<Matrix*>(matrix->dense());
  if (mat.get_rep() == 0)
  {
    error("MatrixToDouble: Matrix could not be translated into a dense matrix");
    return (false);    
  }
  
  val = mat->get(0,0);
  return (true);
}

bool MatrixConverter::MatrixToInt(MatrixHandle matrix, int &val)
{
  if ((matrix->nrows() * matrix->ncols()) != 1)
  {
    error("MatrixToInt: Matrix has not dimensions 1 x 1");
    return (false);
  }
  
  MatrixHandle mat = dynamic_cast<Matrix*>(matrix->dense());
  if (mat.get_rep() == 0)
  {
    error("MatrixToDouble: Matrix could not be translated into a dense matrix");
    return (false);    
  }
  
  double temp = mat->get(0,0);   
  val = static_cast<int>(temp);
  
  if ((temp - static_cast<double>(val)) != 0.0)
  {
    warning("MatrixToInt: Value in matrix is not of integer value, rounding value to nearest integer value");
  }  
  return (true);
}

bool MatrixConverter::MatrixToVector(MatrixHandle matrix, Vector& vec)
{

  MatrixHandle mat = dynamic_cast<Matrix*>(matrix->dense());
  if (mat.get_rep() == 0)
  {
    error("MatrixToVector: Matrix could not be translated into a dense matrix");
    return (false);    
  }
  
  double* data = mat->get_data_pointer();
  if (data == 0)
  {
    error("MatrixToVector: Could not access the matrix data");
    return (false);      
  }
  
  if ((mat->nrows() * mat->ncols()) == 1)
  {
    vec = Vector(data[0],data[0],data[0]);
    return (true);
  }
  
  if ((mat->nrows() * mat->ncols()) == 2)
  {  
    vec = Vector(data[0],data[1],data[1]);
    return (true);  
  }

  if ((mat->nrows() * mat->ncols()) == 3)
  {    
    vec = Vector(data[0],data[1],data[2]);
    return (true);
  }
  
  error("MatrixToVector: Improper matrix dimensions");
  return (false);
}

bool MatrixConverter::MatrixToTensor(MatrixHandle matrix, Tensor& ten)
{
  MatrixHandle mat = dynamic_cast<Matrix*>(matrix->dense());
  if (mat.get_rep() == 0)
  {
    error("MatrixToTensor: Matrix could not be translated into a dense matrix");
    return (false);    
  }
  
  double* data = mat->get_data_pointer();
  if (data == 0)
  {
    error("MatrixToTensor: Could not access the matrix data");
    return (false);      
  }

  if ((mat->nrows() * mat->ncols()) == 1)
  {
    ten = data[0];
    return (true);
  }

  if (((mat->nrows() == 1)&&(mat->ncols() == 6)) ||
      ((mat->nrows() == 6)&&(mat->ncols() == 1)))
  {
    ten = Tensor(data);
    return (true);
  }

  if (((mat->nrows() == 1)&&(mat->ncols() == 9)) ||
      ((mat->nrows() == 9)&&(mat->ncols() == 1)) ||
      ((mat->nrows()==3)&&(mat->ncols()==3)))
  {
    double tdata[6];
    tdata[0] = data[0]; tdata[1] = data[1]; tdata[2] = data[2];
    tdata[3] = data[4];  tdata[4] = data[5]; tdata[5] = data[8];
    ten = Tensor(tdata);
  }

  error("MatrixToTensor: Improper matrix dimensions");
  return (false);  
}

bool MatrixConverter::MatrixToTransform(MatrixHandle matrix, Transform& trans)
{
  MatrixHandle mat = dynamic_cast<Matrix*>(matrix->dense());
  if (mat.get_rep() == 0)
  {
    error("MatrixToTransform: Matrix could not be translated into a dense matrix");
    return (false);    
  }
  
  double* data = mat->get_data_pointer();
  if (data == 0)
  {
    error("MatrixToTransform: Could not access the matrix data");
    return (false);      
  }

  if ((mat->nrows() * mat->ncols()) == 1)
  {
    trans.load_identity();
    trans.post_scale(Vector(data[0],data[0],data[0]));
    return (true);
  }

  if (((mat->nrows() == 1)&&(mat->ncols() == 16)) ||
      ((mat->nrows() == 16)&&(mat->ncols() == 1)) ||
      ((mat->nrows()==4)&&(mat->ncols()==4)))
  {
    trans.set(data);
    return (true);
  }
  
  error("MatrixToTransform: Improper matrix dimensions");
  return (false);    
}

} // end namespace

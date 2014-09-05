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

#include <Core/Algorithms/Converter/ConverterAlgo.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <sgi_stl_warnings_on.h>

#include <Core/Algorithms/Converter/NrrdToField.h>
#include <Core/Algorithms/Converter/MatrixToField.h>
#include <Core/Algorithms/Converter/MatricesToDipoleField.h>

namespace SCIRunAlgo {

using namespace SCIRun;

ConverterAlgo::ConverterAlgo(ProgressReporter* pr) :
  AlgoLibrary(pr)
{
}

bool ConverterAlgo::MatrixToDouble(MatrixHandle matrix, double &val)
{
  if (matrix.get_rep() == 0) return (false);

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

bool ConverterAlgo::MatrixToInt(MatrixHandle matrix, int &val)
{
  if (matrix.get_rep() == 0) return (false);

  if ((matrix->nrows() * matrix->ncols()) != 1)
  {
    error("MatrixToInt: Matrix has not dimensions 1 x 1");
    return (false);
  }
  
  MatrixHandle mat = dynamic_cast<Matrix*>(matrix->dense());
  if (mat.get_rep() == 0)
  {
    error("MatrixToInt: Matrix could not be translated into a dense matrix");
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

bool ConverterAlgo::MatrixToVector(MatrixHandle matrix, Vector& vec)
{
  if (matrix.get_rep() == 0) return (false);

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

bool ConverterAlgo::MatrixToTensor(MatrixHandle matrix, Tensor& ten)
{
  if (matrix.get_rep() == 0) return (false);

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

bool ConverterAlgo::MatrixToTransform(MatrixHandle matrix, Transform& trans)
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


bool ConverterAlgo::DoubleToMatrix(double val, MatrixHandle& matrix)
{
  matrix = dynamic_cast<Matrix*>(scinew DenseMatrix(1,1));
  if (matrix.get_rep() == 0) 
  {
    error("DoubleToMatrix: Could not allocate memory");
    return (false);
  }
  matrix->put(0,0,val);
  return (true);
}

bool ConverterAlgo::IntToMatrix(int val, MatrixHandle& matrix)
{
  matrix = dynamic_cast<Matrix*>(scinew DenseMatrix(1,1));
  if (matrix.get_rep() == 0) 
  {
    error("IntToMatrix: Could not allocate memory");
    return (false);
  }
  matrix->put(0,0,static_cast<int>(val));
  return (true);
}

bool ConverterAlgo::VectorToMatrix(Vector& vec, MatrixHandle& matrix)
{
  matrix = dynamic_cast<Matrix*>(scinew DenseMatrix(3,1));
  if (matrix.get_rep() == 0) 
  {
    error("VectorToMatrix: Could not allocate memory");
    return (false);
  }
  matrix->put(0,0,vec.x());
  matrix->put(1,0,vec.y());
  matrix->put(2,0,vec.z());
  return (true);
}

bool ConverterAlgo::TensorToMatrix(Tensor& ten, MatrixHandle matrix)
{
  matrix = dynamic_cast<Matrix*>(scinew DenseMatrix(3,3));
  if (matrix.get_rep() == 0) 
  {
    error("TensorToMatrix: Could not allocate memory");
    return (false);
  }
  matrix->put(0,0,ten.mat_[0][0]);
  matrix->put(1,0,ten.mat_[1][0]);
  matrix->put(2,0,ten.mat_[2][0]);
  matrix->put(0,1,ten.mat_[0][1]);
  matrix->put(1,1,ten.mat_[1][1]);
  matrix->put(2,1,ten.mat_[2][1]);
  matrix->put(0,2,ten.mat_[0][2]);
  matrix->put(1,2,ten.mat_[1][2]);
  matrix->put(2,2,ten.mat_[2][2]);

  return (true);
}

bool ConverterAlgo::TransformToMatrix(Transform& trans, MatrixHandle& matrix)
{
  matrix = dynamic_cast<Matrix*>(scinew DenseMatrix(4,4));
  if (matrix.get_rep() == 0) 
  {
    error("DoubleToMatrix: Could not allocate memory");
    return (false);
  }

  double *dptr;
  double sptr[16];
  
  trans.get(sptr);
  dptr = matrix->get_data_pointer();
  for (int p=0;p<16;p++) dptr[p] = sptr[p];
  
  return (true);
}


bool ConverterAlgo::MatricesToDipoleField(MatrixHandle locations,MatrixHandle strengths,FieldHandle& Dipoles)
{
  MatricesToDipoleFieldAlgo algo;
  return(algo.MatricesToDipoleField(pr_,locations,strengths,Dipoles));
}


bool ConverterAlgo::MatrixToField(MatrixHandle input, FieldHandle& output,std::string datalocation)
{
  MatrixToFieldAlgo algo;
  return(algo.MatrixToField(pr_,input,output,datalocation));
}


bool ConverterAlgo::NrrdToField(NrrdDataHandle input, FieldHandle& output,std::string datalocation)
{
  NrrdToFieldAlgo algo;
  return(algo.NrrdToField(pr_,input,output,datalocation));
}


} // end namespace SCIRunAlgo


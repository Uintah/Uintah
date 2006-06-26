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

//    File       : Tikhonov.cc
//    Author     : Yesim Serinagaoglu & Alireza Ghodrati
//    Date       : 07 Aug. 2001
//    Last update: Nov. 2002


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <stdio.h>
#include <math.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>
using std::ostringstream;


namespace BioPSE
{

using namespace SCIRun;

class Tikhonov : public Module
{
  GuiDouble     lambda_fix_;
  GuiDouble     lambda_sld_;
  GuiInt        have_ui_;
  GuiString     reg_method_;
  GuiDouble     lambda_min_;
  GuiDouble     lambda_max_;
  GuiInt        lambda_num_;
  GuiDouble     tex_var_;

public:
  //! Constructor
  Tikhonov(GuiContext *context);

  //! Destructor
  virtual ~Tikhonov();

  virtual void execute();

  DenseMatrix  * mat_identity(int len);
  DenseMatrix  * mat_trans_mult_mat(DenseMatrix *A);
  DenseMatrix  * mat_mult(DenseMatrix *A, DenseMatrix *B);

  double FindCorner(Array1<double>  &rho, Array1<double>  &eta,
                    Array1<double>  &lambdaArray,
                    ColumnMatrix *kapa, int *lambda_index, int nLambda);
};

//! Module Maker
DECLARE_MAKER(Tikhonov)


//! Constructor
Tikhonov::Tikhonov(GuiContext *context) :
  Module("Tikhonov", context, Source, "Inverse", "BioPSE"),
  lambda_fix_(context->subVar("lambda_fix")),
  lambda_sld_(context->subVar("lambda_sld")),
  have_ui_(context->subVar("have_ui")),
  reg_method_(context->subVar("reg_method")),
  lambda_min_(context->subVar("lambda_min")),
  lambda_max_(context->subVar("lambda_max")),
  lambda_num_(context->subVar("lambda_num")),
  tex_var_(context->subVar("tex_var"))
{
}


//! Destructor
Tikhonov::~Tikhonov()
{
}


//! Create Identity Matrix
DenseMatrix *
Tikhonov::mat_identity(int len)
{
  DenseMatrix *eye = scinew DenseMatrix(len, len);
  // Does this make sure all the elements are 0?
  eye->zero();
  for(int i=0; i<len; i++)
  {
    eye->put(i, i, 1.0);
  }
  return eye;
}


//! This function computes A^T * A for a DenseMatrix
DenseMatrix *
Tikhonov::mat_trans_mult_mat(DenseMatrix *A)
{
  int nRows = A->nrows();
  int nCols = A->ncols();
  int beg = -1;
  int end = -1;
  int i, j; // i: column index, j: row index
  int flops, memrefs;

  DenseMatrix *B = scinew DenseMatrix(nCols, nCols);
  ColumnMatrix *Ai = scinew ColumnMatrix(nRows);
  ColumnMatrix *Bi = scinew ColumnMatrix(nCols);

  // For each column (i) of A, first create a column vector Ai = A[:][i]
  // Bi is then the i'th column of AtA, Bi = At * Ai

  for (i=0; i<nCols; i++)
  {
    // build copy of this column
    for (j=0; j<nRows; j++)
    {
      (*Ai)[j] = (*A)[j][i];
    }
    A->mult_transpose(*Ai, *Bi, flops, memrefs, beg, end);
    for (j=0; j<nCols; j++)
    {
      (*B)[j][i] = (*Bi)[j];
    }
  }
  return B;
}


//! This function returns the multiplication of A and B
DenseMatrix *
Tikhonov::mat_mult(DenseMatrix *A, DenseMatrix *B)
{
  int nRows = B->nrows();
  int nCols = B->ncols();
  int beg = -1;
  int end = -1;
  int i, j; // i: column index, j: row index
  int flops, memrefs;
  DenseMatrix *C = scinew DenseMatrix(A->nrows(), B->ncols());
  ColumnMatrix *Ci = scinew ColumnMatrix(A->nrows());
  ColumnMatrix *Bi = scinew ColumnMatrix(B->nrows());
  // For each column (i) of C, first create a column vector Bi = B[:][i]
  // Ci is then the i'th column of A*B, Ci = A * Bi

  for (i=0; i<nCols; i++)
  {
    // build copy of this column
    for (j=0; j<nRows; j++)
    {
      (*Bi)[j] = (*B)[j][i];
    }
    A->mult(*Bi, *Ci, flops, memrefs, beg, end);
    for (j=0; j<nCols; j++)
    {
      (*C)[j][i] = (*Ci)[j];
    }
  }
  return C;
}


//! Find Corner
double
Tikhonov::FindCorner(Array1<double> &rho, Array1<double> &eta,
                     Array1<double> &lambdaArray,
                     ColumnMatrix *kapa, int *lambda_index, int nLambda)
{
  Array1<double> deta, ddeta, drho, ddrho, lrho, leta;

  leta.setsize(nLambda);
  deta.setsize(nLambda);
  ddeta.setsize(nLambda);
  lrho.setsize(nLambda);
  drho.setsize(nLambda);
  ddrho.setsize(nLambda);

  double  maxKapa=-1e10;
  int   i;

  for(i=0; i<nLambda; i++)
  {
    lrho[i] = log(rho[i])/log(10.0);
    leta[i] = log(eta[i])/log(10.0);
    if(i>0)
    {
      deta[i] = (leta[i]-leta[i-1])/(lambdaArray[i]-lambdaArray[i-1]);
      drho[i] = (lrho[i]-lrho[i-1])/(lambdaArray[i]-lambdaArray[i-1]);
    }
    if(i>1)
    {
      ddeta[i] = (deta[i]-deta[i-1])/(lambdaArray[i]-lambdaArray[i-1]);
      ddrho[i] = (drho[i]-drho[i-1])/(lambdaArray[i]-lambdaArray[i-1]);
    }
  }
  drho[0] = drho[1];
  deta[0] = deta[1];
  ddrho[0] = ddrho[2];
  ddrho[1] = ddrho[2];
  ddeta[0] = ddeta[2];
  ddeta[1] = ddeta[2];

  *lambda_index=0;
  for(i=0; i<nLambda; i++)
  {
    (*kapa)[i] = 2*(drho[i]*ddeta[i] - ddrho[i]*deta[i])/sqrt(pow((deta[i]*deta[i]+drho[i]*drho[i]),3));
    if((*kapa)[i]>maxKapa)
    {
      maxKapa = (*kapa)[i];
      *lambda_index = i;
    }
  }
  double lambda_cor = lambdaArray[*lambda_index];
  return lambda_cor;
}


//! Module execution
void
Tikhonov::execute()
{
  MatrixIPort *iportRegMat = (MatrixIPort *)get_iport("RegularizationMat");

  // DEFINE MATRIX HANDLES FOR INPUT/OUTPUT PORTS
  MatrixHandle hMatrixForMat, hMatrixRegMat, hMatrixMeasDat;
  if (!get_input_handle("ForwardMat", hMatrixForMat, true)) return;
  if (!get_input_handle("MeasuredPots", hMatrixMeasDat, true)) return;

  // TYPE CHECK
  DenseMatrix *matrixForMatD = hMatrixForMat->dense();
  ColumnMatrix *matrixMeasDatD = hMatrixMeasDat->column();

  // DIMENSION CHECK!!
  const int M = matrixForMatD->nrows();
  const int N = matrixForMatD->ncols();
  if (M != matrixMeasDatD->nrows())
  {
    error("Input matrix dimensions must agree.");
    return;
  }

  //.........................................................................
  // OPERATE ON DATA:
  // SOLVE (A^T * A + LAMBDA * LAMBDA * R^T * R) * X = A^T * Y        FOR "X"
  //.........................................................................

  // calculate A^T * A
  DenseMatrix *mat_AtrA = mat_mult(matrixForMatD->transpose(),matrixForMatD);

  // calculate R^T * R
  DenseMatrix *mat_RtrR, *matrixRegMatD;

  if (!iportRegMat->get(hMatrixRegMat) && !hMatrixRegMat.get_rep())
  {
    matrixRegMatD = mat_identity(matrixForMatD->ncols());
    mat_RtrR = mat_identity(matrixForMatD->ncols());
  }
  else
  {
    matrixRegMatD = hMatrixRegMat->dense();
    if (N != matrixRegMatD->ncols())
    {
      error("The dimension of RegularizationMat is not compatible with ForwardMat.");
      return;
    }
    mat_RtrR = mat_trans_mult_mat(matrixRegMatD);
  }

  int beg = -1;
  int end = -1;
  double lambda = 0, lambda2 = 0;
  double temp;
  int   lambda_index;

  // calculate A^T * Y
  int flops, memrefs;
  ColumnMatrix *mat_AtrY = scinew ColumnMatrix(N);
  matrixForMatD->mult_transpose(*matrixMeasDatD, *mat_AtrY, flops, memrefs);

  DenseMatrix *regForMatrix = scinew DenseMatrix(N, N);
  ColumnMatrix *solution = scinew ColumnMatrix(N);
  ColumnMatrix *Ax = scinew ColumnMatrix(M);
  ColumnMatrix *Rx = scinew ColumnMatrix(N);

  if ((reg_method_.get() == "single") || (reg_method_.get() == "slider"))
  {
    if (reg_method_.get() == "single")
    {
      // Use single fixed lambda value, entered in UI
      lambda = lambda_fix_.get();
      msg_stream_ << "  method = " << reg_method_.get() << "\n";//DISCARD
    }
    else if (reg_method_.get() == "slider")
    {
      // Use single fixed lambda value, select via slider
      lambda = tex_var_.get(); //lambda_sld_.get();
      msg_stream_ << "  method = " << reg_method_.get() << "\n";//DISCARD
    }
  }
  else if (reg_method_.get() == "lcurve")
  {
    // Use L-curve, lambda from corner of the L-curve
    msg_stream_ << "method = " << reg_method_.get() << "\n";//DISCARD

    int i, j, k, l;
    Array1<double> lambdaArray, rho, eta;
    const int nLambda = lambda_num_.get();

    ColumnMatrix *kapa = scinew ColumnMatrix(nLambda);

    lambdaArray.setsize(nLambda);
    rho.setsize(nLambda);
    eta.setsize(nLambda);

    lambdaArray[0] = lambda_min_.get();
    const double lam_step =
      pow(10.0, log10(lambda_max_.get() / lambda_min_.get()) / (nLambda-1));

    for (j=0; j<nLambda; j++)
    {
      if (j)
      {
        lambdaArray[j] = lambdaArray[j-1] * lam_step;
      }

      lambda2 = lambdaArray[j] * lambdaArray[j];

      ///////////////////////////////////////
      ////Calculating the solution directly
      ///////////////////////////////////////
      for (i=0; i<N; i++)
      {
        for (l=0; l<N; l++)
        {
          temp = (*mat_RtrR)[i][l];
          (*regForMatrix)[i][l] = (*mat_AtrA)[i][l] + lambda2 * temp;
        }
      }

      for (k=0; k<N; k++)
      {
        (*solution)[k] = (*mat_AtrY)[k];
      }

      //Before, solution will be equal to (A^T * y)
      //After, solution will be equal to x_reg

      regForMatrix->solve(*solution);
      ////////////////////////////////
      matrixForMatD->mult(*solution, *Ax, flops, memrefs, beg, end);
      matrixRegMatD->mult(*solution, *Rx, flops, memrefs, beg, end);
      rho[j] = 0;
      eta[j] = 0;

      // Calculate the norm of Ax-b and Rx

      for (k=0; k<M; k++)
      {
        (*Ax)[k] = (*Ax)[k]-(*matrixMeasDatD)[k];
        rho[j] = rho[j] + (*Ax)[k]*(*Ax)[k];
      }
      for (k=0; k<N; k++)
      {
        eta[j] = eta[j] + (*Rx)[k]*(*Rx)[k];
      }
      rho[j] = sqrt(rho[j]);
      eta[j] = sqrt(eta[j]);
    }

    lambda = FindCorner(rho, eta, lambdaArray, kapa, &lambda_index, nLambda);

    double lower_y = eta[0] / 10.0;
    if (eta[nLambda-1] < lower_y)
    {
      lower_y = eta[nLambda-1];
    }

    if (have_ui_.get())
    {
      ostringstream str;
      str << get_id() << " plot_graph \" ";
      for (i=0; i<nLambda; i++)
        str << rho[i] << " " << eta[i] << " ";
      str << "\" \" " << rho[0]/10 << " " << eta[lambda_index] << " ";
      str << rho[lambda_index] << " " << eta[lambda_index] << " ";
      str << rho[lambda_index] << " " << lower_y << " \" ";
      str << lambda << " ; update idletasks";
      get_gui()->execute(str.str().c_str());
    }
  } // END  else if (reg_method_.get() == "lcurve")
  lambda2 = lambda*lambda;

  ColumnMatrix *RegParameter = scinew ColumnMatrix(1);
  (*RegParameter)[0] = lambda;

  for (int i=0; i<N; i++)
  {
    for (int l=0; l<N; l++)
    {
      temp = (*mat_RtrR)[i][l];
      (*regForMatrix)[i][l] = (*mat_AtrA)[i][l] + lambda2 * temp;
    }
  }
  regForMatrix->solve(*mat_AtrY);
  DenseMatrix *InverseMatrix = scinew DenseMatrix(N, M);

  regForMatrix->invert();

  Mult(*InverseMatrix, *regForMatrix, *(matrixForMatD->transpose()));

  //...........................................................
  // SEND RESULTS TO THE OUTPUT PORTS
  MatrixHandle AtrYHandle(mat_AtrY);
  send_output_handle("InverseSoln", AtrYHandle);
  MatrixHandle RegParameterHandle(RegParameter);
  send_output_handle("RegParam", RegParameterHandle);
  MatrixHandle InverseMatrixHandle(InverseMatrix);
  send_output_handle("RegInverseMat", InverseMatrixHandle);
}

} // End namespace BioPSE

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
#include <Core/Datatypes/MatrixOperations.h>
#include <Dataflow/GuiInterface/GuiVar.h>
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

  DenseMatrix *mat_trans_mult_mat(const DenseMatrix &A);

  double FindCorner(const vector<double> &rho, const vector<double> &eta,
                    const vector<double> &lambdaArray,
                    ColumnMatrix &kapa, int *lambda_index, int nLambda);
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


//! This function computes A^T * A for a DenseMatrix
// This is probably just an inefficient implementation of Mult_trans_X
// from DenseMatrix.h.  No regression test available though.
DenseMatrix *
Tikhonov::mat_trans_mult_mat(const DenseMatrix &A)
{
  const int nRows = A.nrows();
  const int nCols = A.ncols();
  int i, j; // i: column index, j: row index
  int flops, memrefs;

  DenseMatrix *B = scinew DenseMatrix(nCols, nCols);
  ColumnMatrix Ai(nRows);
  ColumnMatrix Bi(nCols);

  // For each column (i) of A, first create a column vector Ai = A[:][i]
  // Bi is then the i'th column of AtA, Bi = At * Ai

  for (i=0; i<nCols; i++)
  {
    // build copy of this column
    for (j=0; j<nRows; j++)
    {
      Ai[j] = A[j][i];
    }
    A.mult_transpose(Ai, Bi, flops, memrefs);
    for (j=0; j<nCols; j++)
    {
      (*B)[j][i] = Bi[j];
    }
  }
 
  return B;
}


//! Find Corner
double
Tikhonov::FindCorner(const vector<double> &rho, const vector<double> &eta,
                     const vector<double> &lambdaArray,
                     ColumnMatrix &kapa, int *lambda_index, int nLambda)
{
  vector<double> deta(nLambda, 0.0);
  vector<double> ddeta(nLambda, 0.0);
  vector<double> drho(nLambda, 0.0);
  vector<double> ddrho(nLambda, 0.0);
  vector<double> lrho(nLambda, 0.0);
  vector<double> leta(nLambda, 0.0);

  double  maxKapa = -1.0e10;
  int i;
  for (i=0; i<nLambda; i++)
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

  *lambda_index = 0;
  for (i=0; i<nLambda; i++)
  {
    kapa[i] = 2.0 * (drho[i] * ddeta[i] - ddrho[i] * deta[i]) / 
      sqrt(pow((deta[i]*deta[i]+drho[i]*drho[i]), 3.0));
    if (kapa[i] > maxKapa)
    {
      maxKapa = kapa[i];
      *lambda_index = i;
    }
  }

  return lambdaArray[*lambda_index];
}


//! Module execution
void
Tikhonov::execute()
{
  MatrixIPort *iportRegMat = (MatrixIPort *)get_iport("RegularizationMat");

  // DEFINE MATRIX HANDLES FOR INPUT/OUTPUT PORTS
  MatrixHandle forward_matrix_h, hMatrixRegMat, hMatrixMeasDat;
  if (!get_input_handle("ForwardMat", forward_matrix_h, true)) return;
  if (!get_input_handle("MeasuredPots", hMatrixMeasDat, true)) return;

  // TYPE CHECK
  MatrixHandle matrixForMat_handle = forward_matrix_h->dense();
  DenseMatrix &matrixForMatD = *(matrixForMat_handle->as_dense());
  MatrixHandle matrixMeasDat_handle = hMatrixMeasDat->column();
  ColumnMatrix &matrixMeasDatD = *(matrixMeasDat_handle->as_column());

  // DIMENSION CHECK!!
  const int M = forward_matrix_h->nrows();
  const int N = forward_matrix_h->ncols();
  if (M != matrixMeasDatD.nrows())
  {
    error("Input matrix dimensions must agree.");
    return;
  }

  //.........................................................................
  // OPERATE ON DATA:
  // SOLVE (A^T * A + LAMBDA * LAMBDA * R^T * R) * X = A^T * Y        FOR "X"
  //.........................................................................

  // calculate A^T * A
  MatrixHandle forward_transpose_h = forward_matrix_h->transpose();
  MatrixHandle mat_AtrA_h = (forward_transpose_h * forward_matrix_h)->dense();
  DenseMatrix &mat_AtrA = *(mat_AtrA_h->as_dense());

  // calculate R^T * R
  MatrixHandle mat_RtrR_handle;
  MatrixHandle matrixRegMat_handle;
  if (!iportRegMat->get(hMatrixRegMat) && !hMatrixRegMat.get_rep())
  {
    matrixRegMat_handle = DenseMatrix::identity(N);
    mat_RtrR_handle = DenseMatrix::identity(N);
  }
  else
  {
    matrixRegMat_handle = hMatrixRegMat->dense();
    if (N != matrixRegMat_handle->ncols())
    {
      error("The dimension of RegularizationMat is not compatible with ForwardMat.");
      return;
    }
    mat_RtrR_handle = mat_trans_mult_mat(*(matrixRegMat_handle->as_dense()));
  }
  DenseMatrix &mat_RtrR = *(mat_RtrR_handle->as_dense());
  DenseMatrix &matrixRegMatD = *(matrixRegMat_handle->as_dense());

  double lambda = 0, lambda2 = 0;
  int lambda_index;

  // calculate A^T * Y
  int flops, memrefs;
  MatrixHandle AtrYHandle = scinew ColumnMatrix(N);
  ColumnMatrix &mat_AtrY = *(AtrYHandle->as_column());
  matrixForMatD.mult_transpose(matrixMeasDatD, mat_AtrY, flops, memrefs);

  MatrixHandle regForMatrix_handle = scinew DenseMatrix(N, N);
  DenseMatrix &regForMatrix = *(regForMatrix_handle->as_dense());
  MatrixHandle solution_handle = scinew ColumnMatrix(N);
  MatrixHandle Ax_handle = scinew ColumnMatrix(M);
  MatrixHandle Rx_handle = scinew ColumnMatrix(N);
  ColumnMatrix &solution = *(solution_handle->as_column());
  ColumnMatrix &Ax = *(Ax_handle->as_column());
  ColumnMatrix &Rx = *(Rx_handle->as_column());

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
    const int nLambda = lambda_num_.get();

    ColumnMatrix *kapa = scinew ColumnMatrix(nLambda);

    vector<double> lambdaArray(nLambda, 0.0);
    vector<double> rho(nLambda, 0.0);
    vector<double> eta(nLambda, 0.0);

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
          regForMatrix[i][l] = mat_AtrA[i][l] + lambda2 * mat_RtrR[i][l];
        }
      }

      for (k=0; k<N; k++)
      {
        solution[k] = mat_AtrY[k];
      }

      //Before, solution will be equal to (A^T * y)
      //After, solution will be equal to x_reg

      regForMatrix.solve(solution);
      ////////////////////////////////
      matrixForMatD.mult(solution, Ax, flops, memrefs);
      matrixRegMatD.mult(solution, Rx, flops, memrefs);
      rho[j] = 0.0;
      eta[j] = 0.0;

      // Calculate the norm of Ax-b and Rx

      for (k=0; k<M; k++)
      {
        Ax[k] -= matrixMeasDatD[k];
        rho[j] += Ax[k] * Ax[k];
      }
      for (k=0; k<N; k++)
      {
        eta[j] += Rx[k] * Rx[k];
      }
      rho[j] = sqrt(rho[j]);
      eta[j] = sqrt(eta[j]);
    }

    lambda = FindCorner(rho, eta, lambdaArray, *kapa, &lambda_index, nLambda);

    if (have_ui_.get())
    {
      const double lower_y = Min(eta[0] / 10.0, eta[nLambda-1]);
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
    delete kapa;
  } // END  else if (reg_method_.get() == "lcurve")

  lambda2 = lambda * lambda;

  MatrixHandle RegParameter = scinew ColumnMatrix(1);
  RegParameter->put(0, 0, lambda);

  for (int i=0; i<N; i++)
  {
    for (int l=0; l<N; l++)
    {
      regForMatrix[i][l] = mat_AtrA[i][l] + lambda2 * mat_RtrR[i][l];
    }
  }
  regForMatrix.solve(mat_AtrY);
  regForMatrix.invert();

  MatrixHandle inverse_matrix = regForMatrix_handle * forward_transpose_h;

  //...........................................................
  // SEND RESULTS TO THE OUTPUT PORTS
  send_output_handle("InverseSoln", AtrYHandle);
  send_output_handle("RegParam", RegParameter);
  send_output_handle("RegInverseMat", inverse_matrix);
}

} // End namespace BioPSE

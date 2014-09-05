/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  
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

//    File       : SolveInverseProblemWithTikhonovSVD.cc
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
#include <Dataflow/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>
using std::ostringstream;

#define	SVD_method	1
#define	GSVD_method 	2


namespace BioPSE 
{

using namespace SCIRun;

class SolveInverseProblemWithTikhonovSVD : public Module 
{
  GuiDouble 	lambda_fix_;
  GuiDouble 	lambda_sld_;
  GuiInt    	have_ui_;
  GuiString 	reg_method_;
  GuiDouble	lambda_min_;
  GuiDouble	lambda_max_;
  GuiInt		lambda_num_;
  GuiDouble 	tex_var_;	

public:
  // CONSTRUCTOR
  SolveInverseProblemWithTikhonovSVD(GuiContext *context);

  // DESTRUCTOR
  virtual ~SolveInverseProblemWithTikhonovSVD();

  virtual void execute();

  
  double Inner_Product(DenseMatrix& A, int col_num, ColumnMatrix& w);
  void	tikhonov_fun(ColumnMatrix& X_reg, DenseMatrix& InvMat, DenseMatrix& U, ColumnMatrix& Uy,DenseMatrix& S, DenseMatrix& V, DenseMatrix& X, double lam);
  void	prep_lcurve_data(double *rho, double *eta, ColumnMatrix& Uy, DenseMatrix& U, DenseMatrix& S, DenseMatrix& V, DenseMatrix& X, ColumnMatrix& y, double lam);
  double FindCorner(Array1<double>  &rho, Array1<double>  &eta, Array1<double>  &lambdaArray, 
                    ColumnMatrix *kapa, int *lambda_index, int nLambda);

};
	
// MODULE MAKER
DECLARE_MAKER(SolveInverseProblemWithTikhonovSVD)


  // CONSTRUCTOR
  SolveInverseProblemWithTikhonovSVD::SolveInverseProblemWithTikhonovSVD(GuiContext *context)
    : Module("SolveInverseProblemWithTikhonovSVD", context, Source, "Inverse", "BioPSE"),
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

// DESTRUCTOR
SolveInverseProblemWithTikhonovSVD::~SolveInverseProblemWithTikhonovSVD()
{
}


////////////////////////////////////////////////////////////////////////////
// THIS FUNCTION returns the inner product of one column of matrix A
// and w , B=A(:,i)'*w
///////////////////////////////////////////////////////////////////////////
double
SolveInverseProblemWithTikhonovSVD::Inner_Product(DenseMatrix& A, int col_num, ColumnMatrix& w)
{
  int i;
  int nRows=A.nrows();
  double B=0;
  for(i=0;i<nRows;i++)
    B += A[i][col_num] * w[i];
  return B;
}


//////////////////////////////////////////////////////////////////////
// THIS FUNCTION returns regularized solution by tikhonov method
//////////////////////////////////////////////////////////////////////
void
SolveInverseProblemWithTikhonovSVD::tikhonov_fun(ColumnMatrix& X_reg, DenseMatrix& InvMat,
                          DenseMatrix& U, ColumnMatrix& Uy,DenseMatrix& S,
                          DenseMatrix& V, DenseMatrix& X, double lam)
{
  int i,j;
  int rank=S.nrows();
  double temp;
  DenseMatrix  *Mat_temp;
  if(S.ncols() == 1)
  {
    temp = S[0][0]/(lam*lam + S[0][0] * S[0][0]) * Uy[0];
    for(j=0;j<V.nrows();j++)
    {
      X_reg[j]=temp*V[j][0];
    }
    for(i=1; i < rank; i++)
    {
      temp = S[i][0] / (lam*lam + S[i][0] * S[i][0]) * Uy[i];
      for(j=0;j<V.nrows();j++)
      {
        X_reg[j] = X_reg[j] + temp * V[j][i];
				
      }
    }
		
    //Finding Regularized Inverse Matrix
    Mat_temp = scinew  DenseMatrix(V.nrows(),V.ncols());
    for(i=0; i < rank; i++)
    {
      temp = S[i][0] / (lam*lam + S[i][0] * S[i][0]);
      for(j=0;j<V.nrows();j++)
      {
        (*Mat_temp)[j][i]=temp * V[j][i];
      }
    }
    Mult(InvMat,(*Mat_temp),*(U.transpose()));
  }
  else
  {

    temp = S[0][0]/(lam*lam*S[0][1] * S[0][1] + S[0][0] * S[0][0]) * Uy[0];
    for(j=0;j<X.nrows();j++)
    {
      X_reg[j]=temp*X[j][0];
    }
    for(i=1; i < rank; i++)
    {
      temp = S[i][0] / (lam*lam*S[i][1] * S[i][1] + S[i][0] * S[i][0]) * Uy[i];
      for(j=0;j<X.nrows();j++)
      {
        X_reg[j]+= temp * X[j][i];
      }
    }
    for(i=rank; i < X.ncols(); i++)
    {
      for(j=0;j<X.nrows();j++)
      {
        X_reg[j] += Uy[i] * X[j][i];
      }
    }
		
    //Finding Regularized Inverse Matrix
    Mat_temp = scinew DenseMatrix(X.nrows(),X.ncols());
    (*Mat_temp)=X;
    for(i=0; i < rank; i++)
    {
      temp = S[i][0] / (lam*lam*S[i][1] * S[i][1] + S[i][0] * S[i][0]);
      for(j=0;j<X.nrows();j++)
      {
        (*Mat_temp)[j][i] = temp * X[j][i];
      }
    }
    Mult(InvMat,*Mat_temp,*(U.transpose()));

  }
}


/////////////////////////////////////////////////////////////	
// THIS FUNCTION Calculate ro and eta for lcurve
/////////////////////////////////////////////////////////////
void
SolveInverseProblemWithTikhonovSVD::prep_lcurve_data(double *rho, double *eta, ColumnMatrix& Uy,
                              DenseMatrix& U, DenseMatrix& S, DenseMatrix& V,
                              DenseMatrix& X, ColumnMatrix& y, double lam)
{
  int i,j;
  ColumnMatrix *AX_reg = scinew ColumnMatrix(U.nrows());
  ColumnMatrix *RX_reg = scinew ColumnMatrix(V.nrows());
  ColumnMatrix *X_reg; 
  int rank=S.nrows();
  double temp, temp1;
  if(S.ncols() == 1)
  {
    X_reg = scinew ColumnMatrix(V.nrows());		
    temp = S[0][0]/(lam*lam + S[0][0] * S[0][0]) * Uy[0];
    for(j=0;j<V.nrows();j++)
    {
      (*X_reg)[j]=temp*V[j][0];
    }
    for(j=0;j<U.nrows();j++)
    {
      (*AX_reg)[j]=temp*S[0][0]*U[j][0];
    }
    for(i=1; i < rank; i++)
    {
      temp = S[i][0] / (lam*lam + S[i][0] * S[i][0]) * Uy[i];
      for(j=0;j<V.nrows();j++)
      {
        (*X_reg)[j] = (*X_reg)[j] + temp * V[j][i];
      }
      temp *=S[i][0];
      for(j=0;j<U.nrows();j++)
      {
        (*AX_reg)[j]+= temp*U[j][i];
      }
    }
    // Calculate the norm of Ax-b and Rx  
    *rho=0;
    *eta=0;
    for(i=0; i<U.nrows(); i++)
    {
      (*AX_reg)[i] -= y[i];
      *rho += (*AX_reg)[i]*(*AX_reg)[i];
    }
    for(i=0; i<V.nrows(); i++)
    {

      *eta += (*X_reg)[i]*(*X_reg)[i];
    }
    *rho = sqrt(*rho);
    *eta = sqrt(*eta);

  }
  else
  {
    X_reg = scinew ColumnMatrix(X.nrows());
    temp = S[0][0]/(lam*lam*S[0][1] * S[0][1] + S[0][0] * S[0][0]) * Uy[0];

    for(j=0;j<U.nrows();j++)
    {
      (*AX_reg)[j]=temp*S[0][0]*U[j][0];
    }
    for(j=0;j<V.nrows();j++)
    {
      (*RX_reg)[j]=temp*S[0][1]*V[j][0];
    }
    for(i=1; i < rank; i++)
    {
      temp = S[i][0] / (lam*lam*S[i][1] * S[i][1] + S[i][0] * S[i][0]) * Uy[i];
      temp1 =temp * S[i][0];
      for(j=0;j<U.nrows();j++)
      {
        (*AX_reg)[j]+= temp1 * U[j][i];
      }
      temp1 =temp *S[i][1];
      for(j=0;j<V.nrows();j++)
      {
        (*RX_reg)[j]+= temp1 * V[j][i];
      }
    }
    for(i=rank; i < X.ncols(); i++)
    {
      for(j=0;j<U.nrows();j++)
      {
        (*AX_reg)[j]+= Uy[i] * U[j][i];
      }
    }		

    // Calculate the norm of Ax-b and Rx  
    *rho=0;
    *eta=0;
    for(i=0; i<U.nrows(); i++)
    {
      (*AX_reg)[i] -= y[i];
      *rho += (*AX_reg)[i]*(*AX_reg)[i];
    }
    for(i=0; i<V.nrows(); i++)
    {

      *eta += (*RX_reg)[i]*(*RX_reg)[i];
    }
    *rho = sqrt(*rho);
    *eta = sqrt(*eta);

  }
}


////////////////////////////////////////////
// FIND CORNER
////////////////////////////////////////////
double
SolveInverseProblemWithTikhonovSVD::FindCorner(Array1<double> &rho, Array1<double> &eta,
                        Array1<double> &lambdaArray, ColumnMatrix *kapa,
                        int *lambda_index, int nLambda)
{
  Array1<double> deta, ddeta, drho, ddrho, lrho, leta;

  leta.setsize(nLambda); 
  deta.setsize(nLambda); 
  ddeta.setsize(nLambda);
  lrho.setsize(nLambda);   
  drho.setsize(nLambda);   
  ddrho.setsize(nLambda);   

  double  maxKapa=-1e10;
  int	i;

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


/////////////////////////////////////////
// MODULE EXECUTION
/////////////////////////////////////////
void
SolveInverseProblemWithTikhonovSVD::execute()
{
  // DEFINE MATRIX HANDLES FOR INPUT/OUTPUT PORTS
  MatrixHandle hMatrixMeasDat, hMatrixU, hMatrixS, hMatrixV ,hMatrixX;
    	    
  if (!get_input_handle("U", hMatrixU)) return;
  if (!get_input_handle("S", hMatrixS)) return;
  if (!get_input_handle("V", hMatrixV)) return;

  if (!get_input_handle("MeasuredPots", hMatrixMeasDat)) return;

  // TYPE CHECK
  ColumnMatrix *matrixMeasDatD = hMatrixMeasDat->column();
  DenseMatrix *matrixU = hMatrixU->dense();
  DenseMatrix *matrixS = hMatrixS->dense();	
  DenseMatrix *matrixV = hMatrixV->dense();
  DenseMatrix *matrixX;
	
  int	Method;
  int M, N, i, j;
  double lambda, lambda2;
  int	lambda_index;

  if(matrixS->ncols()==1)
    Method=SVD_method;
  else
    if(matrixS->ncols()==2)
    {
      Method=GSVD_method;
		
      if (!get_input_handle("X", hMatrixX)) return;
      matrixX = hMatrixX->dense();
    }
    else
    {
      error("S matrix dimensions incorrect.");
      return;	
    }
	
  M=matrixU->nrows();

  if(Method == SVD_method)
  {
    N=matrixV->nrows();
    if(matrixS->nrows() != N || matrixU->ncols() != N || matrixMeasDatD->nrows() != M )
    {
      error("Input matrix dimensions incorrect.");
      return;
    }
  }
  else
  {
    N=matrixX->nrows();
    int P=matrixV->nrows();
    if(M<N)
    {
      error("The forward matrix should be overdetermined.");
      return;
    }
    if(matrixX->ncols()!=N || matrixS->nrows()!= P || matrixU->ncols() !=N || P>N || matrixMeasDatD->nrows()!=M)
    {
      error("Input matrix dimensions incorrect.");
      return;
    }
  }
  ColumnMatrix *Uy=scinew ColumnMatrix(matrixU->ncols());
  DenseMatrix  *InverseMat = scinew DenseMatrix(N, M);
  ColumnMatrix *solution = scinew ColumnMatrix(N);
	
	
  for(i=0;i<matrixU->ncols();i++)
    (*Uy)[i]=Inner_Product(*matrixU, i, *matrixMeasDatD);
	
   	   
    
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

    int nLambda;
    Array1<double> lambdaArray, rho, eta;
    double rhotemp, etatemp;
    double lower_y;
    double	lam_step;
		

    nLambda=lambda_num_.get();
    ColumnMatrix *kapa = scinew ColumnMatrix(nLambda);
    lambdaArray.setsize(nLambda); 
    rho.setsize(nLambda);
    eta.setsize(nLambda);   
		
    lambdaArray[0]=lambda_min_.get();
    lam_step=pow(10.0,log10(lambda_max_.get()/lambda_min_.get())/(nLambda-1));
	
		
    for(j=0; j<nLambda; j++)
    {
      if(j) 
        lambdaArray[j] = lambdaArray[j-1]*lam_step;
      prep_lcurve_data(&rhotemp, &etatemp, *Uy, *matrixU, *matrixS, *matrixV, *matrixX, 
                       *matrixMeasDatD, lambdaArray[j]);
      rho[j]=rhotemp;
      eta[j]=etatemp;
    }

    lambda = FindCorner(rho, eta, lambdaArray, kapa, &lambda_index, nLambda);

    lower_y = eta[0]/10.;
    if (eta[nLambda-1] < lower_y)  
      lower_y = eta[nLambda-1];

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
  ColumnMatrix  *RegParameter =scinew ColumnMatrix(1);
  (*RegParameter)[0]=lambda;

  tikhonov_fun(*solution, *InverseMat, *matrixU, *Uy, *matrixS, *matrixV, *matrixX, lambda);

	
  //...........................................................
  // SEND RESULTS TO THE OUTPUT PORTS
  MatrixHandle solution_handle(solution);
  send_output_handle("InverseSoln", solution_handle);

  MatrixHandle RegParameterHandle(RegParameter);
  send_output_handle("RegParam", RegParameterHandle);

  MatrixHandle InverseMatHandle(InverseMat);
  send_output_handle("RegInverseMat", InverseMatHandle);
}
  
} // End namespace BioPSE

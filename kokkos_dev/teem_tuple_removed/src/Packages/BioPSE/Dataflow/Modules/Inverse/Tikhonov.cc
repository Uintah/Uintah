//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File       : Tikhonov.cc
//    Author     : Yesim Serinagaoglu & Alireza Ghodrati
//    Date       : 07 Aug. 2001
//    Last update: Nov. 2002


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/BioPSE/share/share.h>

#include <stdio.h>
#include <math.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
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

class BioPSESHARE Tikhonov : public Module 
{
  GuiDouble 	lambda_fix_;
  GuiDouble 	lambda_sld_;
  GuiInt    	haveUI_;
  GuiString 	reg_method_;
  GuiDouble	lambda_min_;
  GuiDouble	lambda_max_;
  GuiInt	lambda_num_;
  GuiDouble 	tex_var_;	

public:
  //! Constructor
  Tikhonov(GuiContext *context);

  //! Destructor
  virtual ~Tikhonov();

  virtual void execute();

  DenseMatrix *mat_identity(int len);
  DenseMatrix *mat_trans_mult_mat(DenseMatrix *A);
  DenseMatrix *Tikhonov::mat_mult(DenseMatrix *A, DenseMatrix *B); 
  DenseMatrix *make_dense(MatrixHandle A);
  ColumnMatrix *make_column(MatrixHandle A);
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
    haveUI_(context->subVar("haveUI")),
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
DenseMatrix *Tikhonov::mat_identity(int len) 
{
  DenseMatrix *eye = scinew DenseMatrix(len, len);
  // Does this make sure all the elements are 0?
  eye->zero();
  for(int i=0; i<len; i++)
  {
    (*eye)[i][i]=1;
  }
  return eye;
}

//! This function computes A^T * A for a DenseMatrix 
DenseMatrix *Tikhonov::mat_trans_mult_mat(DenseMatrix *A) 
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
DenseMatrix *Tikhonov::mat_mult(DenseMatrix *A, DenseMatrix *B) 
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

//! This function make sure that the matrix is a DenseMatrix.
DenseMatrix *Tikhonov::make_dense(MatrixHandle A) 
{
  DenseMatrix *Adense = dynamic_cast<DenseMatrix *>(A.get_rep());
  if (Adense) 
    return Adense;
  SparseRowMatrix *Asparse = dynamic_cast<SparseRowMatrix *>(A.get_rep());
  if (!Asparse) 
  {
    ColumnMatrix *Acol = dynamic_cast<ColumnMatrix *>(A.get_rep());
    if (!Acol) 
    {
      warning("Bad input types.");
      return Adense;
    }
    return Acol->dense();
  } 
  else 
  {
    return Asparse->dense();
  }
}

//! This function make sure that the matrix is a ColumnMatrix.
ColumnMatrix *Tikhonov::make_column(MatrixHandle A) 
{
  ColumnMatrix *Acol = dynamic_cast<ColumnMatrix *>(A.get_rep());
  if (Acol) 
    return Acol;
  SparseRowMatrix *Asparse = dynamic_cast<SparseRowMatrix *>(A.get_rep());
  if (!Asparse) 
  {
    DenseMatrix *Adense = dynamic_cast<DenseMatrix *>(A.get_rep());
    if (!Adense) 
    {
      warning("Bad input types.");
      return Acol;
    }
    return Adense->column();
  }
  else 
  {
    return Asparse->column();
  }
}

//! Find Corner
double Tikhonov::FindCorner(Array1<double> &rho, Array1<double> &eta, 
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

//! Module execution
void Tikhonov::execute()
{
  MatrixIPort *iportForMat = (MatrixIPort *)get_iport("ForwardMat");
  MatrixIPort *iportRegMat = (MatrixIPort *)get_iport("RegularizationMat");
  MatrixIPort *iportMeasDat = (MatrixIPort *)get_iport("MeasuredPots");
	
  MatrixOPort *oportInvSol = (MatrixOPort *)get_oport("InverseSoln");
  MatrixOPort *oportRegParam = (MatrixOPort *)get_oport("RegParam");
  MatrixOPort *oportRegInvMat = (MatrixOPort *)get_oport("RegInverseMat");


  if (!iportForMat) 
  {
    error("Unable to initialize iport 'ForwardMat'.");
    return;
  }
  if (!iportRegMat) 
  {
    error("Unable to initialize iport 'RegularizationMat'.");
    return;
  }
  if (!iportMeasDat) 
  {
    error("Unable to initialize iport 'MeasuredPots'.");
    return;
  }
  if (!oportInvSol) 
  {
    error("Unable to initialize oport 'InverseSoln'.");
    return;
  }
  if (!oportRegInvMat) 
  {
    error("Unable to initialize oport 'RegInverseMat'.");
    return;
  }
    	
  // DEFINE MATRIX HANDLES FOR INPUT/OUTPUT PORTS
  MatrixHandle hMatrixForMat, hMatrixRegMat, hMatrixMeasDat;
  //MatrixHandle hMatrixRegInvMat, hMatrixInvSol;
    
  if(!iportForMat->get(hMatrixForMat)) 
  { 
  
    error("Couldn't get handle to the Forward Prob Matrix.");
    return;
		
  }
		
  if(!iportMeasDat->get(hMatrixMeasDat)) 
  { 
    error("Couldn't get handle to the measured data.");
    return;
  }
 	

   
  // TYPE CHECK
  DenseMatrix *matrixForMatD = make_dense(hMatrixForMat);
  ColumnMatrix *matrixMeasDatD = make_column(hMatrixMeasDat);
 
	

  // DIMENSION CHECK!!
  int M = matrixForMatD->nrows();
  int N = matrixForMatD->ncols();
  if (M!=matrixMeasDatD->nrows()) 
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
	
  if (!iportRegMat->get(hMatrixRegMat)) 
  {
    matrixRegMatD = mat_identity(matrixForMatD->ncols());
    mat_RtrR = mat_identity(matrixForMatD->ncols());
  } 
  else 
  {
    matrixRegMatD = make_dense(hMatrixRegMat);
    if (N != matrixRegMatD->ncols()) 
    {
    	error("The dimension of Reg. Matrix is not compatible with forward matrix.");
    	return;
    }
    mat_RtrR = mat_trans_mult_mat(matrixRegMatD);
  }

  int beg = -1; 
  int end = -1;
  double lambda, lambda2;
  double temp;
  int	lambda_index;

  // calculate A^T * Y
  int flops, memrefs;
  ColumnMatrix *mat_AtrY = scinew ColumnMatrix(N);
  matrixForMatD->mult_transpose(*matrixMeasDatD, *mat_AtrY, flops, memrefs);

    
  DenseMatrix  *regForMatrix = scinew DenseMatrix(N, N);
  ColumnMatrix *solution = scinew ColumnMatrix(N);
  ColumnMatrix *Ax = scinew ColumnMatrix(M);
  ColumnMatrix *Rx = scinew ColumnMatrix(N);

    
  if ((reg_method_.get() == "single") || (reg_method_.get() == "slider"))
  {
    if (reg_method_.get() == "single")
    {
      // Use single fixed lambda value, entered in UI
      lambda = lambda_fix_.get();
      msgStream_ << "  method = " << reg_method_.get() << "\n";//DISCARD
    }
    else if (reg_method_.get() == "slider")
    {
      // Use single fixed lambda value, select via slider
      lambda = tex_var_.get(); //lambda_sld_.get();
      msgStream_ << "  method = " << reg_method_.get() << "\n";//DISCARD
    }
  }
  else if (reg_method_.get() == "lcurve")
  {
    // Use L-curve, lambda from corner of the L-curve
    msgStream_ << "method = " << reg_method_.get() << "\n";//DISCARD

    int i, j, k, l, nLambda;
    Array1<double> lambdaArray, rho, eta;
    double	lam_step;
    nLambda=lambda_num_.get();

    ColumnMatrix *kapa = scinew ColumnMatrix(nLambda);

    lambdaArray.setsize(nLambda); 
    rho.setsize(nLambda);
    eta.setsize(nLambda);   
   
    lambdaArray[0]=lambda_min_.get();
    lam_step=pow(10,log10(lambda_max_.get()/lambda_min_.get())/(nLambda-1));
	

    for(j=0; j<nLambda; j++)
    {
      if(j) 
	lambdaArray[j] = lambdaArray[j-1]*lam_step;

      lambda2 = lambdaArray[j] * lambdaArray[j];
			
			
      ///////////////////////////////////////
      ////Calculating the solution directly
      ///////////////////////////////////////
      for (i=0; i<N; i++)
      {
	for (l=0; l<N; l++)
	{
	  temp = (*mat_RtrR)[i][l];
	  (*regForMatrix)[i][l] = (*mat_AtrA)[i][l]+lambda2*temp;
	}
      }
	
      for(k=0; k<N; k++)
	(*solution)[k] = (*mat_AtrY)[k];

      //Before, solution will be equal to (A^T * y)
      //After, solution will be equal to x_reg
	
      regForMatrix->solve(*solution);
      ////////////////////////////////
      matrixForMatD->mult(*solution, *Ax, flops, memrefs, beg, end);
      matrixRegMatD->mult(*solution, *Rx, flops, memrefs, beg, end);
      rho[j]=0;
      eta[j]=0;
	
      // Calculate the norm of Ax-b and Rx

      for(k=0; k<M; k++)
      {
	(*Ax)[k] = (*Ax)[k]-(*matrixMeasDatD)[k];
	rho[j] = rho[j] + (*Ax)[k]*(*Ax)[k];
      }
      for(k=0; k<N; k++)
      {
	eta[j] = eta[j] + (*Rx)[k]*(*Rx)[k];
      }
      rho[j] = sqrt(rho[j]);
      eta[j] = sqrt(eta[j]);
    }

    lambda = FindCorner(rho, eta, lambdaArray, kapa, &lambda_index, nLambda);

    double lower_y = eta[0]/10.;
    if (eta[nLambda-1] < lower_y)  
      lower_y = eta[nLambda-1];

    if (haveUI_.get()) 
    {
      ostringstream str;
      str << id << " plot_graph \" ";
      for (i=0; i<nLambda; i++)
	str << rho[i] << " " << eta[i] << " ";
      str << "\" \" " << rho[0]/10 << " " << eta[lambda_index] << " ";
      str << rho[lambda_index] << " " << eta[lambda_index] << " ";
      str << rho[lambda_index] << " " << lower_y << " \" ";
      str << lambda << " ; update idletasks";
      gui->execute(str.str().c_str());
    }

  } // END  else if (reg_method_.get() == "lcurve")
  lambda2 = lambda*lambda;

  ColumnMatrix  *RegParameter =scinew ColumnMatrix(1);
  (*RegParameter)[0]=lambda;

  for (int i=0; i<N; i++)
  {
    for (int l=0; l<N; l++)
    {
      temp = (*mat_RtrR)[i][l];
      (*regForMatrix)[i][l] = (*mat_AtrA)[i][l]+lambda2*temp;
    }
  }
  regForMatrix->solve(*mat_AtrY);
  DenseMatrix  *InverseMatrix =scinew DenseMatrix(N, M);

  regForMatrix->invert();
//  if (regForMatrix->invert()) {
//    error("Matrix not invertible.");
//    return;
//  }

  Mult(*InverseMatrix,*regForMatrix,*(matrixForMatD->transpose()));



  //...........................................................
  // SEND RESULTS TO THE OUTPUT PORTS
  oportInvSol->send(MatrixHandle(mat_AtrY));
  oportRegParam->send(MatrixHandle(RegParameter));
  oportRegInvMat->send(MatrixHandle(InverseMatrix));

}
  
} // End namespace BioPSE

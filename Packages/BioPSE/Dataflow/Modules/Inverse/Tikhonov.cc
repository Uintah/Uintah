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
//    Last update: Feb. 2002


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

  #define nLambda 80

  class BioPSESHARE Tikhonov : public Module 
  {
    GuiDouble lambda_fix_;
    GuiDouble lambda_sld_;
    GuiInt    haveUI_;
    GuiString reg_method_;

  public:
    // CONSTRUCTOR
    Tikhonov(const string& id);

    // DESTRUCTOR
    virtual ~Tikhonov();

    virtual void execute();

    DenseMatrix *mat_identity(int len);
    DenseMatrix *mat_trans_mult_mat(DenseMatrix *A);
    DenseMatrix *make_dense(MatrixHandle A);
    ColumnMatrix *make_column(MatrixHandle A);
    double FindCorner(Array1<double>  &rho, Array1<double>  &eta, Array1<double>  &lambdaArray, ColumnMatrix *kapa, int *lambda_index);

    // virtual void tcl_command(TCLArgs&, void*);
  };
	
  // MODULE MAKER
  extern "C" BioPSESHARE Module* make_Tikhonov(const string& id) 
  {
    return scinew Tikhonov(id);
  }

  // CONSTRUCTOR
  Tikhonov::Tikhonov(const string& id)
    : Module("Tikhonov", id, Source, "Inverse", "BioPSE"),
      haveUI_("haveUI", id, this),
      lambda_fix_("lambda_fix", id, this),
      lambda_sld_("lambda_sld", id, this),
      reg_method_("reg_method", id, this)
  {
  }

  // DESTRUCTOR
  Tikhonov::~Tikhonov()
  {
  }

  // CREATE IDENTITY MATRIX
  DenseMatrix *Tikhonov::mat_identity(int len) 
  {
    DenseMatrix *eye = scinew DenseMatrix(len, len);
    // Does this make sure all the elements are 0?
    eye->zero();
    for(int i=0; i<len; i++){
      (*eye)[i][i]=1;
    }
    return eye;
  }

  // THIS FUNCTION COMPUTES A^T * A for a DenseMatrix A
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

    for (i=0; i<nCols; i++){
      // build copy of this column
      for (j=0; j<nRows; j++){
	(*Ai)[j] = (*A)[j][i];
      }
      A->mult_transpose(*Ai, *Bi, flops, memrefs, beg, end);
      for (j=0; j<nCols; j++){
	(*B)[j][i] = (*Bi)[j];
      }
    }
    return B;
  }

  // THIS FUNCTION MAKES SURE THAT THE MATRIX IS A DENSE MATRIX
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
	    cerr << "Bad input.\n";
	    return Adense;
	  }
	return Acol->dense();
      } 
    else 
      {
    	return Asparse->dense();
      }
  }

  // THIS FUNCTION MAKES SURE THAT THE MATRIX IS A COLUMN MATRIX
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
	    cerr << "Bad input.\n";
	    return Acol;
	  }
    	return Adense->column();
      }
    else 
      {
    	return Asparse->column();
      }
  }

  // FIND CORNER
  double Tikhonov::FindCorner(Array1<double> &rho, Array1<double> &eta, Array1<double> &lambdaArray, ColumnMatrix *kapa, int *lambda_index)
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

  // MODULE EXECUTION
  void Tikhonov::execute()
  {
    MatrixIPort *iportForMat = (MatrixIPort *)get_iport("ForwardMat");
    MatrixIPort *iportRegMat = (MatrixIPort *)get_iport("RegularizationMat");
    MatrixIPort *iportMeasDat = (MatrixIPort *)get_iport("MeasuredPots");

    MatrixOPort *oportInvSol = (MatrixOPort *)get_oport("InverseSoln");
    MatrixOPort *oportRegForMat = (MatrixOPort *)get_oport("RegForwardMat");
    //    MatrixOPort *oportKAPA = (MatrixOPort *)get_oport("KAPA");

    if (!iportForMat) 
      {
    	postMessage("Unable to initialize "+name+"'s iport\n");
    	return;
      }
    if (!iportRegMat) 
      {
    	postMessage("Unable to initialize "+name+"'s iport\n");
	return;
      }
    if (!iportMeasDat) 
      {
    	postMessage("Unable to initialize "+name+"'s iport\n");
	return;
      }
    if (!oportInvSol) 
      {
    	postMessage("Unable to initialize "+name+"'s oport\n");
	return;
      }
    if (!oportRegForMat) 
      {
    	postMessage("Unable to initialize "+name+"'s oport\n");
	return;
      }
    /*
    if (!oportKAPA) 
      {
    	postMessage("Unable to initialize "+name+"'s oport\n");
	return;
      }
    */

    // DEFINE MATRIX HANDLES FOR INPUT/OUTPUT PORTS
    MatrixHandle hMatrixForMat, hMatrixRegMat, hMatrixMeasDat;
    //MatrixHandle hMatrixRegForMat, hMatrixInvSol;
    
    if(!iportForMat->get(hMatrixForMat)) 
      { 
    	msgStream_ << "Couldn't get handle to the Forward Prob Matrix. Returning." << endl;
	return;
      }
    
    if(!iportMeasDat->get(hMatrixMeasDat)) 
      { 
    	msgStream_ << "Couldn't get handle to the measured data. Returning." << endl;
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
    	cerr << "Matrix dimensions must agree.  " << M << " " << matrixMeasDatD->nrows() << "\n";
	return;
      }
    
    //...........................................................
    // OPERATE ON DATA: 
    // SOLVE (A^T * A + LAMBDA * LAMBDA * R^T * R) * X = A^T * Y
    // FOR "X"
    
    // calculate A^T * A
    DenseMatrix *mat_AtrA = mat_trans_mult_mat(matrixForMatD);
    
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
	    cerr << "  method = " << reg_method_.get() << "\n";//DISCARD
	  }
	else if (reg_method_.get() == "slider")
	  {
	    // Use single fixed lambda value, select via slider
	    lambda = lambda_sld_.get();
	    cerr << "  method = " << reg_method_.get() << "\n";//DISCARD
	  }
      }
    else if (reg_method_.get() == "lcurve")
      {
	// Use L-curve, lambda from corner of the L-curve
	cerr << "method = " << reg_method_.get() << "\n";//DISCARD

	int i, j, k, l;
	Array1<double> lambdaArray, rho, eta;
	ColumnMatrix *kapa = scinew ColumnMatrix(nLambda);

	lambdaArray.setsize(nLambda); 
	rho.setsize(nLambda);
	eta.setsize(nLambda);   

	lambdaArray[0]=1e-6;

	for(j=0; j<nLambda; j++)
	  {
	    if(j) 
	      lambdaArray[j] = lambdaArray[j-1]*1.23;

	    lambda2 = lambdaArray[j] * lambdaArray[j];
		
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

	lambda = FindCorner(rho, eta, lambdaArray, kapa, &lambda_index);

	int lower_y = eta[0]/10;
	if (eta[nLambda-1] < lower_y)  lower_y = eta[nLambda-1];

	if (haveUI_.get()) {
	  ostringstream str;
	  str << id << " plot_graph \" ";
	  for (i=0; i<nLambda; i++)
	      str << rho[i] << " " << eta[i] << " ";
	  str << "\" \" " << rho[0]/10 << " " << eta[lambda_index] << " ";
	  str << rho[lambda_index] << " " << eta[lambda_index] << " ";
	  str << rho[lambda_index] << " " << lower_y << " \" ";
	  str << lambda << " ; update idletasks";
	  TCL::execute(str.str().c_str());
	}

      } // END  else if (reg_method_.get() == "lcurve")

    cout << "lambda = " << lambda << endl;
    lambda2 = lambda*lambda;

    for (int i=0; i<N; i++)
      {
	for (int l=0; l<N; l++)
	  {
	    temp = (*mat_RtrR)[i][l];
	    (*regForMatrix)[i][l] = (*mat_AtrA)[i][l]+lambda2*temp;
	  }
      }

    regForMatrix->solve(*mat_AtrY);
    
    //...........................................................
    // SEND RESULTS TO THE OUTPUT PORTS
    oportInvSol->send(MatrixHandle(mat_AtrY));
    oportRegForMat->send(MatrixHandle(regForMatrix));
    //    oportKAPA->send(MatrixHandle(kapa));
  }
  
  // void Tikhonov::tcl_command(TCLArgs& args, void* userdata)
  // {
  //   Module::tcl_command(args, userdata);
  // }
  
} // End namespace BioPSE

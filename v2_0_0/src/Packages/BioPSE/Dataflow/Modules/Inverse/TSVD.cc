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
//    File       : TSVD.cc
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

#define	SVD_method	1
#define	GSVD_method 	2


namespace BioPSE 
{

  using namespace SCIRun;

  class BioPSESHARE TSVD : public Module 
  {
    	GuiInt	 	lambda_fix_;
    	GuiDouble 	lambda_sld_;
    	GuiInt    	haveUI_;
    	GuiString 	reg_method_;
    	GuiInt		lambda_max_;
	
    		

  	public:
   	 	// CONSTRUCTOR
   	 	TSVD(GuiContext *context);

	    	// DESTRUCTOR
 	   	virtual ~TSVD();

  	  	virtual void execute();

  
		double Inner_Product(DenseMatrix& A, int col_num, ColumnMatrix& w);
		void	find_solution(ColumnMatrix& X_reg, DenseMatrix& InvMat, DenseMatrix& U, ColumnMatrix& Uy, DenseMatrix& S, DenseMatrix& V, DenseMatrix& X, int lam);
		void	prep_lcurve_data(Array1<double>  &rho, Array1<double>  &eta, ColumnMatrix& Uy, DenseMatrix& U, 
						 DenseMatrix& S, DenseMatrix& V, DenseMatrix& X, ColumnMatrix& y);
		DenseMatrix *make_dense(MatrixHandle A);
    		ColumnMatrix *make_column(MatrixHandle A);
		void TSVD::Conv(Array1<double> &sp, ColumnMatrix& coef, Array1<double> &basis, int nLambda);  
  		void FindCorner(Array1<double>  &rho, Array1<double>  &eta, ColumnMatrix *kapa, int *lambda_index, int nLambda);
		
  };
	
  // MODULE MAKER
  DECLARE_MAKER(TSVD)


  // CONSTRUCTOR
  TSVD::TSVD(GuiContext *context)
	: Module("TSVD", context, Source, "Inverse", "BioPSE"),
      	lambda_fix_(context->subVar("lambda_fix")),
      	lambda_sld_(context->subVar("lambda_sld")),
      	haveUI_(context->subVar("haveUI")),
      	reg_method_(context->subVar("reg_method")),
	lambda_max_(context->subVar("lambda_max"))
	
	
  {
  }

  // DESTRUCTOR
  TSVD::~TSVD()
  {
  }
  //////////////////////////////////////////////////////////////////////////////////////////
  // THIS FUNCTION returns the inner product of one column of matrix A and w , B=A(:,i)'*w
  //////////////////////////////////////////////////////////////////////////////////////////
  double TSVD::Inner_Product(DenseMatrix& A, int col_num, ColumnMatrix& w)
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
  void TSVD::find_solution(ColumnMatrix& X_reg, DenseMatrix& InvMat, DenseMatrix& U, ColumnMatrix& Uy, 
					DenseMatrix& S, DenseMatrix& V, DenseMatrix& X, int lam)
  {
	int i,j;
	int rank=S.nrows();	
	double temp;
	DenseMatrix  *Mat_temp;	
	if(S.ncols() == 1)
	{

		temp =  Uy[0]/S[0][0];
		for(j=0;j<V.nrows();j++)
		{
			X_reg[j]=temp*V[j][0];
		}
	
		for(i=1; i < lam; i++)
		{
			temp = Uy[i]/S[i][0];
			for(j=0;j<V.nrows();j++)
			{
                                X_reg[j] = X_reg[j] + temp * V[j][i];
			}
		}

		//Finding Regularized Inverse Matrix
		Mat_temp = scinew  DenseMatrix(V.nrows(),V.ncols());
		for(i=0; i < lam; i++)
		{
			for(j=0;j<V.nrows();j++)
			{
                                (*Mat_temp)[j][i] = V[j][i]/S[i][0];
			}
		}
		Mult(InvMat,V,*(U.transpose()));


	}
	else
	{
		temp =  Uy[rank-1]/S[rank-1][0];
		for(j=0;j<X.nrows();j++)
		{
			X_reg[j]=temp*X[j][rank-1];
		}
		for(i=rank-2; i > rank-lam-1; i--)
		{
			temp = Uy[i]/S[i][0];
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
 
		Mat_temp = scinew  DenseMatrix(X.nrows(),X.ncols());
		(*Mat_temp)=X; 

		for(i=rank-1; i > rank-lam-1; i--)
		{
			for(j=0;j<X.nrows();j++)
			{
				(*Mat_temp)[j][i]= X[j][i]/S[i][0];
			}
		}
		Mult(InvMat,X,*(U.transpose()));
	}
  }
  /////////////////////////////////////////////////////////////	
  // THIS FUNCTION Calculate ro and eta for lcurve
  /////////////////////////////////////////////////////////////
  void	TSVD::prep_lcurve_data(Array1<double> &rho, Array1<double> &eta, ColumnMatrix& Uy, DenseMatrix& U, 
				DenseMatrix& S, DenseMatrix& V, DenseMatrix& X, ColumnMatrix& y)
  {
	int i,j;
	ColumnMatrix *AX_reg = scinew ColumnMatrix(U.nrows());
	ColumnMatrix *Residual = scinew ColumnMatrix(U.nrows());
    	ColumnMatrix *RX_reg = scinew ColumnMatrix(V.nrows());
	ColumnMatrix *X_reg; 
	int rank=S.nrows();
	double temp, temp1;
	if(S.ncols() == 1)
	{

		for(i=0;i<S.nrows();i++)
			if(S[i][0]<1e-10)
				break;
		if(i<S.nrows())
			rank=i;

		X_reg = scinew ColumnMatrix(V.nrows());		
		temp =  Uy[0]/S[0][0];
		for(j=0;j<V.nrows();j++)
		{
			(*X_reg)[j]=temp*V[j][0];
		}
		for(j=0;j<U.nrows();j++)
		{
			(*AX_reg)[j]=Uy[0]*U[j][0];
		}
		Sub(*Residual,*AX_reg,y);
		rho[0]=Residual->vector_norm();
		eta[0]=X_reg->vector_norm();
		
		for(i=1; i < rank; i++)
		{
			temp = Uy[i]/S[i][0];
			for(j=0;j<V.nrows();j++)
			{
                                (*X_reg)[j] = (*X_reg)[j] + temp * V[j][i];
			}
			for(j=0;j<U.nrows();j++)
			{
				(*AX_reg)[j]+= Uy[i]*U[j][i];
			}
			Sub(*Residual,*AX_reg,y);
			rho[i]=Residual->vector_norm();
			eta[i]=X_reg->vector_norm();
		}
	}
	else
	{
		X_reg = scinew ColumnMatrix(X.nrows());
		AX_reg->zero();
		RX_reg->zero();
		for(i=S.nrows()-1;i>=0;i--)
			if(S[i][0]<1e-10)
				break;
		if(i>=0)
			rank=S.nrows()-i;
		
		for(i=S.nrows(); i < X.ncols(); i++)
		{
			for(j=0;j<U.nrows();j++)
			{
				(*AX_reg)[j]+= Uy[i] * U[j][i];
			}
		}
		for(i=S.nrows()-1; i >= (S.nrows()-rank); i--)
		{
			temp = Uy[i];
			for(j=0;j<U.nrows();j++)
			{
				(*AX_reg)[j]+= temp * U[j][i];
			}
			temp1 =temp *S[i][1]/S[i][0];
			for(j=0;j<V.nrows();j++)
			{
				(*RX_reg)[j]+= temp1 * V[j][i];
			}
			Sub(*Residual,*AX_reg,y);
			rho[S.nrows()-i-1]=Residual->vector_norm();
			eta[S.nrows()-i-1]=RX_reg->vector_norm();
		}
	}
  }

  /////////////////////////////////////////////////////////////	
  // THIS FUNCTION MAKES SURE THAT THE MATRIX IS A DENSE MATRIX
  /////////////////////////////////////////////////////////////
  DenseMatrix *TSVD::make_dense(MatrixHandle A) 
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
  ////////////////////////////////////////////////////////////////
  // THIS FUNCTION MAKES SURE THAT THE MATRIX IS A COLUMN MATRIX
  ////////////////////////////////////////////////////////////////
  ColumnMatrix *TSVD::make_column(MatrixHandle A) 
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
  ////////////////////////////////////////////
  //          getting curve from coeficient and basis
  // This is like convolving basis function with the coeficients 
  // after inserting 3 zeros in between any two consecutive coefficients 
  ////////////////////////////////////////////
  void TSVD::Conv(Array1<double> &sp, ColumnMatrix& coef, Array1<double> &basis, int nLambda)
  {
	int i, j;
	for(i=0;i<4*nLambda;i++)
	{
		sp[i]=0;
		for(j=0;j<15;j++)
		{
			if(j-7+i>= 0 && j-7+i < nLambda*4 )
			{
				if((i+j-7)%4 == 0)
				{
					sp[i]+=basis[j]*coef[(i+j-7)/4];
					
				}	
			}
		}
	}
  }
  ////////////////////////////////////////////
  // find corner of the L-curve
  ////////////////////////////////////////////
  void TSVD::FindCorner(Array1<double> &rho, Array1<double> &eta, ColumnMatrix *kapa, int *lambda_index, int nLambda)
  {
    	Array1<double> deta, ddeta, drho, ddrho, lrho, leta;
    	double  maxKapa=-1e10;
    	int	i;
	Array1<double> basis;
	basis.setsize(15);
	leta.setsize(4*nLambda); 
    	deta.setsize(4*nLambda); 
    	ddeta.setsize(4*nLambda);
    	lrho.setsize(4*nLambda);   
    	drho.setsize(4*nLambda);   
    	ddrho.setsize(4*nLambda);   
	
	
	// Finding the coefficient of Bsplines that reconstruct the curve
	// The number of knots are the same as the number of curve points

	DenseMatrix  *Bspline_Bases = scinew DenseMatrix(nLambda,nLambda);
	ColumnMatrix  *rho_coef = scinew ColumnMatrix(nLambda);
	ColumnMatrix  *eta_coef = scinew ColumnMatrix(nLambda);
		
	Bspline_Bases->zero();
	(*Bspline_Bases)[0][0]=2/3;
	(*Bspline_Bases)[1][0]=1/6;
	for(int col=1;col<nLambda-1;col++)
	{
		(*Bspline_Bases)[col-1][col]=1/6;
		(*Bspline_Bases)[col][col]=2/3;
		(*Bspline_Bases)[col+1][col]=1/6;
	}
	(*Bspline_Bases)[nLambda-2][nLambda-1]=1/6;
	(*Bspline_Bases)[nLambda-1][nLambda-1]=2/3;
	for(i=0;i<nLambda;i++)
	{
		(*rho_coef)[i]=log10(rho[i]);	
		(*eta_coef)[i]=log10(eta[i]);	
	}
	Bspline_Bases->solve(*rho_coef);
	Bspline_Bases->solve(*eta_coef);
		
	
	//Interpolation to have 4 times higher resolution

	//bspline basis 
	
	basis[0]=0.0026;
	basis[1]=0.0208;
	basis[2]=0.0703;
	basis[3]=0.1667;
	basis[4]=0.3151;
	basis[5]=0.4792;
	basis[6]=0.6120;
	basis[7]=0.6667;
	for(i=14;i>7;i--)
		basis[i]=basis[14-i];


	Conv(lrho, *rho_coef, basis,nLambda);
	Conv(leta, *eta_coef, basis,nLambda);

	// First derivative of Bspline basis
	
	basis[0]=0.0312;
	basis[1]=0.1250;
	basis[2]=0.2812;
	basis[3]=0.5000;
	basis[4]=0.6562;
	basis[5]=0.6250;
	basis[6]=0.4062;
	basis[7]=0;
	for(i=14;i>7;i--)
		basis[i]=-basis[14-i];
	
	Conv(drho, *rho_coef, basis,nLambda);
	Conv(deta, *eta_coef, basis,nLambda);

	// Second derivative of Bspline basis
	basis[0]=0.2500;
	basis[1]=0.5000;
	basis[2]=0.7500;
	basis[3]=1.0000;
	basis[4]=0.2500;
	basis[5]=-0.5000;
	basis[6]=-1.2500;
	basis[7]=-2.0000;
	for(i=14;i>7;i--)
		basis[i]=basis[14-i];
	
	Conv(ddrho, *rho_coef, basis,nLambda);
	Conv(ddeta, *eta_coef, basis,nLambda);

  
  	*lambda_index=0;
 
	// finding the maximum curvature
		

	for(i=0; i<4*nLambda; i++)
      	{
		(*kapa)[i] = 2*(drho[i]*ddeta[i] - ddrho[i]*deta[i])/sqrt(pow((deta[i]*deta[i]+drho[i]*drho[i]),3));  
		if((*kapa)[i]>maxKapa)
	  	{	
	    		maxKapa = (*kapa)[i];
	    		*lambda_index = i;
	  	}

      	}
	(*lambda_index)/=4;
  }
  /////////////////////////////////////////
  // MODULE EXECUTION
  /////////////////////////////////////////
  void TSVD::execute()
  {
     	MatrixIPort *iportMeasDat = (MatrixIPort *)get_iport("MeasuredPots");
	
	MatrixIPort *iportU = (MatrixIPort *)get_iport("U");
	MatrixIPort *iportS = (MatrixIPort *)get_iport("S");
	MatrixIPort *iportV = (MatrixIPort *)get_iport("V");
	//MatrixIPort *iportX = (MatrixIPort *)get_iport("X");

    	MatrixOPort *oportInvSol = (MatrixOPort *)get_oport("InverseSoln");
  	MatrixOPort *oportRegParam = (MatrixOPort *)get_oport("RegParam");
    	MatrixOPort *oportRegInvMat = (MatrixOPort *)get_oport("RegInverseMat");


    	if (!iportMeasDat) 
      	{
    		error("Unable to initialize iport 'MeasuredPots'.");
		return;
      	}
	if (!iportU) 
      	{
    		error("Unable to initialize iport 'U'.");
		return;
      	}

	if (!iportS) 
      	{
    		error("Unable to initialize iport 'S'.");
		return;
      	}

	if (!iportV) 
      	{
    		error("Unable to initialize iport 'V'.");
		return;
      	}
/*	if (!iportX) 
     	{
  		error("Unable to initialize iport 'X'.");
		return;
     	}*/
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
    	
    	//DEFINE MATRIX HANDLES FOR INPUT/OUTPUT PORTS
    	
	MatrixHandle hMatrixMeasDat, hMatrixU, hMatrixS, hMatrixV, hMatrixX;
    	    
    	
     	if( (!iportU->get(hMatrixU)) || (!iportS->get(hMatrixS)) || (!iportV->get(hMatrixV)) )
	{
		error("Couldn't get handle to the U, S or V Matrices.");
		return;	
	}
			
    	if(!iportMeasDat->get(hMatrixMeasDat)) 
      	{ 
    		error("Couldn't get handle to the measured data.");
		return;
      	}
   	 	

   
    	// TYPE CHECK
    	
    	ColumnMatrix *matrixMeasDatD = make_column(hMatrixMeasDat);
   	DenseMatrix *matrixU = make_dense(hMatrixU);
  	DenseMatrix *matrixS = make_dense(hMatrixS);	
   	DenseMatrix *matrixV = make_dense(hMatrixV);
	DenseMatrix *matrixX;
	

	
	int	Method;
	int M, N, i;
	int	lambda;
	int rank = matrixS->nrows();

	if(matrixS->ncols()==1)
	{	
		Method=SVD_method;
		for(i=0;i<matrixS->nrows();i++)
			if((*matrixS)[i][0]<1e-10)
				break;
		if(i<matrixS->nrows())
			rank=i;
	}
	else
	if(matrixS->ncols()==2)
	{
		// TGSVD is not implemented yet.

		error("S matrix dimensions incorrect.");
		return;	
		Method=GSVD_method;
		
		/*if(!iportX->get(hMatrixX)) 
	      	{ 
 	   		error("Couldn't get handle X input.");
			return;
   	   	}*/
		matrixX = make_dense(hMatrixX);
		
		for(i=matrixS->nrows()-1;i>=0;i--)
			if((*matrixS)[i][0]<1e-10)
				break;
		if(i>=0)
			rank=matrixS->nrows()-i;

	}
	else
	{
		error("S matrix dimensions incorrect.");
		return;	
	}
	lambda_max_.set(rank);
	
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
		N=matrixX->nrows();

	ColumnMatrix *Uy=scinew ColumnMatrix(matrixU->ncols());
    	ColumnMatrix *solution = scinew ColumnMatrix(N);
    	DenseMatrix  *InverseMat = scinew DenseMatrix(N, M);
	
	for(i=0;i<matrixU->ncols();i++)
		(*Uy)[i]=Inner_Product(*matrixU, i, *matrixMeasDatD);
	
   	   
    
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
			lambda = floor(lambda_sld_.get()/10*rank);
	 		msgStream_ << "  method = " << reg_method_.get() << "\n";//DISCARD
		}
		
	}
	else if (reg_method_.get() == "lcurve")
 	{
		// Use L-curve, lambda from corner of the L-curve
		msgStream_ << "method = " << reg_method_.get() << "\n";//DISCARD

		int nLambda;
		Array1<double> rho, eta, lambdaArray;
		double lower_y;
		nLambda=rank;

		ColumnMatrix *kapa = scinew ColumnMatrix(4*nLambda); 
		rho.setsize(rank);
		eta.setsize(rank);
		lambdaArray.setsize(rank);   
		prep_lcurve_data(rho, eta, *Uy, *matrixU, *matrixS, *matrixV, *matrixX, *matrixMeasDatD);
		FindCorner(rho, eta, kapa, &lambda, rank);
		
		lower_y = eta[0];
		if (eta[nLambda-1] < lower_y)  
			lower_y = eta[nLambda-1];

		if (haveUI_.get()) 
		{
	  		ostringstream str;
	  		str << id << " plot_graph \" ";
	  		for (i=0; i<nLambda; i++)
	      			str << rho[i] << " " << eta[i] << " ";
  			str << "\" \" " << rho[nLambda-1]/10 << " " << eta[lambda] << " ";
  			str << rho[lambda] << " " << eta[lambda] << " ";
  			str << rho[lambda] << " " << lower_y/10 << " \" ";
  			str << lambda << " ; update idletasks";
  			gui->execute(str.str().c_str());
		}
	

		
	} 
	
	ColumnMatrix  *RegParameter =scinew ColumnMatrix(1);
  	(*RegParameter)[0]=lambda;

	find_solution(*solution,*InverseMat, *matrixU, *Uy, *matrixS, *matrixV, *matrixX, lambda);

	
	//...........................................................
    	// SEND RESULTS TO THE OUTPUT PORTS
    	oportInvSol->send(MatrixHandle(solution));
	oportRegParam->send(MatrixHandle(RegParameter));
    	oportRegInvMat->send(MatrixHandle(InverseMat));
	    	


  }
  
} // End namespace BioPSE


/*
 *  Matrix.h: Matrix definitions
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/Matrix.h>
#include <Datatypes/ColumnMatrix.h>
#include <Classlib/Assert.h>
#include <Classlib/NotFinished.h>

PersistentTypeID Matrix::type_id("Matrix", "Datatype", 0);

Matrix::Matrix(Sym sym)
: sym(sym)
{
}

Matrix::~Matrix()
{
}

void Matrix::io(Piostream&)
{
    NOT_FINISHED("Matrix::io");
}

Matrix* Matrix::clone()
{
    return 0;
}

int Matrix::isolve(ColumnMatrix& lhs, ColumnMatrix& rhs, double max_error,
		   MatrixUpdater* updater)
{
    int size=nrows();
    ASSERT(ncols() == size);
    ASSERT(lhs.nrows() == size);
    ASSERT(rhs.nrows() == size);

    ColumnMatrix diag(size);
    {
	// We should try to do a better job at preconditioning...
	for(int i=0;i<size;i++){
	    diag[i]=1./get(i,i);
	}
    }
    ColumnMatrix R(size);
    mult(lhs, R);
    for(int i=0;i<size;i++){
	R[i]=rhs[i]-R[i];
    }
    double bnorm=rhs.vector_norm();
    ColumnMatrix Z(size);
    mult(R, Z);
    int niter=0;
    int toomany=2*size;
    ColumnMatrix P(size);
    double bkden=0;
    double err=2*max_error;
    double orig_err=err;
    if(updater)
	updater->update(niter, orig_err, err, max_error);
    while(niter < toomany && err > max_error){
	niter++;
	// Ugly preconditioning...
	for(int id=0;id<size;id++){
	    Z[id]=R[id]*diag[id];
	}
	
	// Calculate coefficient bk and direction vectors p and pp
	double bknum=0;
	for(int i=0;i<size;i++)
	    bknum+=Z[i]*R[i];
	if(niter==1){
	    P=Z;
	} else {
	    double bk=bknum/bkden;
	    for(int i=0;i<size;i++){
		P[i]=bk*P[i]+Z[i];
	    }
	}
	bkden=bknum;
	// Calculate coefficient ak, new iterate x and new residuals r and rr
	mult(P, Z);
	double akden=0.0;
	for(i=0;i<size;i++)
	    akden+=Z[i]*P[i];
	double ak=bknum/akden;
	for(i=0;i<size;i++){
	    lhs[i]+=ak*P[i];
	    R[i]-=ak*Z[i];
	}
	err=R.vector_norm()/bnorm;
	if(updater && niter%10 == 0)
	    updater->update(niter, orig_err, err, max_error);
    }
    return niter;
}


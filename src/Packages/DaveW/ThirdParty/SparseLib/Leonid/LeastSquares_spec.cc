#include "LeastSquares.h"

#define Max(a,b) (a>b?a:b)
#define Min(a,b) (a<b?a:b)

template<> void LeastSquares<double>::solve(){  
    
//LAPACK staff: 
    int INFO,LDA,LDB,LWORK,M,N,NRHS,RANK;
    double RCOND;
    double *S,*WORK;


    M = nrows;
    N = ncols;    
    NRHS = nrhs;
    LDA = M;
    LDB = Max(M,N);
    S = new double[Min(M,N)];
    RCOND =rcond;
    LWORK = 3*Min(M,N) + Max(2*Min(M,N),Max(Max(M,N),NRHS));

    solver_ = "dgelss";
    time = clock();
#if 0
    MatrixDense<double> *MD = dynamic_cast<MatrixDense<double> *>(*A);
    if (!MD) {
	cerr << "Error A is wrong type.\n";
	return;
    }
#endif
    dgelss_(&M,&N,&NRHS,((MatrixDense<double>*) A)->get_p(),&LDA,B->get_p(),&LDB,S,&RCOND,&RANK,WORK,&LWORK,&INFO);  
    time = clock() - time;
    
    if (INFO == 0)
      message_ = "Done!";
    if (INFO < 0)
      message_ = "Wrong Arguments!";
    if (INFO > 0)
      message_ = "SVD failed to converge";
    
    info_ = INFO;
    X = B;   
    // we are copying pointers here!
    
   SV = new ZVector<double> (Min(M,N),S); 

}
  

//----------------------------------------------------------------------------
#if 0
void LeastSquares<Complex>::solve(){
    
   
 
}
#endif

#include "GSVD.h"

#define Max(a,b) (a>b?a:b)

template<> void GSVD<double>::solve(){

 data_type = "double";
  
//LAPACK staff:
  char JOBQ,JOBU,JOBV;
  int INFO,K,L,LDA,LDB,LDQ,LDU,LDV,M,N,P;
  int *IWORK;
 double *ALPHA,*BETA,*Q,*U,*V,*WORK;
  
  M = nrows_A;
  N = ncols_A;
  P = nrows_B;
  LDA = M;
  LDB = P;
  ALPHA = new double[N];
  BETA = new double[N];
  IWORK = new int[N];
  WORK = new double[Max(Max(3*N,M),P)+N];
 
  
  if (vec_u == 0) { 
    JOBU ='N';
    LDU = 1;
    U = new double[1];    
  }

  else 
    {
    JOBU ='U';
    LDU = M;
    U = new double[LDU*M];
  }

 
if (vec_v == 0) { 
    JOBV ='N';
    LDV = 1;
    V = new double[1];    
  }

  else 
    {
    JOBV ='V';
    LDV = P ;
    V = new double[LDV*P];
  }


if (vec_q == 0) { 
    JOBQ ='N';
    LDQ = 1;
    Q = new double[1];    
  }

  else 
    {
    JOBQ ='Q';
    LDQ = N;
    Q = new double[LDU*N];
  }
  
 solver_ = "dggsvd_";
 time = clock();
 dggsvd_(&JOBU,&JOBV,&JOBQ,&M,&N,&P,&K,&L, A->get_p(), &LDA, B->get_p(), &LDB, ALPHA, BETA, U, &LDU, V, &LDV, Q, &LDQ, WORK, IWORK, &INFO);
 time = clock() - time;
  
  if (INFO == 0)
    messege_ = "Done!";
  if (INFO < 0)
    messege_= "Wrong Arguments!";
  if (INFO > 0)
    messege_ =  "Jacoby Procedure failed to converge!";

  info_ = INFO;

  
 UU = new MatrixDense<double>(M,M,U);
 VV = new MatrixDense<double>(P,P,V);
 QQ = new MatrixDense<double>(N,N,Q);
 CC = new ZVector<double>(L,&ALPHA[K]);
 SS = new ZVector<double>(L,&BETA[K]);

 
 }
//----------------------------------------------------------------------------
#if 0
void GSVD<Complex>::solve(){

 data_type = "Complex";
  
//LAPACK staff:
  char JOBQ,JOBU,JOBV;
  int INFO,K,L,LDA,LDB,LDQ,LDU,LDV,M,N,P;
  int *IWORK;
  double* ALPHA, *BETA, *RWORK;
  Complex *Q,*U,*V,*WORK;
  
  M = nrows_A;
  N = ncols_A;
  P = nrows_B;
  LDA = M;
  LDB = P;
  ALPHA = new double[N];
  BETA = new double[N];
  RWORK = new double[2*N];
  IWORK = new int[N];
  WORK = new Complex[Max(Max(3*N,M),P)+N];
 
  
  if (vec_u == 0) { 
    JOBU ='N';
    LDU = 1;
    U = new Complex[1];    
  }

  else 
    {
    JOBU ='U';
    LDU = M;
    U = new Complex[LDU*M];
  }

 
if (vec_v == 0) { 
    JOBV ='N';
    LDV = 1;
    V = new Complex[1];    
  }

  else 
    {
    JOBV ='V';
    LDV = P ;
    V = new Complex[LDV*P];
  }


if (vec_q == 0) { 
    JOBQ ='N';
    LDQ = 1;
    Q = new Complex[1];    
  }

  else 
    {
    JOBQ ='Q';
    LDQ = N;
    Q = new Complex[LDU*N];
  }
  
 solver_ = "zggsvd_";
 time = clock();
 zggsvd_(&JOBU,&JOBV,&JOBQ,&M,&N,&P,&K,&L, A->get_p(), &LDA, B->get_p(), &LDB, ALPHA, BETA, U, &LDU, V, &LDV, Q, &LDQ, WORK, RWORK,IWORK, &INFO);
 time = clock() - time;
  
  if (INFO == 0)
    messege_ = "Done!";
  if (INFO < 0)
    messege_= "Wrong Arguments!";
  if (INFO > 0)
    messege_ =  "Jacoby Procedure failed to converge!";

  info_ = INFO;

  
 UU = new MatrixDense<Complex>(M,M,U);
 VV = new MatrixDense<Complex>(P,P,V);
 QQ = new MatrixDense<Complex>(N,N,Q);
 CC = new ZVector<double>(L,&ALPHA[K]);
 SS = new ZVector<double>(L,&BETA[K]);

}
#endif

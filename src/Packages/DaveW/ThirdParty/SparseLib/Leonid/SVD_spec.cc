#include "SVD.h"

template<> void SVD<double>::solve(){

 data_type = "double";
  
//LAPACK staff:
  char JOBU,JOBVT;
  int INFO,LDA,LDU,LDVT,LWORK,M,N;
  double *S,*U,*VT,*WORK;
  
  M = nrows;
  N = ncols; 
  LDA = M;
  S = new double[ZMin(M,N)]; 

  
  if (lvec == 0) { 
    JOBU ='N';
    LDU = 1;
    U = new double[1];    
  }

  else
 
    {
    JOBU ='A';
    LDU = M;
    U = new double[LDU*M];
  }

 
 if (rvec == 0) { 
   JOBVT ='N';
   LDVT = 1;
   VT = new double[1]; 
 }
 else

 {
    JOBVT ='A';
    LDVT = N;
    VT = new double[LDVT*N];
  }

 
  LWORK = ZMax(3*ZMin(M,N)+ZMax(M,N),5*ZMin(M,N)-4);
  WORK = new double[LWORK];
  
 solver_ = "dgesvd_";
 time = clock();
 dgesvd_(&JOBU, &JOBVT, &M, &N, A->get_p(), &LDA, S, U, &LDU, VT, &LDVT, WORK,&LWORK, & INFO );
 time = clock() - time;
  
  if (INFO == 0)
    message_ = "Done!";
  if (INFO < 0)
    message_= "Wrong Arguments!";
  if (INFO > 0)
    message_ =  "Algorithm did not converge!";

  info_ = INFO;

  
  UU = new MatrixDense<double>(M,M,U);
  SS = new ZVector<double>(ZMin(M,N),S);
  VVT = new MatrixDense<double>(N,N,VT);
 
 }
//----------------------------------------------------------------------------
#if 0
void SVD<Complex>::solve(){

 data_type = "Complex";
  
//LAPACK staff:
  char JOBU,JOBVT;
  int INFO,LDA,LDU,LDVT,LWORK,M,N;
  double *RWORK, *S;
  Complex *U,*VT,*WORK;
  
  M = nrows;
  N = ncols; 
  LDA = M;
  S = new double[ZMin(M,N)]; 

  
  if (lvec == 0) { 
    JOBU ='N';
    LDU = 1;
    U = new Complex[1];    
  }

  else
 
    {
    JOBU ='A';
    LDU = M;
    U = new Complex[LDU*M];
  }

 
 if (rvec == 0) { 
   JOBVT ='N';
   LDVT = 1;
   VT = new Complex[1]; 
 }
 else

 {
    JOBVT ='A';
    LDVT = N;
    VT = new Complex[LDVT*N];
  }

 
  LWORK = 2*ZMin(M,N)+ZMax(M,N);
  WORK = new Complex[LWORK];

  RWORK = new double[ZMax(3*ZMin(M,N),5*ZMin(M,N)-4)];

 
 solver_ = "zgesvd_";
 time = clock();
 zgesvd_(&JOBU, &JOBVT, &M, &N,A->get_p(), &LDA, S, U, &LDU, VT, &LDVT, WORK,&LWORK, RWORK,& INFO );
 time = clock() - time;
  
  if (INFO == 0)
    message_ = "Done!";
  if (INFO < 0)
    message_= "Wrong Arguments!";
  if (INFO > 0)
    message_ =  "Algorithm did not converge!";

  info_ = INFO;

  
  UU = new MatrixDense<Complex>(M,M,U);
  SS = new ZVector<double>(ZMin(M,N),S);
  VVT = new MatrixDense<Complex>(N,N,VT);
 
 }  
#endif


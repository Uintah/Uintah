/****************************************************************
 *  Class GSVD.h   (Interface to LAPACK)                        *
 *                                                              *
 *  Written by:                                                 *
 *   Leonid Zhukov                                              *
 *   Department of Computer Science                             *
 *   University of Utah                                         *
 *   October 1997                                               *
 *                                                              *
 *  Copyright (C) 1997 SCI Group                                *
 *                                                              *
 *                                                              *
 ****************************************************************/

#ifndef GSVD_H
#define GSVD_H 1

#include "LinearMethods.h"
#include "MatrixDense.h"
#include "Vector.h"
#include "Complex.h"
#include "utils.h"
#include "time.h"

//GSVD<Data_Type> Class Definitions
//----------------------------------------------------------------------------
template <class Data_Type>
class GSVD: public LinearMethods<Data_Type>{

private:
  MatrixDense<Data_Type>* A;
  MatrixDense<Data_Type>* B; 
  MatrixDense<Data_Type>* UU;
  MatrixDense<Data_Type>* VV;
  MatrixDense<Data_Type>* QQ;
  ZVector<double>* SS;
  ZVector<double>* CC;
  int nrows_A;
  int ncols_A;
  int nrows_B;
  int ncols_B;
  clock_t time;
  char* messege_;
  char* data_type;
  char* solver_;
  int info_;
  int vec_u;
  int vec_v;
  int vec_q;
 
  
public:
  GSVD(MatrixDense<Data_Type>& AA,MatrixDense<Data_Type>& BB,char* memory = "no");
  void solve(); 
  void info();
  void compute_QUV(){ vec_u=1;vec_v = 1;vec_q=1;};

  MatrixDense<Data_Type>* get_U(){return UU->get_p();};
  MatrixDense<Data_Type>* get_V(){return VV->get_p();};
  MatrixDense<Data_Type>* get_Q(){return QQ->get_p();};
  ZVector<double>* get_S(){return SS->get_p();};
  ZVector<double>* get_C(){return CC->get_p();};

  MatrixDense<Data_Type> solution_U(){return *UU;};
  MatrixDense<Data_Type> solution_V(){return *VV;};
  MatrixDense<Data_Type> solution_Q(){return *QQ;};
  ZVector<double> solution_S(){return *SS;};
  ZVector<double> solution_C(){return *CC;};
};


//LAPACK declarations
//---------------------------------------------------------------------------  
extern "C"{
int dggsvd_(char *jobu, char *jobv, char *jobq, int *m, 
        int *n, int *p, int *k, int *l, double *a, 
        int *lda, double *b, int *ldb, double *alpha, 
        double *beta, double *u, int *ldu, double *v, int
        *ldv, double *q, int *ldq, double *work, int *iwork, 
        int *info);

int zggsvd_(char *jobu, char *jobv, char *jobq, int *m, 
        int *n, int *p, int *k, int*l, Complex *a, 
        int *lda, Complex *b, int *ldb, double *alpha, 
        double *beta, Complex *u, int *ldu, Complex *v, 
        int *ldv, Complex *q, int *ldq, Complex *work, 
        double *rwork, int *iwork, int *info);

};

//GSVD<Data_Type> Class Implementation
//---------------------------------------------------------------------------
template <class Data_Type>
GSVD<Data_Type>::GSVD(MatrixDense<Data_Type> &AA,MatrixDense<Data_Type> &BB,char* memory){
   
 assert(AA.nr() == BB.nc());
 
  vec_u = vec_v = vec_q = 1;
  
  nrows_A = AA.nr();
  ncols_A = AA.nc();

  nrows_B = BB.nr();
  ncols_B = BB.nc(); 

  if (!strcmp(memory,"copy")){
    A = new MatrixDense<Data_Type>(AA);   
    B = new MatrixDense<Data_Type>(BB);
  }
  else{
    A = &AA;
    B = &BB;
  }
  
}

//----------------------------------------------------------------------------
void GSVD<double>::solve(){

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

//---------------------------------------------------------------------------
template <class Data_Type>
void GSVD<Data_Type>:: info(){

  cout<<"********************************************"<<endl; 
  cout<<"GSVD:"<<endl;
  cout<<"Data Type ="<<data_type<<endl;
  cout<<"Matrix A = "<<nrows_A<<" x "<<ncols_A<<endl;
  cout<<"Matrix L = "<<nrows_B<<" x "<<ncols_B<<endl;
  cout<<"Matrix Type: dense"<<endl;
  cout<<"************"<<endl;
  cout<<"LAPACK Solver:"<<solver_<<endl; 
  cout<<"LAPACK info = "<<info_<<endl;
  cout<<"LAPACK result =  "<<messege_<<endl;
  cout<<"LAPACK time = "<<(double)time/CLOCKS_PER_SEC<<"s"<<endl;
  cout<<"********************************************"<<endl;
}

//---------------------------------------------------------------------------  



#endif


/****************************************************************
 *  Class SVD.h   (Interface to LAPACK)                         *
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

#ifndef SVD_H
#define SVD_H 1

#include "LinearMethods.h"
#include "MatrixDense.h"
#include "Vector.h"
#include "Complex.h"
#include "utils.h"
#include "time.h"

//SVD<Data_Type> Class Definitions
//----------------------------------------------------------------------------
template <class Data_Type>
class SVD: public LinearMethods<Data_Type>{

private:
  MatrixDense<Data_Type>* A;
  MatrixDense<Data_Type>* UU;
  MatrixDense<Data_Type>* VVT;
  ZVector<double>* SS;
  int nrows;
  int ncols;
  clock_t time;
  char* message_;
  char* data_type;
  char* solver_;
  int info_;
  int lvec;
  int rvec;
  
public:
  SVD(MatrixDense<Data_Type>& AA,char* memory = "no");
  void solve();
  void print();
  void info();
  void compute_UV(){ lvec=1;rvec = 1;};
  //MatrixDense<Data_Type>* get_U(){return UU->get_p();};
  //MatrixDense<Data_Type>* get_VT(){return VVT->get_p();};
  //ZVector<double>* get_S(){return SS->get_p();};

  inline MatrixDense<Data_Type>* get_U(){return UU;};
  inline MatrixDense<Data_Type>* get_VT(){return VVT;};
  inline ZVector<double>* get_S(){return SS;};

  inline MatrixDense<Data_Type> solution_U(){return *UU;};
  inline MatrixDense<Data_Type> solution_VT(){return *VVT;};
  inline ZVector<double> solution_S(){return *SS;};

};


//LAPACK declarations
//---------------------------------------------------------------------------  
extern "C"{
int dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *lda, double *s, double *u, int *ldu, double *vt, int *ldvt, double *work, int *lwork,int *info);

int zgesvd_(char *jobu, char *jobvt, int *m, int *n,Complex *a, int *lda, double *s,Complex *u,int *ldu, Complex *vt, int *ldvt,Complex *work,int *lwork, double *rwork, int *info);

}

//SVD<Data_Type> Class Implementation
//---------------------------------------------------------------------------
template <class Data_Type>
SVD<Data_Type>::SVD(MatrixDense<Data_Type> &AA,char* memory){
   
  lvec = rvec = 1;
  
  nrows = AA.nr();
  ncols = AA.nc();


 if (!strcmp(memory,"copy"))
    A = new MatrixDense<Data_Type>(AA);   
  else
    A = &AA;
    
}

//----------------------------------------------------------------------------
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

//---------------------------------------------------------------------------
template <class Data_Type>
void SVD<Data_Type>:: info(){

  cout<<"********************************************"<<endl; 
  cout<<"SVD:"<<endl;
  cout<<"Data Type ="<<data_type<<endl;
  cout<<"Matrix = "<<nrows<<" x "<<ncols<<endl;
  cout<<"Matrix Type: dense"<<endl;
  cout<<"************"<<endl;
  cout<<"LAPACK Solver:"<<solver_<<endl; 
  cout<<"LAPACK info = "<<info_<<endl;
  cout<<"LAPACK result =  "<<message_<<endl;
  cout<<"LAPACK time = "<<(double)time/CLOCKS_PER_SEC<<"s"<<endl;
  cout<<"********************************************"<<endl;
}

//---------------------------------------------------------------------------  
template <class Data_Type>
void SVD<Data_Type>:: print(){
  cout << endl<<endl;
  cout << "U"<<endl;
//  cout <<(*UU)<<endl;
  cout << endl;
  cout << "S"<<endl;
  cout <<( *SS) << endl;
  cout<<endl;
  cout<<"VT";
//  cout <<(*VVT)<<endl;
}
//----------------------------------------------------------------------------





#endif


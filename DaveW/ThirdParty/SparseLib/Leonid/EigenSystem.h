/****************************************************************
 *  EigenSystem.h  (interface to LAPACK)                        *
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

#ifndef EIGEN_SYSTEM_H
#define EIGEN_SYSTEM_H 1


#include "LinearMethods.h"
#include "MatrixDense.h"
#include "MatrixTridiag.h"
#include "Vector.h"
#include "Complex.h"


//EigenSystem<Data_Type> Class Definitions
//----------------------------------------------------------------------------
template <class Data_Type>
class EigenSystem :public LinearMethods<Data_Type>{
  
private:
  ZMatrix<Data_Type>* A;
  ZVector<Complex>* cEV;
  ZVector<double>* dEV;
  ZMatrix<Data_Type>* EVR;
  ZMatrix<Data_Type>* EVL;
  int NN;
  char* messege_;
  int info_;
  int rvec;
  int lvec;
  char* matrix_type;
  //this will be changed with dynamic cast on new compilers

  
public:
  EigenSystem(MatrixDense<Data_Type>& AA,char* memory = "no");
  EigenSystem(MatrixDense<Data_Type>* AA,char* memory = "no");

  EigenSystem(MatrixTridiag<Data_Type>& AA,char* memory = "no");
  EigenSystem(MatrixTridiag<Data_Type>* AA,char* memory = "no");
  
  void solve();
  void print();
  void compute_r_eigenvectors(){rvec = 1;};
  void compute_l_eigenvectors(){lvec = 1;};			  
  char* messege(){return messege_;}; 
  int info(){return info_;};
  int memory_save(int l){memory_save_ = l;};
  ZVector<Complex>* get_eigenvalues(){return cEV;};
  MatrixDense<Data_Type>* get_r_eigenvectors(){return EVR;};
  MatrixDense<Data_Type>* get_l_eigenvectors(){return EVL;};
  
};



//LAPACK declarations
//---------------------------------------------------------------------------  
extern "C"{
  int dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda, double *wr, double *wi, double *vl,int *ldvl, double *vr, int *ldvr, double *work,int *lwork, int *info);

  int zgeev_(char *jobvl, char *jobvr, int *n, Complex *a, int *lda, Complex *w, Complex *vl,int *ldvl, Complex *vr, int *ldvr, Complex *work, int *lwork, double *rwork, int *info);
  
  int dstev_(char *jobz, int *n, double* d,double* e , double *z,int* ldz, double *work, int *info);  
  };

  
  
//EigenSystem<Data_Type> Class Implementation
//---------------------------------------------------------------------------
template <class Data_Type>
EigenSystem<Data_Type>::EigenSystem(MatrixDense<Data_Type> &AA,char* memory){
  
  assert(AA.nr() == AA.nc());
  
  NN = AA.nr();

  matrix_type = "dense";
  lvec = rvec = 0;
  
  if (!strcmp(memory,"copy"))
    A = new MatrixDense<Data_Type>(AA);
  else
    A = &AA;  
}
//---------------------------------------------------------------------------
template <class Data_Type>
EigenSystem<Data_Type>::EigenSystem(MatrixDense<Data_Type> *AA,char* memory){
  
  assert(AA->nr() == AA->nc());
  
  NN = AA->nr();

  matrix_type = "dense";
  lvec = rvec = 0;
  
  if (!strcmp(memory,"copy"))
    A = new MatrixDense<Data_Type>(*AA);
  else
    A = AA;  
}
//---------------------------------------------------------------------------
template <class Data_Type>
EigenSystem<Data_Type>::EigenSystem(MatrixTridiag<Data_Type> &AA,char* memory){
  
  assert(AA.nr() == AA.nc());
  
  NN = AA.nr();

  matrix_type = "tridiag";
  lvec = rvec = 0;
  
  if (!strcmp(memory,"copy"))
    A = new MatrixTridiag<Data_Type>(AA);
  else
    A = &AA;  
}
//---------------------------------------------------------------------------
template <class Data_Type>
EigenSystem<Data_Type>::EigenSystem(MatrixTridiag<Data_Type> *AA,char* memory){
  
  assert(AA->nr() == AA->nc());
  
  NN = AA->nr();

  matrix_type = "tridiag";
  lvec = rvec = 0;
  
  if (!strcmp(memory,"copy"))
    A = new MatrixTridiag<Data_Type>(*AA);
  else
    A = AA;  
}

//----------------------------------------------------------------------------
void EigenSystem<double>::solve(){
  
  if (!strcmp(matrix_type,"dense")){
    
//LAPACK staff:
    char JOBVL,JOBVR;
    int INFO,LDA,LDVL,LDVR,LWORK,N;
    double *VL,*VR,*WORK,*WI,*WR;
    
    N = NN;
    LDA = N;
    WR = new double[N];
    WI = new double[N];
    
    if (rvec == 1) { 
      JOBVR ='V';
      LDVR = N;
      VR = new double[LDVR*N];
      LWORK = 4*N;
      WORK = new double[LWORK];
    }
    else{
      JOBVR = 'N';
      LDVR = 1;
      VR = new double[1];
    }
    
    
    if (lvec == 1){
      JOBVL = 'V';
      LDVL = N; 
      VL = new double[LDVL*N];
      LWORK = 4*N;
      WORK = new double[LWORK];
    }
    else{
      JOBVL = 'N';
      LDVL = 1;
      VL = new double[1];
    }

    
    if ((lvec !=1) && (rvec != 1)){
      LWORK = 3*N;
      WORK = new double[LWORK];
    }
    
    cout << "Calling LAPACK dgeev_() "<<endl;
    dgeev_(&JOBVL,&JOBVR, &N,((MatrixDense<double>*) A)->get_p(), &LDA, WR, WI, VL, &LDVL, VR, &LDVR, WORK, &LWORK, &INFO);
    
    if (INFO == 0)
      messege_ = "Done!";
    if (INFO < 0)
      messege_ =  "Wrong Arguments!";
    if (INFO > 0)
      messege_ = "QR Algorithm Failed !"; 
    
    info_ = INFO;
    
    cEV = new ZVector<Complex>(N,WR,WI);
    
    if (rvec == 1)
      EVR = new MatrixDense<double>(N,N,VR);
    
    if (lvec == 1)
      EVL = new MatrixDense<double>(N,N,VL);
    
  }
  
  if (!strcmp(matrix_type,"tridiag")){

//LAPACK staff:
    char JOBZ;
    int INFO,LDZ,N;
    double *WORK,*Z;
    
    N = NN;
    
    if (rvec == 1)  {
      JOBZ ='V';
      LDZ = N;
      Z = new double[LDZ*N];
      WORK = new double[2*N-2];
    }
    else{
      JOBZ ='N';
      LDZ = 1;
      Z = new double[1];
      WORK = new double[1];
    }
    
    cout << "Calling LAPACK dstev_() "<<endl;
    dstev_(&JOBZ, &N,((MatrixTridiag<double>*)A)->get_pd(),((MatrixTridiag<double>*)A)->get_pdu(),Z,&LDZ,WORK,&INFO);
    
    if (INFO == 0)
      messege_ = "Done!";
    if (INFO < 0)
      messege_ = "Wrong Arguments!";
    if (INFO > 0)
      messege_ = "Failed to Converge!"; 

    info_ = INFO;

    dEV = new ZVector<Complex>(N,((MatrixTridiag<double>*)A)->get_pd(),((MatrixTridiag<double>*)A)->get_pd());
    
    if (rvec == 1)
      EVR = new MatrixDense<double>(N,N,Z);
    
  }
  
  else
    cerr << "I should not be here!"<<endl;
  
  
}  
//---------------------------------------------------------------------------
#if 0
void EigenSystem<Complex>::solve(){
  
//LAPACK staff:
  char JOBVL,JOBVR;
  int INFO,LDA,LDVL,LDVR,LWORK,N;
  double *RWORK;
  Complex *WORK,*W,*VL,*VR;
  
  
  N = NN;
  LDA = N;
  W = new Complex[N];
  LWORK = 2*N;
  WORK = new Complex[LWORK];
  RWORK= new double[2*N];

  
  if (rvec == 1) { 
    JOBVR ='V';
    LDVR = N;
    VR = new Complex[LDVR*N];
  }
  else{
    JOBVR = 'N';
    LDVR = 1;
    VR = new Complex[1];
  }
  
  
  if (lvec == 1){
    JOBVL = 'V';
    LDVL = N; 
    VL = new Complex[LDVL*N];
  }
  else{
    JOBVL = 'N';
    LDVL = 1;
    VL = new Complex[1];
  }
   

  cout << "Calling LAPACK zgeev_() "<<endl;
  zgeev_(&JOBVL,&JOBVR, &N,((MatrixDense<Complex>*) A)->get_p(), &LDA, W, VL, &LDVL, VR, &LDVR, WORK, &LWORK,RWORK, &INFO);
  
  if (INFO == 0)
    messege_ = "Done!";
  if (INFO < 0)
    messege_ = "Wrong Arguments!";
  if (INFO > 0)
    messege_ = "QR Algorithm Failed !"; 

  info_ = INFO;
  
  cEV = new ZVector<Complex>(N,W); 

  if (rvec == 1)
   EVR = new MatrixDense<Complex>(N,N,VR);

  if (lvec == 1)
   EVL = new MatrixDense<Complex>(N,N,VL);
    
}
#endif
//---------------------------------------------------------------------------  
template <class Data_Type>
void EigenSystem<Data_Type>:: print(){
  cout << endl<<endl;
  cout << "Eigenvalues"<<endl;
  cout <<(*cEV)<<endl;
  cout << endl;
  if ( rvec == 1){
  cout<<"R - Eigenvectors";
  cout <<(*(MatrixDense<Data_Type>* )EVR)<<endl;
  }
  if (lvec == 1){
  cout<<"L - Eigenvectors";
  cout <<(*(MatrixDense<Data_Type>* )EVL)<<endl;
  }
}
//----------------------------------------------------------------------------


#endif


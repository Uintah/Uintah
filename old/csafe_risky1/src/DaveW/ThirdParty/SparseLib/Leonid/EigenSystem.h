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

template<class Data_Type> 
void EigenSystem<Data_Type>::solve(){
    cerr << "Error generic EigenSystem<Data_Type>::solve() not implemented.\n";
}

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


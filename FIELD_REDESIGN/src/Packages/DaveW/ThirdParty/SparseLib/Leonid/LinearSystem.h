/****************************************************************
 *  Class LinearSystem.h  (interface to LAPACK & SuperLU)       *
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


#ifndef LINEAR_SYSTEM_H
#define LINEAR_SYSTEM_H 1

#include "LinearMethods.h"
#include "MatrixDense.h"
#include "MatrixTridiag.h"
#include "Vector.h"
#include "Complex.h"
#include <string.h>
#include <time.h>


//LinearSystem<> Class Definitions
//----------------------------------------------------------------------------
template <class Data_Type>
class LinearSystem :public LinearMethods<Data_Type>{

private:
  ZMatrix<Data_Type>* A;
  MatrixDense<Data_Type>* B;
  MatrixDense<Data_Type>* X;
  int n;
  int nrhs;
  clock_t time;
  char* messege_;
  int info_;
  char* matrix_type;
  //this will be changed with dynamic cast on new compilers
  
public:
  LinearSystem(MatrixDense<Data_Type>& AA, ZVector<Data_Type>& bb,char* memory = "no");
  LinearSystem(MatrixDense<Data_Type>& AA, MatrixDense<Data_Type>& BB,char* memory = "no");
  
  LinearSystem(MatrixTridiag<Data_Type>& AA, ZVector<Data_Type>& bb,char* memory = "no");
  LinearSystem(MatrixTridiag<Data_Type>& AA, MatrixDense<Data_Type>& BB,char* memory = "no");
  
  
  void solve();
  void print(); 
  void info();
  inline int memory_save(int l){memory_save_ = l;};
  inline MatrixDense<Data_Type>* get_solution(){return X;};

  inline MatrixDense<Data_Type>  solution(){return *X;};

  inline ZVector<Data_Type> solution(int i){ ZVector<Data_Type> V;V = (*X)(i);return V;}
  
};

//----------------------------------------------------------------------------
//LAPACK
extern "C"{
  int dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
  
  int zgesv_(int *n, int *nrhs, Complex *a, int *lda, int *ipiv, Complex *b, int *ldb, int *info);
  
  int dgtsv_(int *n, int *nrhs, double *dl, double *d, double *du, double *b, int *ldb, int *info);
int dptsv_(int *n, int *nrhs, double *d, double *e, double *b, int *ldb, int *info);
  
  int zgtsv_(int *n, int *nrhs, Complex *dl, Complex *d, Complex *du, Complex *b, int *ldb, int *info);

}

//Implementation
//----------------------------------------------------------------------------
template <class Data_Type>
LinearSystem<Data_Type>::LinearSystem(MatrixDense<Data_Type> &AA,ZVector<Data_Type>& bb, char* memory){
  
  assert(AA.nr() == AA.nc());
  assert(AA.nc() == bb.size());
  
  n = bb.size();
  nrhs = 1;
  
  matrix_type = "dense";
  
  if (!strcmp(memory,"copy")){
    A = new MatrixDense<Data_Type>(AA);
    MatrixDense<Data_Type>* lB = new MatrixDense<Data_Type>(n,1,bb.get_p());
    //this constructor does not copy things around, it just assembles
    B = new MatrixDense<Data_Type>(*lB);
    //this one makes a copy of data
  }
  else{
    A = &AA;
    B = new MatrixDense<Data_Type>(n,1,bb.get_p());
   }

}

//----------------------------------------------------------------------------
template <class Data_Type>
LinearSystem<Data_Type>::LinearSystem(MatrixDense<Data_Type> &AA,MatrixDense<Data_Type>& BB, char* memory){
  
  assert(AA.nr() == AA.nc());
  assert(AA.nc() == BB.nr ());
  
  n = BB.nr();
  nrhs = BB.nc();
  
  matrix_type = "dense";
  
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
template <class Data_Type>
LinearSystem<Data_Type>::LinearSystem(MatrixTridiag<Data_Type> &AA,ZVector<Data_Type>& bb, char* memory){
  
  assert(AA.nr() == AA.nc());
  assert(AA.nr() == bb.size());
  
  n = bb.size();
  nrhs = 1;
  
  matrix_type = "tridiag";
  
  if (!strcmp(memory,"copy")){
    A = new MatrixTridiag<Data_Type>(AA);
    MatrixDense<Data_Type>* lB = new MatrixDense<Data_Type>(n,1,bb.get_p());
    B = new MatrixDense<Data_Type>(*lB);
  }
  else{
    A = &AA;
    B = new MatrixDense<Data_Type>(n,1,bb.get_p());
  }
  
}
//----------------------------------------------------------------------------
template <class Data_Type>
LinearSystem<Data_Type>::LinearSystem(MatrixTridiag<Data_Type> &AA,MatrixDense<Data_Type>& BB, char* memory){
  
  assert(AA.nr() == AA.nc());
  assert(AA.nr() == BB.nr());
  
  n = BB.nr();
  nrhs = BB.nc();
  
  matrix_type = "tridiag";
  
  if (!strcmp(memory,"copy")){
    A = new MatrixTridiag<Data_Type>(AA);
    B = new MatrixDense<Data_Type>(BB);
  }
  else{
    A = &AA;
    B = &BB;
  }
  
}

//----------------------------------------------------------------------------
template <class Data_Type>
void LinearSystem<Data_Type>:: print(){
 
  cout << endl;
  cout << "Linear System Solution:"<<endl<<endl;
  cout <<(* X)<<endl;
  
}

template <class Data_Type>
void LinearSystem<Data_Type>:: solve(){
    cerr << "Error generic LinearSystem<Data_Type>::solve() not implemented.\n";
}

template <class Data_Type>
void LinearSystem<Data_Type>:: info(){
    cerr << "Error generic LinearSystem<Data_Type>::info() not implemented.\n";
}

#endif



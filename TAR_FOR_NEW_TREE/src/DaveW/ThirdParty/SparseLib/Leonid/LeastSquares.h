/****************************************************************
 *  Class LeastSquares.h  (interface to LAPACK)                 *
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


#ifndef LEAST_SQUARES_H
#define LEAST_SQUARES_H 1

#include "LinearMethods.h"
#include "MatrixDense.h"
#include "Vector.h"
#include "Complex.h"
#include <string.h>
#include <time.h>



//LeastSquares<> Class Definitions
//----------------------------------------------------------------------------
template <class Data_Type>
class LeastSquares :public LinearMethods<Data_Type>{

private:
  ZMatrix<Data_Type>* A;
  MatrixDense<Data_Type>* B;
  MatrixDense<Data_Type>* X;
  ZVector<double>* SV;
  int nrows;
  int ncols;
  double rcond;
  int nrhs;
  clock_t time;
  char* message_;
  char* data_type;
  char* solver_;
  int info_;
 
  //this will be changed with dynamic cast on new compilers
  
public:
  LeastSquares(MatrixDense<Data_Type>& AA, ZVector<Data_Type>& bb,char* memory = "no");
  LeastSquares(MatrixDense<Data_Type>& AA, MatrixDense<Data_Type>& BB,char* memory = "no");
  
 
  
  void solve();
  void print(); 
  void info();
  int memory_save(int l){memory_save_ = l;};
  MatrixDense<Data_Type>* get_solution(){return X;};

  MatrixDense<Data_Type>  solution(){return *X;};

  inline ZVector<Data_Type> solution(int i){   
    ZVector<Data_Type> V;   V = (*X)(i);return V;}
  
};

//----------------------------------------------------------------------------
//LAPACK
extern "C"{
 
int dgelss_(int *m, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *s, double *rcond, int *rank, double *work, int *lwork,int *info);

int zgelss_(int *m, int *n, int *nrhs, Complex *a, int *lda, Complex *b, int *ldb, double *s, double *rcond, int *rank, Complex *work, int *lwork, double *rwork, int *info);

  
}

//Implementation
//----------------------------------------------------------------------------
template <class Data_Type>
LeastSquares<Data_Type>::LeastSquares(MatrixDense<Data_Type> &AA,ZVector<Data_Type>& bb, char* memory){
  
  assert(AA.nc() == bb.size());
  
  nrows = AA.nr();
  ncols = AA.nc();
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
LeastSquares<Data_Type>::LeastSquares(MatrixDense<Data_Type> &AA,MatrixDense<Data_Type>& BB, char* memory){
  
 
  assert(AA.nc() == BB.nr ());
  
  nrows = AA.nr();
  ncols = AA.nc();
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
void LeastSquares<Data_Type>:: info(){

  cout<<"********************************************"<<endl; 
  cout<<"Least Squares:"<<endl;
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
//----------------------------------------------------------------------------
 
template <class Data_Type>
void LeastSquares<Data_Type>:: solve(){
    cerr << "Error generic LeastSquares<Data_Type>::solve() not implemented.\n";
}


template <class Data_Type>
void LeastSquares<Data_Type>:: print(){

 cout << "SV = "<<endl;
 cout << SV<<endl;
 cout << endl;
 cout << "X = "<<endl;
 cout << X<<endl;
  
}
//----------------------------------------------------------------------------

#endif

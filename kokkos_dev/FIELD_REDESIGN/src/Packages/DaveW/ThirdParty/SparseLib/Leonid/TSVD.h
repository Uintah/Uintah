/****************************************************************
 *  Class TSVD.h   (Truncated SVD, INVPACK )                     *
 *                                                              *
 *  Written by:                                                 *
 *   Leonid Zhukov                                              *
 *   Department of Computer Science                             *
 *   University of Utah                                         *
 *   February 1998                                              *
 *                                                              *
 *  Copyright (C) 1997 SCI Group                                *
 *                                                              *
 *                                                              *
 ****************************************************************/

#ifndef TSVD_H
#define TSVD_H 1

#include "LinearMethods.h"
#include "MatrixDense.h"
#include "Vector.h"
#include "Complex.h"
#include "utils.h"
#include "SVD.h"



//TSVD<Data_Type> Class Definitions
//---------------------------------------------------------------------------
template <class Data_Type>
class TSVD: public LinearMethods<Data_Type>{
  
private:
  MatrixDense<Data_Type> *A;
  ZVector<Data_Type> *B;
  int nrows;
  int ncols;
  double sigma;
  ZVector<double>* P;
  ZVector<double>* X;
  MatrixDense<double>* U;
  MatrixDense<double>* VT;
  ZVector<double>* S;
  

public:
  TSVD(MatrixDense<Data_Type> &AA,ZVector<Data_Type> &BB);
  void factor();
  void solve();
  void set_sigma(double sigma);
  void print();
  void info();
  
};

//TSVD<Data_Type> Class Implementation
//---------------------------------------------------------------------------
template<class Data_Type>
TSVD<Data_Type>::TSVD(MatrixDense<Data_Type> &AA,ZVector<Data_Type> &BB) 
{

  A = &AA;
  B = &BB;
  sigma = 0.000001;
  
  nrows = A->nr();
  ncols = A->nc();  

  P = new ZVector<double>(nrows,0.0);
  X = new ZVector<double>(ncols,0.0);
  
}

//---------------------------------------------------------------------------
template <class Data_Type>
void TSVD<Data_Type>:: info(){
  
  cout<<"********************************************"<<endl; 
  cout<<"TSVD:"<<endl;
  cout<<"Cut-off sigma = "<<sigma<<endl;
  cout<<"********************************************"<<endl;
}

template <class Data_Type>
void TSVD<Data_Type>:: solve(){
    cerr << "Error generic TSVD<Data_Type>:: solve() not implemented.\n";
}

//---------------------------------------------------------------------------  
template <class Data_Type>
void TSVD<Data_Type>:: print(){
//cout << "Condition Number = "<<S(0)/S(Min(nrows,ncols))<<endl;  
  cout<<"S = "<<(*S)<<endl;  
  cout<<"P = "<<(*P)<<endl;
  cout<<"X = "<<(*X)<<endl;
  
}
//----------------------------------------------------------------------------

#endif

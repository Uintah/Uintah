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

template <class Data_Type>
void GSVD<Data_Type>:: solve(){
    cerr << "Error generic GSVD<Data_Type>:: solve() not implemented.\n";
}

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


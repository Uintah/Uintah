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

template <class Data_Type>
void SVD<Data_Type>:: solve(){
    cerr < "Error - SVD<>::solve() undefined for generic.\n";
}

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

/****************************************************************
 *  Class MatrixTridiag.h                                       *
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

#ifndef MATRIX_TRIDIAG_H
#define MATRIX_TRIDIAG_H 1

#include <iostream>
#include <assert.h>

#include "Matrix.h"
#include "Vector.h"
#include "Index.h"
#include "utils.h"
#include "Complex.h"

//using namespace std;

template<class Data_Type> class MatrixDense;


//MatrixTridiag Class Definitions
//-------------------------------------------------------------------------
template <class Data_Type>
class MatrixTridiag:public ZMatrix<Data_Type>{
  
private:
  
  Data_Type *d;
  Data_Type *dl;
  Data_Type *du;
  
public:  
  
//constructors:
  MatrixTridiag():d(0),du(0),dl(0){}
  MatrixTridiag(int nrows);
  MatrixTridiag(int nrows,Data_Type x);
  MatrixTridiag(int nrows,Data_Type* xd,Data_Type* xdl, Data_Type* xdu );
  MatrixTridiag(const MatrixTridiag &B);  
  
//destructor:
  ~MatrixTridiag();

//"In place, modifying the matrix" operations:
  void add(const MatrixTridiag &B,int col_begin=0,int col_end=0);
  void subtr(const MatrixTridiag &B,int col_begin=0,int col_end=0);
  void mult(const MatrixTridiag &B,MatrixTridiag &tmp,
	    int col_begin=0,int col_end=0);
  void mult(const ZVector<Data_Type> &b,ZVector<Data_Type> &x ,
	    int col_begin=0, int col_end=0);
  void mult(Data_Type alpha,int col_begin=0, int col_end=0);
  

//"Put results  in the provided space" operations:  
  void sum(const MatrixTridiag &A,const MatrixTridiag &B,
	   int col_begin=0,int col_end=0);
  void diff(const MatrixTridiag &A,const MatrixTridiag &B,
	    int col_begin=0,int col_end=0);
  void prod(const MatrixTridiag &A,const MatrixTridiag &B,
	    int col_begin=0,int col_end=0);
  void prod(const MatrixTridiag &A,Data_Type alpha,
	    int col_begin=0, int col_end=0);
  void set(const MatrixTridiag &B,int col_begin=0, int col_end=0); 
  void set(const MatrixTridiag &B,Index IB,Index JB,
	   int col_begin=0,int col_end=0);   
  
  
//"Using new " operations:
  MatrixTridiag &operator=(const MatrixTridiag& B); 
  ZVector<Data_Type> operator*(const ZVector<Data_Type> &b);
  MatrixTridiag operator+(const MatrixTridiag& B);
  MatrixTridiag operator-(const MatrixTridiag& B);  
  MatrixTridiag operator*(const MatrixTridiag& B);  
  MatrixTridiag transpose();
  MatrixTridiag operator*(Data_Type x);
  friend MatrixTridiag operator*(Data_Type x, const MatrixTridiag &B);


  Data_Type* get_pd(){return d;};
  Data_Type* get_pdu(){return du;};
  Data_Type* get_pdl(){return dl;};
  Data_Type &operator()(int i, int j);
  MatrixDense<Data_Type> to_Dense();
  

//Output:
  void print();
  void read(char* filename);
  void write(char* filename);
  friend std::ostream& operator << (std::ostream& output, MatrixTridiag<Data_Type> &A);

  
  
};




//MatrixTridiag Class Implementation
//--------------------------------------------------------------------
template <class Data_Type>
MatrixTridiag<Data_Type>::MatrixTridiag(int nrows){
  
  this->nrows = nrows;
  this->ncols = nrows;
  
  d = new Data_Type [nrows];
  dl = new Data_Type [nrows-1];
  du = new Data_Type [nrows-1]; 
  
}

//---------------------------------------------------------------------
template <class Data_Type>
MatrixTridiag<Data_Type>::MatrixTridiag(int nrows,Data_Type x){
  int i;
  
  this->nrows = nrows;
  this->ncols = nrows;
  
  d = new Data_Type [nrows];
  dl = new Data_Type [nrows-1];
  du = new Data_Type [nrows-1]; 
  
  
  for(i=0;i<nrows-1;i++){
    d[i] = x;
    du[i] = x;
    dl[i] = x;
  }
  d[nrows-1] = x;
  
}

//---------------------------------------------------------------------
template <class Data_Type>
MatrixTridiag<Data_Type>::MatrixTridiag(int nrows,Data_Type* xd,Data_Type* xdl, Data_Type* xdu ){
  
  this->nrows = nrows;
  this->ncols = nrows;
  
  d = xd;
  dl =xdl;
  du = xdu;  
}

//---------------------------------------------------------------------
template <class Data_Type>
MatrixTridiag<Data_Type>::MatrixTridiag(const MatrixTridiag<Data_Type> &B){
  int i;
  
  nrows = B.nrows;
  ncols = B.ncols;
  
  d = new Data_Type [nrows];
  dl = new Data_Type [nrows-1];
  du = new Data_Type [nrows-1];
  
  for(i=0;i<nrows-1;i++){
    d[i] = B.d[i];
    du[i] = B.du[i];
    dl[i] = B.dl[i];
  }
  d[nrows-1] = B.d[nrows-1];
  
}

//---------------------------------------------------------------------
template <class Data_Type>
MatrixTridiag<Data_Type>::~MatrixTridiag(){
  
  if(d){
    delete [] d;
    delete [] du;
    delete [] dl;
  }
  
}

//-----------------------------------------------------------------
template <class Data_Type>
MatrixTridiag<Data_Type> &MatrixTridiag<Data_Type>::operator=(const MatrixTridiag<Data_Type> &B){
  
  if(d){
    delete [] d;
    delete [] du;
    delete [] dl;
  }
  
  nrows = B.nrows;
  ncols = B.ncols;
  
  d = new Data_Type [nrows];
  dl = new Data_Type [nrows-1];
  du = new Data_Type [nrows-1];
  
  for(int i=0;i<nrows-1;i++){
    d[i] = B.d[i];
    du[i] = B.du[i];
    dl[i] = B.dl[i];
  }
  d[nrows-1] = B.d[nrows-1];
  
  return *this; 
}

//-----------------------------------------------------------------
template <class Data_Type>
MatrixTridiag<Data_Type> MatrixTridiag<Data_Type>::operator+(const MatrixTridiag<Data_Type> &B){

  assert((B.nrows == nrows)&&(B.ncols == ncols));  
  
  MatrixTridiag<Data_Type>  C(nrows); 
  
  int i;  
  for(i=0;i<nrows-1;i++){
    C.d[i] = d[i] + B.d[i]; 
    C.du[i] = du[i] + B.du[i];
    C.dl[i] = dl[i] + B.dl[i];
  }

  C.d[nrows-1] = d[nrows-1] + B.d[nrows-1];
       
  return(C); 
}

//-----------------------------------------------------------------
template <class Data_Type>
MatrixTridiag<Data_Type> MatrixTridiag<Data_Type>::operator-(const MatrixTridiag<Data_Type>&B)
{
  assert((B.nrows == nrows)&&(B.ncols == ncols));
  
  MatrixTridiag<Data_Type>  C (nrows); 
    
  int i;  
  for(i=0;i<nrows-1;i++){
    C.d[i] = d[i] - B.d[i]; 
    C.du[i] = du[i] - B.du[i];
    C.dl[i] = dl[i] - B.dl[i];
  }
  
  C.d[nrows-1] = d[nrows-1] - B.d[nrows-1];
            
  return(C);
}

//-------------------------------------------------------------
template <class Data_Type>
MatrixTridiag<Data_Type> MatrixTridiag<Data_Type>::operator*(Data_Type alpha){
  
  MatrixTridiag<Data_Type>  C(nrows);
  
  int i;  
  for(i=0;i<nrows-1;i++){
    C.d[i] = alpha*d[i]; 
    C.du[i] = alpha* du[i];
    C.dl[i] = alpha*dl[i];
  }
  
  C.d[nrows-1] = alpha*d[nrows-1];

  return (C);
}

//-----------------------------------------------------------------
template <class Data_Type>
MatrixTridiag<Data_Type> operator*(Data_Type alpha, const MatrixTridiag<Data_Type>  &B){
  
  MatrixTridiag<Data_Type>  C(B.nrows);

  int i;  
  for(i=0;i<nrows-1;i++){
    C.d[i] = alpha*B.d[i]; 
    C.du[i] = alpha*B.du[i];
    C.dl[i] = alpha*B.dl[i];
  }
  
  C.d[nrows-1] = alpha*B.d[nrows-1];
 
  return (C);

} 

//---------------------------------------------------------------------
template <class Data_Type>
MatrixTridiag<Data_Type> MatrixTridiag<Data_Type> ::operator*(const MatrixTridiag<Data_Type> &B){

  assert(ncols == B.nrows);
 
  MatrixTridiag<Data_Type> C(nrows,0.0);  
  
  
  return(C);
}

//---------------------------------------------------------------------
template <class Data_Type>
MatrixTridiag<Data_Type> MatrixTridiag<Data_Type> ::transpose(){

  Data_Type * xd = new Data_Type [nrows];
  Data_Type * xdl = new Data_Type [nrows-1];
  Data_Type * xdu = new Data_Type [nrows-1];
  
  for(int i=0;i<nrows-1;i++){
    xd[i] = d[i];
    xdu[i] = dl[i];
    xdl[i] = du[i];
  }
  xd[nrows-1] = d[nrows-1];

 
  MatrixTridiag<Data_Type> C(nrows,xd,xdl,xdu);  
 
  return(C);
}


//---------------------------------------------------------------------
template <class Data_Type>
ZVector<Data_Type> MatrixTridiag<Data_Type> ::operator*(const ZVector<Data_Type> &b){

  assert(ncols == b.Size);
  
  ZVector<Data_Type> c(nrows,0.0);
  
  c.a[0] = d[0]*b.a[0] + du[0]*b.a[1];
  
  int i;
  for(i=1;i<nrows-1;i++)
    c.a[i] = dl[i-1]*b.a[i-1] + d[i]*b.a[i] + du[i]*b.a[i+1];
  
  c.a[nrows-1] = dl[nrows-2]*b.a[nrows-2] + d[nrows-1]*b.a[nrows-1];
  
  return(c);
}

//---------------------------------------------------------------------
template <class Data_Type>
void MatrixTridiag<Data_Type>::set(const MatrixTridiag<Data_Type> &B,
				   int col_begin, int col_end)
{
  assert((B.nrows == nrows)&&(B.ncols == ncols));
  
  for(i=0;i<nrows-1;i++){
    d[i] = B.d[i];
    du[i] = B.du[i];
    dl[i] = B.dl[i];
  }
  d[nrows-1] = B.d[nrows-1];
  
}


//---------------------------------------------------------------------
template <class Data_Type>
Data_Type& MatrixTridiag<Data_Type>::operator()(int i, int j){
  
  assert((i>=0)&&(i<nrows)&&(j>=0)&&(j<ncols));

  Data_Type* x = new Data_Type;
  *x = 0.0;
    
  if ( i==j )
    return(d[i]);

  else if (j == i+1)
    return(du[i]);
  
  else if (j == i-1)
    return (dl[j]);
  
  else
      return (*x);
  
}

//---------------------------------------------------------------------
template <class Data_Type>
void  MatrixTridiag<Data_Type>:: sum(const MatrixTridiag<Data_Type> &A,
				     const MatrixTridiag<Data_Type> &B,
				     int col_begin, int col_end)
{
  assert((A.nrows == B.nrows)&&(A.ncols == B.ncols));
  assert((A.nrows == nrows)&&(A.ncols == ncols));
  
  int i;  
  for(i=0;i<nrows-1;i++){
    d[i] = A.d[i] + B.d[i]; 
    du[i] = A.du[i] + B.du[i];
    dl[i] = A.dl[i] + B.dl[i];
  }
  
  d[nrows-1] = A.d[nrows-1] + B.d[nrows-1];
  
} 

//---------------------------------------------------------------------
template <class Data_Type>
void  MatrixTridiag<Data_Type>:: diff(const MatrixTridiag<Data_Type> &A,
				       const MatrixTridiag<Data_Type> &B,
				       int col_begin, int col_end)
{
  assert((A.nrows == B.nrows)&&(A.ncols == B.ncols));
  assert((A.nrows == nrows)&&(A.ncols == ncols));
  
  int i;  
  for(i=0;i<nrows-1;i++){
    d[i] = A.d[i] - B.d[i]; 
    du[i] = A.du[i] - B.du[i];
    dl[i] = A.dl[i] - B.dl[i];
  }
  
  d[nrows-1] = A.d[nrows-1] - B.d[nrows-1];
  
}

//---------------------------------------------------------------------
template <class Data_Type>
void  MatrixTridiag<Data_Type>:: prod(const MatrixTridiag<Data_Type> &A,
				       const MatrixTridiag<Data_Type> &B,
				       int col_begin, int col_end)
{
  assert((A.ncols == B.nrows)&&(nrows == A.nrows)&&(ncols == B.ncols));
 
}

 //-----------------------------------------------------------------
template <class Data_Type>
void  MatrixTridiag<Data_Type>:: prod(const MatrixTridiag<Data_Type> &A,
				       Data_Type alpha,int col_begin, int col_end)
{
  assert((ncols == A.ncols)&&(nrows == A.nrows));  
 
 int i;  
  for(i=0;i<nrows-1;i++){
    d[i] = alpha*A.d[i]; 
    du[i] = alpha* A.du[i];
    dl[i] = alpha*A.dl[i];
  }
  
  d[nrows-1] = alpha*A.d[nrows-1];
  

} 

//---------------------------------------------------------------------
template <class Data_Type>
void  MatrixTridiag<Data_Type>:: add(const MatrixTridiag<Data_Type> &B,
				      int col_begin, int col_end)
 
{
  assert((B.nrows == nrows)&&(B.ncols == ncols));  

  int i;  
  for(i=0;i<nrows-1;i++){
    d[i] = d[i] + B.d[i]; 
    du[i] = du[i] + B.du[i];
    dl[i] = dl[i] + B.dl[i];
  }

  d[nrows-1] = d[nrows-1] + B.d[nrows-1];

  
}

//---------------------------------------------------------------------
template <class Data_Type>
void  MatrixTridiag<Data_Type>:: subtr(const MatrixTridiag<Data_Type> &B,
					int col_begin, int col_end)
{
  assert((B.nrows == nrows)&&(B.ncols == ncols));  

  int i;  
  for(i=0;i<nrows-1;i++){
    d[i] = d[i] - B.d[i]; 
    du[i] = du[i] - B.du[i];
    dl[i] = dl[i] - B.dl[i];
  }

  d[nrows-1] = d[nrows-1] + B.d[nrows-1];
  
}

//---------------------------------------------------------------------
template <class Data_Type>
void  MatrixTridiag<Data_Type>:: mult(const MatrixTridiag<Data_Type> &B,
				       MatrixTridiag<Data_Type> &tmp,
				       int col_begin, int col_end)
{
  
  
}

//---------------------------------------------------------------------
template <class Data_Type>
void  MatrixTridiag<Data_Type>::mult(const ZVector<Data_Type>& b,
				      ZVector<Data_Type>& x ,
				      int col_begin, int col_end)
{
  assert((b.Size == ncols)&&(b.Size == x.Size));
  
   x.a[0] = d[0]*b.a[0] + du[0]*b.a[1];
  
  int i;
  for(i=1;i<nrows-1;i++)
    x.a[i] = dl[i-1]*b.a[i-1] + d[i]*b.a[i] + du[i]*b.a[i+1];
  
  x.a[nrows-1] = dl[nrows-2]*b.a[nrows-2] + d[nrows-1]*b.a[nrows-1];
    

}

//-----------------------------------------------------------------
template <class Data_Type>
void  MatrixTridiag<Data_Type>:: mult(Data_Type alpha,int col_begin,
				       int col_end)
{
     
  int i;  
  for(i=0;i<nrows-1;i++){
    d[i] = alpha*d[i]; 
    du[i] = alpha* du[i];
    dl[i] = alpha*dl[i];
  }
  
  d[nrows-1] = alpha*d[nrows-1];
  
  
}
//---------------------------------------------------------------------
template <class Data_Type>
MatrixDense<Data_Type> MatrixTridiag<Data_Type>::to_Dense(){
  MatrixDense<Data_Type> A(nrows,ncols,0.0);
  
  A.a[0][0] = d[0];
  A.a[1][0] = du[0];
  
  for(int i=1;i<nrows-1;i++){
    A.a[i-1][i] = dl[i-1];
    A.a[i][i] = d[i];
    A.a[i+1][i] = du[i];
  }
  
  A.a[nrows-2][nrows-1] = dl[nrows-2];
  A.a[nrows-1][nrows-1] = d[nrows-1];
  
return A;

} 
//---------------------------------------------------------------------
template <class Data_Type>
std::ostream &operator<< (std::ostream &output,MatrixTridiag<Data_Type> &A){
  output << endl;  
  for(int i=0 ;i < A.nr();i++){
    output<<"[";
    for(int j=0 ;j < A.nc();j++)
      output<<A(i,j)<<" ";
    output<<"]";
    output<<endl;
  }

  return(output);
}

//---------------------------------------------------------------------
template <class Data_Type>
void MatrixTridiag<Data_Type>::print(){
  
  cout<<"nrows = "<<nrows<<endl;
  cout<<"ncols = "<<ncols<<endl;
  
  cout<<"lower diagonal: ["; 
  for(int i=0 ;i < nrows-1;i++)
    cout << dl[i]<<" ";
  cout<<" ]"<<endl;
  
  cout<<"diagonal: ["; 
  for(i=0 ;i < nrows;i++)
    cout << d[i]<<" ";
  cout<<" ]"<<endl;
  
  cout<<"upper diagonal: ["; 
  for(i=0 ;i < nrows-1;i++)
    cout << du[i]<<" ";
  cout<<" ]"<<endl;
  
} 

//---------------------------------------------------------------------
template <class Data_Type>
void MatrixTridiag<Data_Type>::read(char * filename){

  int i;
  char  matrix[80];
  char matrix_type[80];
  
  ifstream file_in(filename);

  if(!file_in){
     cerr << "Can not open file "<<filename<<endl;
    exit(0);
  }
  
  file_in >> matrix;
  if(strcmp(matrix,"matrix")){
    cerr << "Error: Not a MATRIX in "<<filename<<"!"<<endl;
    exit(0);
  }
  
  file_in >> matrix_type;
  if(strcmp(matrix_type,"tridiag")){
    cerr << "Error: Not a TRIDIAGONAL matrix in "<<filename<<"!"<<endl;
    exit(0);
  }
  
  file_in >> nrows;
  file_in >> ncols;
  
  d = new Data_Type[nrows];
  dl = new Data_Type[nrows-1];
  du = new Data_Type[nrows-1];

  
  for(i=0;i<nrows-1;i++)
    file_in>>dl[i];
  
  for(i=0;i<nrows;i++)
    file_in>>d[i];
  
  for(i=0;i<nrows-1;i++)
    file_in>>du[i];
  
  file_in.close();
  
}
  


//---------------------------------------------------------------------
template <class Data_Type>
void MatrixTridiag<Data_Type>::write(char * filename){

  int i,j,ind_l,ind_r;
  ofstream file_out(filename);

  if(!file_out){
     cerr << "Can not open file "<<filename<<endl;
    exit(0);
  }


  file_out <<"matrix ";
  file_out<<"tridiag"<<endl;
  file_out<<endl;
  file_out << nrows<<" ";
  file_out << ncols<<" "; 
  file_out << endl;
  file_out << endl;
 
 
  for(i=0;i<nrows-1;i++)
    file_out<<dl[i]<<" ";
  file_out<<endl;
  
  for(i=0;i<nrows;i++)
    file_out<<d[i]<<" ";
  file_out<<endl;
  
  for(i=0;i<nrows-1;i++)
    file_out<<du[i]<<" ";
  file_out<<endl;
 
  
  file_out.close();

}
  

//---------------------------------------------------------------------

#endif






/****************************************************************
 *  Class MatrixDense.h                                         *
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


#ifndef MATRIX_DENSE_H
#define MATRIX_DENSE_H 1

#include <iostream.h>
#include <fstream.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "Matrix.h"
#include "Vector.h"
#include "Index.h"
#include "utils.h"


template<class Data_Type> class MatrixSparse;
template<class Data_Type> class MatrixTridiag;


//MatrixDense Class Definitions
//-------------------------------------------------------------------------
template <class Data_Type>
class MatrixDense:public ZMatrix<Data_Type>{


  friend class MatrixTridiag<Data_Type>; 

private:
  
  Data_Type **a;

  Data_Type *p;

  
public:  
  
//constructors:
  MatrixDense():a(new Data_Type*){}
  MatrixDense(int nrows, int ncols);
  MatrixDense(int nrows,int ncols,Data_Type x);
  MatrixDense(int nrows,int ncols,Data_Type* x);
  MatrixDense(const MatrixDense &B);  
  
//destructor:
  ~MatrixDense();

//"In place, modifying the matrix" operations:
  void add(const MatrixDense &B,int col_begin=0,int col_end=0);
  void subtr(const MatrixDense &B,int col_begin=0,int col_end=0);
  void mult(const MatrixDense &B,MatrixDense &tmp,
	    int col_begin=0,int col_end=0);
  void mult(const ZVector<Data_Type> &b,ZVector<Data_Type> &x ,
	    int col_begin=0, int col_end=0);
  void mult(Data_Type alpha,int col_begin=0, int col_end=0);
  

//"Put results  in the provided space" operations:  
  void sum(const MatrixDense &A,const MatrixDense &B,
	   int col_begin=0,int col_end=0);
  void diff(const MatrixDense &A,const MatrixDense &B,
	    int col_begin=0,int col_end=0);
  void prod(const MatrixDense &A,const MatrixDense &B,
	    int col_begin=0,int col_end=0);
  void prod(const MatrixDense &A,Data_Type alpha,
	    int col_begin=0, int col_end=0);
  void set(const MatrixDense &B,int col_begin=0, int col_end=0);
  void set(Index IA,Index JA,const MatrixDense &B,Index IB,Index JB,
	   int col_begin=0, int col_end=0);
  void set(const MatrixDense &B,Index IB,Index JB,
	   int col_begin=0,int col_end=0);   
  void set(Index IA,Index JA,const MatrixDense &B,
	   int col_begin=0 ,int col_end=0);
  
//"Using new " operations:
  MatrixDense &operator=(const MatrixDense& B); 
  MatrixDense &operator=(Data_Type x); 
  ZVector<Data_Type> operator*(const ZVector<Data_Type> &b);
  MatrixDense operator+(const MatrixDense& B);
  MatrixDense operator-(const MatrixDense& B);  
  MatrixDense operator*(const MatrixDense& B);  
  MatrixDense transpose();
  MatrixDense operator*(Data_Type x);
  friend MatrixDense operator*(Data_Type x, const MatrixDense &B);

  //  ZVector<Data_Type> row(int i);
  //ZVector<Data_Type>column(int j);
  

  inline Data_Type* get_p(){return p;};
  Data_Type &operator()(int i, int j);
  ZVector<Data_Type> operator()(int i);
  MatrixDense operator()(Index I, Index J);





  
//Input & Output:
  void read(char* filename);
  void write(char* filename);
  friend ostream& operator << (ostream& output, MatrixDense<Data_Type> &A);
  void info();
  
};




//MatrixDense Class Implementation
//--------------------------------------------------------------------
template <class Data_Type>
MatrixDense<Data_Type>::MatrixDense(int nrows,int ncols){
  
  this->nrows = nrows;
  this->ncols = ncols;
  
  a = new Data_Type*[ncols];
  
  Data_Type*  mp = new Data_Type[nrows*ncols];
  for(int j=0;j<ncols;j++){
    a[j]=mp;
    mp+=nrows;
  }
  
 p = a[0];
}

//---------------------------------------------------------------------
template <class Data_Type>
MatrixDense<Data_Type>::MatrixDense(int nrows,int ncols,Data_Type x){
  int i,j;
  
  this->nrows = nrows;
  this->ncols = ncols;
  
  a = new Data_Type*[ncols];

  Data_Type* mp = new Data_Type[nrows*ncols];
  for(j=0;j<ncols;j++){
    a[j]=mp;
    mp+=nrows;
  }
  
  p = a[0];
  
  for(i=0;i<nrows*ncols;i++)
          p[i] = x;

}

//---------------------------------------------------------------------
template <class Data_Type>
MatrixDense<Data_Type>::MatrixDense(int nrows,int ncols,Data_Type *x){
  int i,j;
  
  this->nrows = nrows;
  this->ncols = ncols;
  
  a = new Data_Type*[ncols];
  
  for(i=0;i<ncols;i++){
    a[i]= x;
    x += nrows;
  }
  
  p = a[0];
}
//---------------------------------------------------------------------
template <class Data_Type>
MatrixDense<Data_Type>::MatrixDense(const MatrixDense<Data_Type> &B){
  int i,j;
  
  nrows = B.nrows;
  ncols = B.ncols;
  
  a = new Data_Type*[ncols];
  
  Data_Type* mp = new Data_Type[nrows*ncols];
  for(j=0;j<ncols;j++){
    a[j]=mp;
    mp+=nrows;
  }

  p = a[0];
  
  for(i=0;i<nrows*ncols;i++)
      p[i] = B.p[i];
  
}

//---------------------------------------------------------------------
template <class Data_Type>
MatrixDense<Data_Type>::~MatrixDense(){

  if(a[0]) delete[] a;
  if(p) delete[] p;
    
}

//-----------------------------------------------------------------
template <class Data_Type>
MatrixDense<Data_Type> &MatrixDense<Data_Type>::operator=(const MatrixDense<Data_Type> &B){
  
  int i,j; 
  
 if(a){
    delete[] a[0];
    delete[] a;   
  }
  
  nrows = B.nrows;
  ncols = B.ncols;
  
  a = new Data_Type*[ncols];
  
  Data_Type* mp = new Data_Type[nrows*ncols];
  for(j=0;j<ncols;j++){
    a[j]=mp;
    mp+=nrows;
  }
  
  p = a[0];
  
  for(i=0;i<nrows*ncols;i++)
      p[i] = B.p[i];
  
  return *this; 
}
//-----------------------------------------------------------------
template <class Data_Type>
MatrixDense<Data_Type> &MatrixDense<Data_Type>::operator=(Data_Type x){
     
  for(int i=0;i<nrows*ncols;i++)
      p[i] = x;
  
  return *this; 
}



//-----------------------------------------------------------------
template <class Data_Type>
MatrixDense<Data_Type> MatrixDense<Data_Type>::operator+(const MatrixDense<Data_Type> &B){

  assert((B.nrows == nrows)&&(B.ncols == ncols));  
  
  MatrixDense<Data_Type>  C (nrows,ncols); 
  
  for(int i=0;i < ncols*nrows;i++) 
    C.p[i] = p[i] + B.p[i]; 
       
  return(C); 
}

//-----------------------------------------------------------------
template <class Data_Type>
MatrixDense<Data_Type> MatrixDense<Data_Type>::operator-(const MatrixDense<Data_Type>&B)
{
  assert((B.nrows == nrows)&&(B.ncols == ncols));
  
  MatrixDense<Data_Type>  C (nrows,ncols); 
    
  for(int i = 0;i < ncols*nrows;i++)
         C.p[i] = p[i] - B.p[i]; 
       
  return(C);
}

//-------------------------------------------------------------
template <class Data_Type>
MatrixDense<Data_Type> MatrixDense<Data_Type>::operator*(Data_Type alpha){
  
  MatrixDense<Data_Type>  C(nrows,ncols);

 
  for(int i = 0;i < ncols*nrows;i++) 
    C.p[i] = alpha*p[i]; 
  
  return (C);
}

//-----------------------------------------------------------------
template <class Data_Type>
MatrixDense<Data_Type> operator*(Data_Type alpha, const MatrixDense<Data_Type>  &B){
  
  MatrixDense<Data_Type>  C(B.nrows,B.ncols);

  for(int i = 0;i < B.ncols*B.nrows;i++)
    C.p[i] = alpha*B.p[i]; 
  
  return (C);

} 

//---------------------------------------------------------------------
template <class Data_Type>
MatrixDense<Data_Type> MatrixDense<Data_Type> ::operator*(const MatrixDense<Data_Type> &B){

  assert(ncols == B.nrows);
 
  MatrixDense<Data_Type> C(nrows,B.ncols,0.0);  
  
  for(int i=0;i<nrows;i++){
    for(int j=0;j<B.ncols;j++){  
      for(int k=0;k<ncols;k++)
	C.a[j][i] = C.a[j][i] + a[k][i]*B.a[j][k];
    }}
 
  return(C);
}

//---------------------------------------------------------------------
template <class Data_Type>
MatrixDense<Data_Type> MatrixDense<Data_Type> ::transpose(){
 
  MatrixDense<Data_Type> C(ncols,nrows);  
  
  for(int i=0;i<C.nrows;i++){
    for(int j=0;j<C.ncols;j++)  
	C.a[j][i] = a[i][j];
    }
 
  return(C);
}


//---------------------------------------------------------------------
template <class Data_Type>
ZVector<Data_Type> MatrixDense<Data_Type> ::operator*(const ZVector<Data_Type> &b){

  assert(ncols == b.Size);
  
  ZVector<Data_Type> c(nrows,0.0);

  int i;
  
  for (int j=0;j<ncols;j++){
    for(i=0;i<nrows ;i++) 
      c.a[i] = c.a[i] + a[j][i]*b.a[j];
  }

  
  return(c);
}

//---------------------------------------------------------------------
template <class Data_Type>
void MatrixDense<Data_Type>::set(const MatrixDense<Data_Type> &B,
				    int col_begin, int col_end)
{
  assert((B.nrows == nrows)&&(B.ncols == ncols));
  check_cols(col_begin,col_end,B.ncols);

  int start = col_begin*nrows;
  int end =  col_end*nrows;
  
  for(int i = start;i <end;i++)
      p[i] =  B.p[i];
  
}

//---------------------------------------------------------------------
template <class Data_Type>
void MatrixDense<Data_Type>::set(Index IA,Index JA,
				    const MatrixDense<Data_Type> &B,
				    Index IB, Index JB,
				    int col_begin,int col_end)
{
  assert((IA.start()>=0)&&(IA.end()<nrows));
  assert((JA.start()>=0)&&(JA.end()<ncols));
  assert((IB.start()>=0)&&(IB.end()<B.nrows));
  assert((JB.start()>=0)&&(JB.end()<B.ncols));
  assert((IA.length() == IB.length())&&(JA.length() == JB.length()));
  check_cols(col_begin,col_end,JA.length());
  
  for(int i = 0;i <IA.length();i++){
    for(int j = col_begin;j < col_end;j++)     
    a[JA.start()+j][IA.start()+i] =  B.a[JB.start()+j][IB.start()+i];
  }
}

//---------------------------------------------------------------------
template <class Data_Type>
void MatrixDense<Data_Type>::set(const MatrixDense<Data_Type> &B,
				    Index IB, Index JB,
				    int col_begin, int col_end)
{
  assert((IB.start()>=0)&&(IB.end()<B.nrows));
  assert((JB.start()>=0)&&(JB.end()<B.ncols));
  assert((nrows == IB.length())&&(ncols == JB.length()));
  check_cols(col_begin,col_end,ncols);
  
  for(int i = 0;i <nrows;i++){
    for(int j = col_begin;j <col_end;j++)
      a[j][i] =  B.a[JB.start()+j][IB.start()+i];

  }
}

//---------------------------------------------------------------------
template <class Data_Type>
void MatrixDense<Data_Type>::set(Index IA,Index JA,
				    const MatrixDense<Data_Type> &B,
				    int col_begin,int col_end)
{
  assert((IA.start()>=0)&&(IA.end()<nrows));
  assert((JA.start()>=0)&&(JA.end()<ncols));
  assert((IA.length() == B.nrows)&&(JA.length() == B.ncols));
  check_cols(col_begin,col_end,B.ncols);

  for(int i = 0;i <nrows;i++){
    for(int j = col_begin;j < col_end;j++)
      a[JA.start()+j][IA.start()+i] =  B.a[j][i];
  }
}

//---------------------------------------------------------------------
template <class Data_Type>
MatrixDense<Data_Type>  MatrixDense<Data_Type>::operator()(Index I,Index J)
{
  assert((I.start()>=0)&&(I.end()<nrows));
  assert((J.start()>=0)&&(J.end()<ncols));

  MatrixDense<Data_Type> C(I.length(),J.length());

  C.set(*this,I,J);
  
  return(C);    
}

//---------------------------------------------------------------------
template <class Data_Type>
Data_Type& MatrixDense<Data_Type>::operator()(int i, int j){
  
  assert((i>=0)&&(i<nrows)&&(j>=0)&&(j<ncols));

  return(a[j][i]);    

}
//---------------------------------------------------------------------
template <class Data_Type>
ZVector<Data_Type> MatrixDense<Data_Type>::operator()(int i){

  ZVector<Data_Type> V(nrows, a[i]);

  //cout << "in Matrix V = "<<V<<endl;
  
  return(V);    

}
//---------------------------------------------------------------------
template <class Data_Type>
void  MatrixDense<Data_Type>:: sum(const MatrixDense<Data_Type> &A,
				      const MatrixDense<Data_Type> &B,
				      int col_begin, int col_end)
{
  assert((A.nrows == B.nrows)&&(A.ncols == B.ncols));
  assert((A.nrows == nrows)&&(A.ncols == ncols));
  check_cols(col_begin,col_end,A.ncols);

  int start = col_begin*nrows;
  int end =  col_end*nrows;

  for(int i = start;i < end;i++)
    p[i] = A.p[i] + B.p[i]; 
 
} 

//---------------------------------------------------------------------
template <class Data_Type>
void  MatrixDense<Data_Type>:: diff(const MatrixDense<Data_Type> &A,
				       const MatrixDense<Data_Type> &B,
				       int col_begin, int col_end)
{
  assert((A.nrows == B.nrows)&&(A.ncols == B.ncols));
  assert((A.nrows == nrows)&&(A.ncols == ncols));
  check_cols(col_begin,col_end,A.ncols);  
 
  int start = col_begin*nrows;
  int end =  col_end*nrows;

  for(int i = start;i < end;i++)
    p[i] = A.p[i] - B.p[i]; 
 
}

//---------------------------------------------------------------------
template <class Data_Type>
void  MatrixDense<Data_Type>:: prod(const MatrixDense<Data_Type> &A,
				       const MatrixDense<Data_Type> &B,
				       int col_begin, int col_end)
{
  assert((A.ncols == B.nrows)&&(nrows == A.nrows)&&(ncols == B.ncols));
  check_cols(col_begin,col_end,nrows);
 
  int i,j,k;

  for(i=col_begin;i<col_end;i++){
    for(j=0;j<ncols;j++)
      a[j][i] = 0;
  }    
  
  for(i=col_begin;i<col_end;i++){
    for(j=0;j<B.ncols;j++){  
      for(k=0;k<A.ncols;k++)
	a[j][i] = a[j][i] + A.a[k][i]*B.a[j][k];
    }}
 
}

 //-----------------------------------------------------------------
template <class Data_Type>
void  MatrixDense<Data_Type>:: prod(const MatrixDense<Data_Type> &A,
				       Data_Type alpha,int col_begin, int col_end)
{
  assert((ncols == A.ncols)&&(nrows == A.nrows));  
  check_cols(col_begin, col_end,ncols);

  int start = col_begin*nrows;
  int end =  col_end*nrows;
  
 
  for(int i = start;i < end;i++)
    p[i] = alpha*A.p[i]; 
  
}


//---------------------------------------------------------------------
template <class Data_Type>
void  MatrixDense<Data_Type>:: add(const MatrixDense<Data_Type> &B,
				      int col_begin, int col_end)
 
{
  assert((B.nrows == nrows)&&(B.ncols == ncols));  
  check_cols(col_begin,col_end,B.ncols);  
 
  int start = col_begin*nrows;
  int end =  col_end*nrows;
  
 
  for(int i = start;i < end;i++)
    p[i] = p[i] + B.p[i]; 
  
}

//---------------------------------------------------------------------
template <class Data_Type>
void  MatrixDense<Data_Type>:: subtr(const MatrixDense<Data_Type> &B,
					int col_begin, int col_end)
{
  assert((B.nrows == nrows)&&(B.ncols == ncols));  
  check_cols(col_begin,col_end,B.ncols);  
 
  int start = col_begin*nrows;
  int end =  col_end*nrows;
  
  
  for(int i=start;i < end;i++)     
    p[i] = p[i] - B.p[i]; 
 
}

//---------------------------------------------------------------------
template <class Data_Type>
void  MatrixDense<Data_Type>:: mult(const MatrixDense<Data_Type> &B,
				       MatrixDense<Data_Type> &tmp,
				       int col_begin, int col_end)
{
  assert((ncols == B.nrows)&&(tmp.nrows == nrows));
  assert((tmp.ncols == ncols)&&(B.nrows == B.ncols));
  check_cols(col_begin, col_end,nrows);
  
  int i,j,k;
    
  for(i=col_begin;i<col_end;i++){
    for(j=0;j<B.ncols;j++)
      tmp.a[j][i] = 0;
  }    
  
  for(i=col_begin;i<col_end;i++){
    for(j=0;j<B.ncols;j++){  
      for(k=0;k<ncols;k++)
	tmp.a[j][i] = tmp.a[j][i] + a[k][i]*B.a[j][k];
    }}
  
  for(i=col_begin;i<col_end;i++){
    for(j=0;j<B.ncols;j++)
      a[j][i] = tmp.a[j][i];
  }
  
}

//---------------------------------------------------------------------
template <class Data_Type>
void  MatrixDense<Data_Type>::mult(const ZVector<Data_Type>& b,
				      ZVector<Data_Type>& x ,
				      int col_begin, int col_end)
{
  assert((b.Size == ncols)&&(b.Size == x.Size));
  check_cols(col_begin,col_end,ncols);
  
  int i,j;   
    
  
  for (j=col_begin;j<col_end;j++){
    for(i=0;i<nrows;i++)  
      x.a[i] = x.a[i] + a[j][i]*b.a[j];
  }

}

//-----------------------------------------------------------------
template <class Data_Type>
void  MatrixDense<Data_Type>:: mult(Data_Type alpha,int col_begin,
				       int col_end)
{
  check_cols(col_begin,col_end,ncols);
   
  int start = col_begin*nrows;
  int end =  col_end*nrows;

  for(int i = start;i < end;i++)  
    p[i] = alpha*p[i]; 
  
}

//---------------------------------------------------------------------
template <class Data_Type>
ostream &operator<< (ostream &output,MatrixDense<Data_Type> &A){
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
void MatrixDense<Data_Type>::read(char * filename){

  int i;
  char matrix[80];
  char matrix_type[80];
    
  ifstream file_in(filename);

  if(!file_in){
     cerr << "Error: Can not open file "<<filename<<"!"<<endl;
    exit(0);
  }

  file_in >> matrix;
  if(strcmp(matrix,"matrix")){
    cerr << "Error: Not a MATRIX in "<<filename<<"!"<<endl;
    exit(0);
  }
  
  file_in >> matrix_type;
  if(strcmp(matrix_type,"dense")){
    cerr << "Error: Not a DENSE matrix in "<<filename<<"!"<<endl;
    exit(0);
  }
  
  
  
  file_in >> nrows;
  file_in >> ncols;
  
  a = new Data_Type*[ncols];
  
  Data_Type*  mp = new Data_Type[nrows*ncols];
  for(int j=0;j<ncols;j++){
    a[j]=mp;
    mp+=nrows;
  }
  
 p = a[0];

//  for(i=0;i<nrows*ncols;i++)
//    file_in >> p[i];
// this is coloumnwise


// to read matrix as a table 

  for(i=0;i<nrows;i++){
    for(j=0;j<ncols;j++){
      file_in >> a[j][i];
    }}

  

 file_in.close(); 
  
}
  
//---------------------------------------------------------------------
template <class Data_Type>
void MatrixDense<Data_Type>::write(char * filename){

  int i;
  ifstream file_out(filename);

  if(!file_out){
     cerr << "Can not open file "<<filename<<endl;
    exit(0);
  }

  file_out <<"matrix ";
  file_out<<"dense"<<endl;
  file_out<<endl;
  file_out << nrows<<" ";
  file_out << ncols<<endl;
  file_out <<endl;
  
  for(i=0;i<nrows;i++){
    for(j=0;j<ncols;j++)
    file_out << a[j][i]<<" ";
     file_out << endl;
  }

 file_out.close();
  
}
template <class Data_Type>
void MatrixDense<Data_Type>::info() {
    cerr << "MatrixDense :: info not implemented for generics.\n";
}
#endif














/****************************************************************
 *  Class Vector.h                                              *
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

#ifndef ZVECTOR_H
#define ZVECTOR_H 1

#include <iostream>
#include <fstream>
#include <assert.h>

#include "Index.h"
#include "utils.h"
#include "Complex.h"

using namespace std;

template<class Data_Type> class MatrixDense;
template<class Data_Type> class MatrixSparse;
template<class Data_Type> class MatrixTridiag;

template <class Data_Type>
class ZVector{

  friend class MatrixDense<Data_Type>;
  friend class MatrixSparse<Data_Type>;
  friend class MatrixTridiag<Data_Type>;
  
private:
  Data_Type *a;
  int Size;
  
public:
//constructors:
  inline ZVector():Size(0),a(0){}
  ZVector (int N); 
  ZVector (int N,Data_Type x);
  ZVector (int N,Data_Type x[]);
  ZVector (int N,double x_re[],double x_im[]);
  ZVector (const ZVector &b);
  
//destructor:
  ~ZVector(); 
  
  int size() const {return Size;};
  Data_Type &operator()(int i);
  ZVector operator()(Index I);
  inline Data_Type* get_p(){return a;};
  
//"In place, modifying the vector" operations: 
  void add(const ZVector& B,int row_begin=0,int row_end=0);
  void subtr(const ZVector& B,int row_begin=0,int row_end=0);
  void mult(const Data_Type alpha,int row_begin=0,int row_end=0);

  
//"Put results  in the provided space" operations:     
  void sum(const ZVector& A,const ZVector& B,int row_begin=0,int row_end=0);
  void diff(const ZVector& A,const ZVector& B,int row_begin=0,int row_end=0);
  void prod(const ZVector& A,const Data_Type alpha,int row_begin=0,int row_end=0);
  void set(const ZVector& B,int row_begin=0,int row_end=0);
  void set(Index IA,const ZVector &B,Index IB,int row_begin=0,int row_end=0);
  void set(const ZVector &B,Index IB,int row_begin=0,int row_end=0);   
  void set(Index IA,const ZVector &B,int row_begin=0,int row_end=0);

    
//"Using new " operations:
  ZVector &operator=(const ZVector& B); 
  ZVector &operator=(double x); 
  ZVector &operator=(Complex x); 
  ZVector operator+(const ZVector& B);
  ZVector operator-(const ZVector& B);  
  Data_Type operator*(const ZVector& B);
  ZVector  operator*(Data_Type x);
  friend ZVector  operator*(Data_Type x, const ZVector &B);
 

//Input & Output:
  void read(char* filename);
  void write(char* filename);
  friend ostream &operator<< (ostream &output, ZVector<Data_Type> &B);
  void info();
  
};



//-----------------------------------------------------------------
template <class Data_Type>
ZVector<Data_Type>::ZVector(int N){
  Size = N;
  
  a = new Data_Type [N];
}

//-----------------------------------------------------------------
template <class Data_Type>
ZVector<Data_Type>::ZVector(int N,Data_Type x){
  Size = N;
  
  a = new Data_Type [N];
  
  for(int i=0;i<N;i++)
    a[i] = x;
}

//-----------------------------------------------------------------
template <class Data_Type>
ZVector<Data_Type>::ZVector(int N,Data_Type x[]){
  Size = N;
  
  a = new Data_Type [N];
  
  for(int i=0;i<N;i++)
    a[i] = x[i];
}

template <class Data_Type> 
ZVector<Data_Type>::ZVector(int N,double x_re[],double x_im[]){
    cerr << "Generic ZVector(int N,double x_re[],double x_im[]) constructor not implemented.\n";
}

//----------------------------------------------------------------
template <class Data_Type>
ZVector<Data_Type>::ZVector(const ZVector<Data_Type> &b){

//  cout << "Copy constructor was called"<<endl;
  
  Size = b.Size;
  a = new Data_Type [Size];
  
  for(int i=0;i<Size;i++)
    a[i] = b.a[i];
  
}

//---------------------------------------------------------------------
template <class Data_Type>
ZVector<Data_Type>::~ZVector(){
//  cout << "Calling ZVector destructor!"<<endl;  
  if(a)
    delete [] a;
}

//-----------------------------------------------------------------
template <class Data_Type>
Data_Type& ZVector<Data_Type>::operator()(int i){
  
  assert((i>=0)&&(i<Size));
  
  return(a[i]); 
}

//-----------------------------------------------------------------
template <class Data_Type>
ZVector<Data_Type> &ZVector<Data_Type>::operator=(const ZVector<Data_Type> &B){


// cout << "Operator = was called!" <<endl;
  
  Size = B.Size;
  
  if(a)
    delete [] a;
  
  a = new Data_Type [Size];
  
  for(int i=0;i<Size;i++)
    a[i] = B.a[i];
  
  return *this;
}
//-----------------------------------------------------------------
template <class Data_Type>
ZVector<Data_Type> &ZVector<Data_Type>::operator=(double x){

  for(int i=0;i<Size;i++)
    a[i] = x;
  
  return *this;
}

template <class Data_Type>
ZVector<Data_Type> &ZVector<Data_Type>::operator=(Complex x){

  for(int i=0;i<Size;i++)
    a[i] = x;
  
  return *this;
}

//-----------------------------------------------------------------
template <class Data_Type>
ZVector<Data_Type> ZVector<Data_Type>::operator+(const ZVector<Data_Type> &B){
  
  assert(Size == B.Size);
  
  ZVector<Data_Type> C(Size);
  
  for(int i=0;i<Size;i++)
    C.a[i] = a[i] + B.a[i];
  
  return(C);
  
}

//-----------------------------------------------------------------
template <class Data_Type>
ZVector<Data_Type> ZVector<Data_Type>::operator-(const ZVector<Data_Type> &B){
  
  assert( Size == B.Size);
  
  ZVector<Data_Type> C(Size);
  
  for(int i=0;i<Size;i++)
    C.a[i] = a[i] - B.a[i];
  
  return(C);
  
}

//-----------------------------------------------------------------
template <class Data_Type>
Data_Type ZVector<Data_Type> ::operator*(const ZVector<Data_Type> &B){
  Data_Type S;
  S = 0;
  
  for(int i=0;i<B.Size;i++)
    S = S + a[i]*B.a[i];
  
  return (S);
}

//-----------------------------------------------------------------
template <class Data_Type>
ZVector<Data_Type> ZVector<Data_Type>::operator*(Data_Type x){
  
  ZVector<Data_Type>  C(Size);
  
  for(int i=0;i<Size;i++)
    C.a[i] = x*a[i];
  
  return (C);
}

//-----------------------------------------------------------------
template<class Data_Type>
ZVector<Data_Type> operator*(Data_Type x, const ZVector<Data_Type> &B){
  
  ZVector<Data_Type>  C(B.Size);
  
  for(int i=0;i<B.Size;i++)
    C.a[i] = x*B.a[i];
  
  return (C);
}

//-----------------------------------------------------------------
template <class Data_Type>
void ZVector<Data_Type>::set(const ZVector<Data_Type>& B,int row_begin,int row_end){
  
  assert(Size == B.Size);
  check_rows(row_begin,row_end,Size); 
  
  for(int i=row_begin;i<row_end;i++)
    a[i] = B.a[i];    
  
}
//---------------------------------------------------------------------
template <class Data_Type>
void ZVector<Data_Type>::set(Index Ia,const ZVector<Data_Type> &b,Index Ib,
			    int row_begin, int row_end)
{
  assert((Ia.start()>=0)&&(Ia.end()<Size));
  assert((Ib.start()>=0)&&(Ib.end()<b.Size));
  assert(Ia.length() == Ib.length());
  check_rows(row_begin,row_end,IA.length());
  
  for(int i = row_begin;i <row_end;i++)
      a[Ia.start()+i] = b.a[Ib.start()+i];
  
}

//---------------------------------------------------------------------
template <class Data_Type>
void ZVector<Data_Type>::set(const ZVector<Data_Type> &b,Index Ib,
			    int row_begin, int row_end)
{
  assert((Ib.start()>=0)&&(Ib.end()<b.Size));
  assert(Size == Ib.length());
  check_rows(row_begin,row_end,Size);
  
  for(int i = row_begin;i <row_end;i++)
      a[i] =  b.a[Ib.start()+i];
  
}

//---------------------------------------------------------------------
template <class Data_Type>
void ZVector<Data_Type>::set(Index Ia,const ZVector<Data_Type> &b,
			    int row_begin, int row_end)
{
  assert((Ia.start()>=0)&&(Ia.end()<nrows));  
  assert(Ia.length() == b.Size);
  check_rows(row_begin,row_end,b.Size);

  for(int i = row_begin;i <row_end;i++)
      a[Ia.start()+i] =  b.a[i];
  
}

//---------------------------------------------------------------------
template <class Data_Type>
ZVector<Data_Type>  ZVector<Data_Type>::operator()(Index I)
{
  assert((I.start()>=0)&&(I.end()<Size));

  ZVector<Data_Type> c(I.length());

  c.set(*this,I);
  
  return(c);    
}


//-----------------------------------------------------------------
template <class Data_Type>
void ZVector<Data_Type>::add(const ZVector<Data_Type>& B,int row_begin,int row_end){

  assert(Size == B.Size);
  check_rows(row_begin,row_end,Size);
 
  for(int i=row_begin;i<row_end;i++)
    a[i] = a[i] + B.a[i];    
}

//-----------------------------------------------------------------
template <class Data_Type>
void ZVector<Data_Type>::subtr(const ZVector<Data_Type>& B,int row_begin,int row_end){

  assert(Size == B.Size);
  check_rows(row_begin,row_end,Size);
   
  for(int i=row_begin;i<row_end;i++)
    a[i] = a[i] - B.a[i];    
}

//-----------------------------------------------------------------
template <class Data_Type>
void ZVector<Data_Type>::mult(const Data_Type alpha,int row_begin,int row_end){
  
  check_rows(row_begin,row_end,Size);
  
  for(int i=row_begin;i<row_end;i++)
    a[i] = alpha*a[i];    
}

//-----------------------------------------------------------------
template <class Data_Type>
void ZVector<Data_Type>::sum(const ZVector<Data_Type>& A,
			    const ZVector<Data_Type>& B,
			    int row_begin,int row_end)
{
  assert((A.size()==B.size())&&(A.size()==Size));
  check_rows(row_begin,row_end,Size);
  
  for(int i=row_begin;i<row_end;i++)
    a[i] = A.a[i] + B.a[i];    
  
  return;   
}

//-----------------------------------------------------------------
template <class Data_Type>
void ZVector<Data_Type>::prod(const ZVector<Data_Type>& A,
			     const Data_Type alpha,
			     int row_begin,int row_end)
{
  assert(Size==A.Size);
  check_rows(row_begin,row_end,Size);
  
  for(int i=row_begin;i<row_end;i++)
    a[i] = alpha*A.a[i];    
  
  return;
}

//-----------------------------------------------------------------
template <class Data_Type>
void ZVector<Data_Type>::diff(const ZVector<Data_Type>& A,
			    const ZVector<Data_Type>& B,
			     int row_begin,int row_end)
{
  assert((A.Size == B.Size)&&(A.Size == Size));
  check_rows(row_begin,row_end,Size);
  
  for(int i=row_begin;i<row_end;i++)
    a[i] = A.a[i] - B.a[i];    
  
return;
}

//-----------------------------------------------------------------
template<class Data_Type>
ostream &operator<< (ostream &output, ZVector<Data_Type>  &b){

  output<<endl;
  for(int i=0 ;i < b.size();i++)
    output<<"["<<b.a[i]<<"]"<<endl;
  output<<endl;
  
  return(output);
}

//-----------------------------------------------------------------
template <class Data_Type>
void ZVector<Data_Type>::read(char * filename){

  int i;
  char ZVector[80];
  ifstream file_in(filename);

  if(!file_in){
     cerr << "Can not open file "<<filename<<endl;
    exit(0);
  }

  file_in >> ZVector;
  if(strcmp(ZVector,"vector")){
    cerr << "Error: Not a VECTOR in "<<filename<<"!"<<endl;
    exit(0);
  }
 
  
  file_in >> Size;

  a = new Data_Type [Size];
  
  for(i=0;i<Size;i++)
     file_in >> a[i];

  file_in.close();
}

//-----------------------------------------------------------------
template <class Data_Type>
void ZVector<Data_Type>::write(char * filename){

  int i;
  ifstream file_out(filename);

  if(!file_out){
     cerr << "Can not open file "<<filename<<endl;
    exit(0);
  }

  file_out<<"vector"<<endl;
  file_out<<endl;
  file_out << Size<<endl;
  filr_out << endl;

  for(i=0;i<Size;i++)
     file_out << a[i];

  file_out.close();
}

template <class Data_Type>
void ZVector<Data_Type>::info() {
    cerr << "ZVector<>::info() not implemented for generic.\n";
}
#endif

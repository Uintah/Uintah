
/*
 *  cVector.cc : ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Core/Datatypes/cVector.h>
#include <fstream>
#include <iostream>
using std::complex;
using std::cerr;
using std::abs;
using std::conj;
using std::endl;

namespace SCIRun {

// Dd: Should this be here?
PersistentTypeID cVector::type_id("cVector", "Datatype", 0);

//-----------------------------------------------------------------
cVector::cVector(int N){
 int i;
 Size = N;

 a = new Complex [N];

 for(i=0;i<N;i++)
          a[i] = 0;
}

//----------------------------------------------------------------
cVector::~cVector(){

 delete [] a;
}

//----------------------------------------------------------------
cVector::cVector(const cVector &B){
 int i;
 int N = B.Size;

 a = new Complex [N];

 Size = B.Size;

 for(i=0;i<N;i++)
    a[i] = B.a[i];
}

//----------------------------------------------------------------
cVector &cVector::operator=(const cVector &B){
 int i;
 int N = B.Size;
  
 delete [] a;

 a = new Complex [N];

 Size = B.Size;

 for(i=0;i<N;i++)
  a[i] = B.a[i];

 return *this;
}

//-----------------------------------------------------------------
cVector::Complex& cVector::operator()(int i){

 if((i>=0)&&(i<Size)){
  return(a[i]);
  }

 else 
    cerr <<"Error: cVector index is out of range!"<<endl;
 return(a[0]);  
}

//-----------------------------------------------------------------
cVector  cVector:: operator+(const cVector& B) const{
 cVector C(Size);

 for(int i = 0;i < Size;i++)
        C.a[i] = a[i] + B.a[i];

 return (C);
}

//-----------------------------------------------------------------
cVector cVector:: operator-(const cVector& B) const{
 cVector C(Size);

 for(int i = 0;i < Size;i++)
         C.a[i] = a[i] - B.a[i];
 
 return (C);
}
//-----------------------------------------------------------------
double cVector::norm(){

double norm2=0;

 for(int i = 0;i < Size;i++)
       
         norm2 = norm2 + abs(a[i])*abs(a[i]);
 
 return (sqrt(norm2));
}

//-----------------------------------------------------------------
int cVector::load(char* filename){

    std::ifstream file_in(filename);

 if(!file_in){
      cerr<<"Error:Cannot open input file:" <<filename<<endl;
  return(-1);
}

 for(int i = 0;i < Size;i++)
       file_in>>a[i];


 file_in.close();
 return(0); 
}

//-----------------------------------------------------------------
void cVector::set(const cVector& B){
  for(int i=0;i<Size;i++)
    a[i] = B.a[i];    
}

//-----------------------------------------------------------------
void cVector::add(const cVector& B){
  for(int i=0;i<Size;i++)
    a[i] = a[i] + B.a[i];    
}

//-----------------------------------------------------------------
void cVector::subtr(const cVector& B){
  for(int i=0;i<Size;i++)
    a[i] = a[i] - B.a[i];    
}

//-----------------------------------------------------------------
void cVector::mult(const Complex x){
  for(int i=0;i<Size;i++)
    a[i] = x*a[i];    
}

//-----------------------------------------------------------------
void cVector::conj(){
  for(int i=0;i<Size;i++)
    a[i] = std::conj(a[i]);    
}

//------------------------------------------------------------------
cVector::Complex operator* (cVector& A, cVector& B){
cVector::Complex S;
  S = 0;

 for(int i=0;i<A.Size;i++)
     S = S + conj(A.a[i])*B.a[i];

 return (S);
}

//-----------------------------------------------------------------
std::ostream &operator<< (std::ostream &output, cVector  &A){

 output<<"[";

 for(int i=0 ;i < A.size();i++)
     output<<A.a[i]<<" ";

 output<<"]";
 output<<endl;

 return(output);
}

//-----------------------------------------------------------------
cVector operator*(cVector::Complex x,const cVector &B){
 cVector  C(B.Size);

 for(int i=0;i<B.Size;i++)
  C.a[i] = x*B.a[i];

 return (C);
}

//-----------------------------------------------------------------
cVector operator*(const cVector &B,cVector::Complex x){
 cVector  C(B.Size);

 for(int i=0;i<B.Size;i++)
  C.a[i]=x*B.a[i];

 return (C);
}
//-----------------------------------------------------------------
cVector operator*(double x,const cVector &B){
 cVector  C(B.Size);

 for(int i=0;i<B.Size;i++)
  C.a[i] = x*B.a[i];

 return (C);
}

//-----------------------------------------------------------------
cVector operator*(const cVector &B,double x){
 cVector  C(B.Size);

 for(int i=0;i<B.Size;i++)
  C.a[i]=x*B.a[i];

 return (C);
}

//-----------------------------------------------------------------

void cVector::io(Piostream&) {
  cerr << "cVector::io not finished\n";
}

} // End namespace SCIRun


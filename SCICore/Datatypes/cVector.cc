//static char *id="@(#) $Id$";

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

#include <SCICore/Datatypes/cVector.h>
#include <fstream>
using std::complex;
#ifndef _WIN32
using std::cerr;
#else
#include <iostream.h>
#endif
using std::abs;
using std::conj;
using std::endl;

namespace SCICore {
namespace Datatypes {

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

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.6  1999/10/26 21:52:05  moulding
// added a #ifndef block for win32.  cerr isn't in visual c++'s std namespace yet.
//
// Revision 1.5  1999/10/07 02:07:36  sparker
// use standard iostreams and complex type
//
// Revision 1.4  1999/09/04 06:01:46  sparker
// Updates to .h files, to minimize #includes
// removed .icc files (yeah!)
//
// Revision 1.3  1999/08/25 03:48:48  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:39:02  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:34  mcq
// Initial commit
//
// Revision 1.1  1999/04/25 04:07:25  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

//static char *id="@(#) $Id$";

/*
 *  cSMatrix.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <SCICore/Datatypes/cSMatrix.h>
#include <iostream.h>

namespace SCICore {
namespace Datatypes {

//-----------------------------------------------------------------
cSMatrix::cSMatrix(int lnrows, int lncols,int lnnz,Complex *la, int * lrow, int *lcol ){
  
 nrows = lnrows;
 ncols = lncols;
 nnz = lnnz;
  
 a = new Complex [nnz];
 col = new int [nnz];
 row_ptr = new int [nrows+1];

  
 int i;
 for(i=0;i<nnz;i++){
   a[i] =la[i];
   col[i] = lcol[i];
 }

for(i=0;i<nrows+1;i++)
   row_ptr[i] =lrow[i];
 
}
//----------------------------------------------------------------
cSMatrix::~cSMatrix(){

 delete[] a;

 delete[] row_ptr;

 delete[] col;
}

//----------------------------------------------------------------

ostream &operator<< (ostream &output, cSMatrix &A){

 cout<<"row_ptr: ["; 
 int i;
 for(i=0 ;i < A.nrows;i++)
     cout << A.row_ptr[i]<<" ";
       cout<<" ]"<<endl;

cout<<"col: ["; 
 for(i=0 ;i < A.nnz;i++)
     cout << A.col[i]<<" ";
       cout<<" ]"<<endl;

cout<<"val: ["; 
 for(i=0 ;i < A.nnz;i++)
     cout << A.a[i]<<" ";
       cout<<" ]"<<endl;
 
 
 return(output);
}

//----------------------------------------------------------------
cVector cSMatrix::operator*(cVector &V){
 cVector V1(V.Size);
 int i,j;
 int ind_l,ind_h;
 
 for(i=0;i<nrows;i++){
    ind_l = row_ptr[i];
    ind_h = row_ptr[i+1];
    
 for(j=ind_l;j<ind_h;j++) 
       V1.a[i] = V1.a[i] + a[j]*V.a[col[j]]; 
    
 }

 return V1;
}

//-----------------------------------------------------------------

void cSMatrix::mult(cVector& V,cVector& tmp){
 int i,j;
 int ind_l,ind_h;
  
  for(i=0;i<V.Size;i++)
    tmp.a[i] = 0;
  
 for(i=0;i<nrows;i++){
    ind_l = row_ptr[i];
    ind_h = row_ptr[i+1];
    
 for(j=ind_l;j<ind_h;j++) 
       tmp.a[i] = tmp.a[i] + a[j]*V.a[col[j]]; 
    
 }

  
  for(i=0;i<V.Size;i++)
    V.a[i] = tmp.a[i];
  
}
//------------------------------------------------------------------

Complex& cSMatrix::get(int i, int j) {
    int row_idx=row_ptr[i];
    int next_idx=row_ptr[i+1];
    int l=row_idx;
    int h=next_idx-1;
    for(;;){
	if(h<l){
	    static Complex zero;
	    zero=Complex(0,0);
	    return zero;
	}
	int m=(l+h)/2;
	if(j<col[m]){
	    h=m-1;
	} else if(j>col[m]){
	    l=m+1;
	} else {
	    return a[m];
	}
    }
}

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/09/08 02:26:49  sparker
// Various #include cleanups
//
// Revision 1.3  1999/08/25 03:48:48  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:39:01  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:34  mcq
// Initial commit
//
// Revision 1.1  1999/04/25 04:07:24  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//

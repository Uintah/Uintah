#include <iostream>
#include "TopRealSurface.h"

using std::cout;
using std::endl;

TopRealSurface::TopRealSurface(){
}


// TopRealSurface::TopRealSurface(int _surfaceIndex, int _TopBottomNo)
//   : RealSurface ( _surfaceIndex ){
//   surfaceIndex = _surfaceIndex;
//   TopBottomNo = _TopBottomNo;  
// }

// n top -- n = 0 i + 0 j + -1 k 
void TopRealSurface::setData(int _surfaceIndex, int _TopBottomNo){
  surfaceIndex = _surfaceIndex;
  TopBottomNo = _TopBottomNo;
}

 void
 TopRealSurface::get_n(){
   n[0] = 0;
   n[1] = 0;
   n[2] = -1;
 }

// t1 top -- t1 = 1 i + 0 j + 0 k

 void
 TopRealSurface::get_t1(){
   t1[0] = 1;
   t1[1] = 0;
   t1[2] = 0;
 }



// t2 top -- t2 = 0 i + -1 j + 0 k
 void
 TopRealSurface::get_t2(){
   t2[0] = 0;
   t2[1] = -1;
   t2[2] = 0;
 }


// get private normal vector n, for class ray
void
TopRealSurface::set_n(double *nn){
  for ( int i = 0; i < 3; i ++ )
    nn[i] = n[i];
}

// known surfaceIndex, to get vIndex and limits of the surface
void
TopRealSurface::get_limits(double *VolTable, int &vIndex){
 int index;
 double *p;

      int VolTopNo = TopBottomNo - 1;
      p = VolTable;
      index = ( int ) * ( p+7 ); // table's topsurface element index
      VolIndex = ( int ) * ( p+6 );
      
      if ( index == surfaceIndex ) {
	yup = * ( p+2 );
	ylow = * ( p+3 );
	xlow = * ( p+4 );
	xup = * ( p+5 );
      }
      else {
	do 
	  {
	    p+=13; // next row
	    index = ( int ) * ( p+7 );
	  }while( (index != surfaceIndex) && (VolIndex <= VolTopNo) );

	if ( ! ( VolIndex <= VolTopNo ) ) {
	  std::cerr << " Error looking up topsurface table" ;
	  exit(1);
	}
	else { // found the index, go and find the boundary x, y
	  yup = * ( p+2 );
	  ylow = * ( p+3 );
	  xlow = * ( p+4 );
	  xup = * ( p+5 );
	  VolIndex = ( int ) * ( p+6 );
	}
      } // end second if else
      
      ztop = * p;

      // from derived class TopRealSurface's private data to Base Class
      // RealSurface's protected data members

      alow = xlow;
      aup = xup;
      blow = ylow;
      bup = yup;
      constv = ztop;
      vIndex = VolIndex;
            
}

  
TopRealSurface::~TopRealSurface(){
}

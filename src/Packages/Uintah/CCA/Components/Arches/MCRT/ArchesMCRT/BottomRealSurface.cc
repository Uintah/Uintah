B#include "BottomRealSurface.h"
#include <iostream>

using std::cout;
using std::endl;

BottomRealSurface::BottomRealSurface(){
}

void BottomRealSurface::setData(int _surfaceIndex,
				int _TopBottomNo,
				int _VolElementNo){
  surfaceIndex = _surfaceIndex;
  TopBottomNo = _TopBottomNo;
  VolElementNo = _VolElementNo;
}

// n top -- n = 0 i + 0 j + 1 k  
void
BottomRealSurface::get_n(){
  n[0] = 0;
  n[1] = 0;
  n[2] = 1;
}


// t1 top -- t1 = 1 i + 0 j + 0 k
void
BottomRealSurface::get_t1(){
  t1[0] = 1;
  t1[1] = 0;
  t1[2] = 0;
}



// t2 top -- t2 = 0 i + 1 j + 0 k
void
BottomRealSurface::get_t2(){
  t2[0] = 0;
  t2[1] = 1;
  t2[2] = 0;
}


// get private normal vector n, for class ray
void
BottomRealSurface::set_n(double *nn){
  for ( int i = 0; i < 3; i ++ )
    nn[i] = n[i];
}

void
BottomRealSurface::get_limits(double *VolTable, int &vIndex){
  int index;
  double *p;
 
      int StartPosition = VolElementNo - TopBottomNo;
      p = VolTable;
      p += StartPosition * 13;
      index = ( int ) * ( p+8 ); // table's bottomsurface element index
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
	    index = ( int ) * ( p+8 );
	    
	  }while( (index != surfaceIndex) ); // && (VolIndex <= VolTopNo) );

	  yup = * ( p+2 );
	  ylow = * ( p+3 );
	  xlow = * ( p+4 );
	  xup = * ( p+5 );
	  VolIndex = ( int ) * ( p+6 );
	  
      } // end second if else      
    
      zbottom = * ( p + 1);

      alow = xlow;
      aup = xup;
      blow = ylow;
      bup = yup;
      constv = zbottom;
      vIndex = VolIndex;
      
//       cout << "xlow = " << xlow << "; xup = " << xup <<  endl;
//       cout << " ylow = " << ylow << " ; yup = " << yup << endl;
//       cout << " VolIndex = " << VolIndex << endl;
//       cout << " zbottom = " << zbottom << endl;
}


BottomRealSurface::~BottomRealSurface(){
}

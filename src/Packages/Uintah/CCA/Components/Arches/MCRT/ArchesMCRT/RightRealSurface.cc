#include "RightRealSurface.h"
#include <cstdlib>
#include <iostream>

using std::cout;
using std::endl;

// RightRealSurface::RightRealSurface(int _surfaceIndex, int _xno)
//   : RealSurface ( _surfaceIndex ){
// }

RightRealSurface::RightRealSurface(){
}

void
RightRealSurface::setData(int _surfaceIndex, int _xno){
  surfaceIndex = _surfaceIndex;
  xno = _xno;
}

// n top -- n = -1 i + 0 j + 0 k  
void
RightRealSurface::get_n(){
  n[0] = -1;
  n[1] = 0;
  n[2] = 0;
}


// t1 top -- t1 = 0 i + -1 j + 0 k
void
RightRealSurface::get_t1(){
  t1[0] = 0;
  t1[1] = -1;
  t1[2] = 0;
}



// t2 top -- t2 = 0 i + 0 j + 1 k
void
RightRealSurface::get_t2(){
  t2[0] = 0;
  t2[1] = 0;
  t2[2] = 1;
}


void
RightRealSurface::set_n(double *nn){
  for ( int i = 0; i < 3; i ++ )
    nn[i] = n[i];
}


void
RightRealSurface::get_limits(double *VolTable, int &vIndex){
  int index, offset;
  double *p;
  offset = xno * 13;

      p = VolTable;
      p += offset - 13; // it is special for the first one
      index = ( int ) * ( p+12 ); // table's back element index
      VolIndex = ( int ) * ( p+6 );
      
      if ( index == surfaceIndex ) {
	zup = * p;
	zlow = * ( p+1 );	
	yup = * ( p+2 );
	ylow = * ( p+3 );
      }
      else {
	do {
	  p += offset; // jump to next row on left surface
	  index = ( int ) * ( p+12 );
	} while ( index != surfaceIndex );
	zup = * p;
	zlow = * ( p+1 );
	yup = * ( p+2 );
	ylow = * ( p+3 );
	VolIndex = ( int ) * ( p+6 );
      } // end if( index == surfaceIndex ) else
      
      xright = * ( p + 5 );

      alow = ylow;
      aup = yup;
      blow = zlow;
      bup = zup;
      constv = xright;
      vIndex = VolIndex;
      
//       cout << " ylow = " << ylow << "; yup = " << yup << endl;
//       cout << " zlow = " << zlow << "; zup = " << zup << endl;
//       cout << " VolIndex = " << VolIndex << endl;
//       cout << " xright = " << xright << endl;
      
}


RightRealSurface::~RightRealSurface(){
}

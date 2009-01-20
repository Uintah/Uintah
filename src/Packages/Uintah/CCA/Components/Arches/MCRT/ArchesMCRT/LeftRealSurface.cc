#include "LeftRealSurface.h"
#include <iostream>
#include <cstdlib>

using std::cout;
using std::endl;

// LeftRealSurface::LeftRealSurface(int _surfaceIndex, int _xno)
//   : RealSurface ( _surfaceIndex ){
// }

LeftRealSurface::LeftRealSurface(){
}

void
LeftRealSurface::setData(int _surfaceIndex, int _xno){
  surfaceIndex = _surfaceIndex;
  xno = _xno;
}

// n top -- n = 1 i + 0 j + 0 k  
void
LeftRealSurface::get_n(){
  n[0] = 1;
  n[1] = 0;
  n[2] = 0;
}


// t1 top -- t1 = 0 i + 1 j + 0 k
void
LeftRealSurface::get_t1(){
  t1[0] = 0;
  t1[1] = 1;
  t1[2] = 0;
}



// t2 top -- t2 = 0 i + 0 j + 1 k
void
LeftRealSurface::get_t2(){
  t2[0] = 0;
  t2[1] = 0;
  t2[2] = 1;
}


void
LeftRealSurface::set_n(double *nn){
  for ( int i = 0; i < 3; i ++ )
    nn[i] = n[i];
}


// VolIndex is the private data member of LeftRealSurface
// alow, aup, blow, bup, constv are RealSurface's protected data member

void
LeftRealSurface::get_limits(double *VolTable, int &vIndex){
			    
  int index, offset;
  double *p;
  offset = xno * 13;
 
      p = VolTable;
      index = ( int ) * ( p+11 ); // table's left element index
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
	  index = ( int ) * ( p+11 );
	  //	  cout << " index = " << index << endl;
	} while ( index != surfaceIndex );

	// add in check if till the end, then error
	
	zup = * p;
	zlow = * ( p+1 );
	yup = * ( p+2 );
	ylow = * ( p+3 );
	VolIndex = ( int ) * ( p+6 );
      } // end if( index == surfaceIndex ) else
      
      xleft = * ( p + 4);

      // set LeftRealSurface's private data to RealSurface's protected data
      alow = ylow;
      aup = yup;
      blow = zlow;
      bup = zup;
      constv = xleft;
      vIndex = VolIndex;
      
//       cout << " ylow = " << ylow << "; yup = " << yup << endl;
//       cout << " zlow = " << zlow << "; zup = " << zup << endl;
//       cout << " VolIndex = " << VolIndex << endl;
//       cout << " xleft = " << xleft << endl;
      
}


LeftRealSurface::~LeftRealSurface(){
}

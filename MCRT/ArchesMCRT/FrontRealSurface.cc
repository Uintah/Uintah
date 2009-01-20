#include "FrontRealSurface.h"
#include "Consts.h"
#include <iostream>

using std::cout;
using std::endl;

FrontRealSurface::FrontRealSurface(){
}

// FrontRealSurface::FrontRealSurface(int _surfaceIndex,
// 				   int _TopBottomNo,int _xno)
//   : RealSurface ( _surfaceIndex ){
// }

void
FrontRealSurface::setData(int _surfaceIndex, int _TopBottomNo, int _xno){
  surfaceIndex = _surfaceIndex;
  TopBottomNo = _TopBottomNo;
  xno = _xno;
}

// n top -- n = 0 i + 1 j + 0 k  
void
FrontRealSurface::get_n(){
  n[0] = 0;
  n[1] = 1;
  n[2] = 0;
}


// t1 top -- t1 = 1 i + 0 j + 0 k
void
FrontRealSurface::get_t1(){
  t1[0] = 1;
  t1[1] = 0;
  t1[2] = 0;
}



// t2 top -- t2 = 0 i + 0 j + -1 k
void
FrontRealSurface::get_t2(){
  t2[0] = 0;
  t2[1] = 0;
  t2[2] = -1;
}


// get private normal vector n, for class ray
void
FrontRealSurface::set_n(double *nn){
  for ( int i = 0; i < 3; i ++ )
    nn[i] = n[i];
}


void
FrontRealSurface::get_limits(double *VolTable, int &vIndex){
  
  int index, counter, offset;
  double *p;
  offset = (TopBottomNo - xno) * 13;
  // std::cout << " Cube line 112 " << std::endl;  

      p = VolTable;
      p +=  offset;
      index = ( int ) * ( p+9 ); // table's front element index
      VolIndex = ( int ) * ( p+6 );
      
      // std::cout << " Cube line 119 " << std::endl;      
      if ( index == surfaceIndex ) {
	zup = * p;
	zlow = * ( p+1 );
	xlow = * ( p+4 );
	xup = * ( p+5 );
      }
      else {
	//  std::cout << " Cube line 127 " << std::endl;	
	do {
	  counter = 0;
	  do 
	    {
	      //    std::cout << " Cube line 132 " << std::endl;  
	      counter++;
	      p+=13; // next row
	      index = ( int ) * ( p+9 );
	    }while( (index != surfaceIndex) && (counter < xno) );

	  if ( counter >= xno ) { // then jump to next row on front surface
	    p += offset; // next row of front surface
	    index = ( int ) * ( p+9 );
	    //    std::cout << " Cube line 141 " << std::endl;
	  }

	}while ( ( index != surfaceIndex ));

	// out of do loop, then index must be equal to surfaceIndex
	if ( index == surfaceIndex ) {
	  // std::cout << " Cube line 148" << std::endl;
	  zup = * p;
	  zlow = * ( p+1 );
	  xlow = * ( p+4 );
	  xup = * ( p+5 );
	  VolIndex = ( int ) * ( p+6 );
	  // std::cout << " Cube line 153 " << std::endl;
	}
	else {
	  std::cerr << " Error looking up front surface table";
	  exit (1);
	}
      } // end if( index == surfaceIndex ) else
      
      yfront = * ( p + 3);

      alow = xlow;
      aup = xup;
      blow = zlow;
      bup = zup;
      constv = yfront;
      vIndex = VolIndex;
      
//       cout << " xlow = " << xlow << " ; xup = " << xup << endl;
//       cout << " zlow = " << zlow << " ; zup = " << zup << endl;
//       cout << " VolIndex = " << VolIndex << endl;
//       cout << " yfront = " << yfront << endl;
}


FrontRealSurface::~FrontRealSurface(){
}

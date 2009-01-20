#include "BackRealSurface.h"
#include <iostream>

using std::cout;
using std::endl;

// BackRealSurface::BackRealSurface(int _surfaceIndex,
// 				 int _TopBottomNo, int _xno)
//   : RealSurface ( _surfaceIndex ){
// }

BackRealSurface::BackRealSurface(){
}

void
BackRealSurface::setData(int _surfaceIndex,
			 int _TopBottomNo, int _xno){
  surfaceIndex = _surfaceIndex;
  TopBottomNo = _TopBottomNo;
  xno = _xno;
}

// n top -- n = 0 i + -1 j + 0 k  
void
BackRealSurface::get_n(){
  n[0] = 0;
  n[1] = -1;
  n[2] = 0;
}


// t1 top -- t1 = 1 i + 0 j + 0 k
void
BackRealSurface::get_t1(){
  t1[0] = 1;
  t1[1] = 0;
  t1[2] = 0;
}



// t2 top -- t2 = 0 i + 0 j + 1 k
void
BackRealSurface::get_t2(){
  t2[0] = 0;
  t2[1] = 0;
  t2[2] = 1;
}


void
BackRealSurface::set_n(double *nn){
  for ( int i = 0; i < 3; i ++ )
    nn[i] = n[i];
}


void
BackRealSurface::get_limits(double *VolTable, int &vIndex){
  
  int index, counter, offset;
  double *p;
  offset = (TopBottomNo - xno) * 13;
  //  cout << "offset = " << offset << endl;
  //  cout << " backRealSurface line 68 surfaceIndex = " << surfaceIndex << endl;
      p = VolTable;
      //    cout << " back line 70 " << endl; 
      index = ( int ) * ( p+10 ); // table's back element index
      //   cout << "index = " << index << endl;
      VolIndex = ( int ) * ( p+6 );
      //  cout << " line 73 " << endl;

      //   cout << "here at line 71 in BackRealSurface " << endl;
      
      if ( index == surfaceIndex ) {
	zup = * p;
	zlow = * ( p+1 );
	xlow = * ( p+4 );
	xup = * ( p+5 );
      }
      else {
	
	do {
	  counter = 0;
	  do 
	    {
	      counter++;
	      p+=13; // next row
	      index = ( int ) *( p+10 );
	      //   cout<<"index interor do = " << index << endl;
	    }while( (index != surfaceIndex) && (counter < xno) );
	 
	  if ( counter >= xno ) { // then jump to next row on back surface
	   
	    p += offset; // next row of back surface
	   
	    index = ( int ) *( p+10 );
	    // cout << "BackRealSurface index = " << index << endl;
         
	  }
	  //  cout << "here at line 94 in BackRealSurface " << endl;
	}while ( ( index != surfaceIndex ));
	// cout << "here at line 96 in BackRealSurface " << endl;
	// out of do loop, then index must be equal to surfaceIndex
	if ( index == surfaceIndex ) {
	  zup = * p;
	  zlow = * ( p+1 );
	  xlow = * ( p+4 );
	  xup = * ( p+5 );
	  VolIndex = ( int ) * ( p+6 );
	}
	else {
	  std::cerr << " Error looking up back surface table";
	  exit (1);
	}

      } // end if( index == surfaceIndex ) else
      //  cout << "here at line 111 in BackRealSurface " << endl;      
      yback = * ( p + 2);

      alow = xlow;
      aup = xup;
      blow = zlow;
      bup = zup;
      constv = yback;
      vIndex = VolIndex;
      
//       cout << " xlow = " << xlow << "; xup = " << xup << endl;
//       cout << " zlow = " << zlow << "; zup = " << zup << endl;
//       cout << " VolIndex = " << VolIndex << endl;
//       cout << " yback = " << yback << endl;
}

BackRealSurface::~BackRealSurface(){
}

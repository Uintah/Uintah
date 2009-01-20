#include "BackRealSurface.h"
#include <iostream>

using std::cout;
using std::endl;


//inline
BackRealSurface::BackRealSurface(const int &iIndex,
				 const int &jIndex,
				 const int &kIndex,
				 const int &Ncx){
  
  surfaceiIndex = iIndex;
  surfacejIndex = jIndex+1;
  surfacekIndex = kIndex;
  //  FrontBackNo = NoSurface;
  surfaceIndex = iIndex + kIndex * Ncx;
  
}


BackRealSurface::BackRealSurface(){
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
BackRealSurface::get_limits(const double *X,
			    const double *Y,
			    const double *Z){
  
  // i, j, k is settled at the center of the VOLUME cell
  xlow = X[surfaceiIndex];
  xup = X[surfaceiIndex+1];
  
  ylow = Y[surfacejIndex];
  yup = Y[surfacejIndex];

  // note that for top surface, zlow = ztop
  zlow = Z[surfacekIndex];
  zup = Z[surfacekIndex+1];

}


BackRealSurface::~BackRealSurface(){
}

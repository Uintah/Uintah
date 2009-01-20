#include "BottomRealSurface.h"
#include <iostream>

using std::cout;
using std::endl;

//inline
BottomRealSurface::BottomRealSurface(const int &iIndex,
				     const int &jIndex,
				     const int &kIndex,
				     const int &Ncx){
  
  surfaceiIndex = iIndex;
  surfacejIndex = jIndex;
  surfacekIndex = kIndex;
  // TopBottomNo = NoSurface;
  surfaceIndex = iIndex + jIndex * Ncx;
  
}


BottomRealSurface::BottomRealSurface(){
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
BottomRealSurface::get_limits(const double *X,
			      const double *Y,
			      const double *Z){
  
  // i, j, k is settled at the center of the VOLUME cell
  xlow = X[surfaceiIndex];
  xup = X[surfaceiIndex+1];
  
  ylow = Y[surfacejIndex];
  yup = Y[surfacejIndex+1];
  
  // note that for top surface, zlow = ztop
  zlow = Z[surfacekIndex];
  zup = Z[surfacekIndex];

}


BottomRealSurface::~BottomRealSurface(){
}

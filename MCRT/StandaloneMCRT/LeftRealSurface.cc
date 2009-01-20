#include "LeftRealSurface.h"
#include <iostream>
#include <cstdlib>

using std::cout;
using std::endl;

//inline
LeftRealSurface::LeftRealSurface(const int &iIndex,
				 const int &jIndex,
				 const int &kIndex,
				 const int &Ncy){
  
  surfaceiIndex = iIndex;
  surfacejIndex = jIndex;
  surfacekIndex = kIndex;
  //  LeftRightNo = NoSurface;
  surfaceIndex = jIndex + kIndex * Ncy;
  
}


LeftRealSurface::LeftRealSurface(){
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



void
LeftRealSurface::get_limits(const double *X,
			    const double *Y,
			    const double *Z){
  
  // i, j, k is settled at the center of the VOLUME cell
  xlow = X[surfaceiIndex];
  xup = X[surfaceiIndex];
  
  ylow = Y[surfacejIndex];
  yup = Y[surfacejIndex+1];

  // note that for top surface, zlow = ztop
  zlow = Z[surfacekIndex];
  zup = Z[surfacekIndex+1];

}



LeftRealSurface::~LeftRealSurface(){
}

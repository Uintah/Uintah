#include "RightRealSurface.h"
#include <cstdlib>
#include <iostream>

using std::cout;
using std::endl;


//inline
RightRealSurface::RightRealSurface(const int &iIndex,
				   const int &jIndex,
				   const int &kIndex,
				   const int &Ncy){
  
  surfaceiIndex = iIndex+1;
  surfacejIndex = jIndex;
  surfacekIndex = kIndex;
  surfaceIndex = jIndex + kIndex * Ncy;
  
}



RightRealSurface::RightRealSurface(){
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
RightRealSurface::get_limits(const double *X,
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



RightRealSurface::~RightRealSurface(){
}

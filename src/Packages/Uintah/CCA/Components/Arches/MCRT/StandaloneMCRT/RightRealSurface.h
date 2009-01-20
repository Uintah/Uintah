#ifndef RightRealSurface_H
#define RightRealSurface_H

#include "RealSurface.h"

class RightRealSurface:public RealSurface{
  
public:
  
  RightRealSurface(const int &iIndex,
		   const int &jIndex,
		   const int &kIndex,
		   const int &Ncy);
  
  RightRealSurface();
  ~RightRealSurface();
  
  virtual void set_n(double *nn);
  virtual void get_n();
  virtual void get_t1();
  virtual void get_t2();
  virtual void get_limits(const double *X,
			  const double *Y,
			  const double *Z);
  
  
};

#endif
  

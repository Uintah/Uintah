#ifndef LeftRealSurface_H
#define LeftRealSurface_H

#include "RealSurface.h"

class LeftRealSurface:public RealSurface{
  
public:

  LeftRealSurface(const int &iIndex,
		  const int &jIndex,
		  const int &kIndex,
		  const int &Ncy);
  
  LeftRealSurface();
  ~LeftRealSurface();

  virtual void set_n(double *nn);
  virtual void get_n();
  virtual void get_t1();
  virtual void get_t2();
  virtual void get_limits(const double *X,
			  const double *Y,
			  const double *Z);
  
  
// private:
//   int LeftRightNo;
};

#endif
  

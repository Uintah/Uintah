#ifndef BackRealSurface_H
#define BackRealSurface_H

#include "RealSurface.h"


class BackRealSurface:public RealSurface{
public:
  
  BackRealSurface(const int &iIndex,
		  const int &jIndex,
		  const int &kIndex,
		  const int &Ncx);
  
  BackRealSurface();
  
  ~BackRealSurface();
  virtual void set_n(double *nn);
  virtual void get_n();
  virtual void get_t1();
  virtual void get_t2();
  virtual void get_limits(const double *X,
			  const double *Y,
			  const double *Z);
// private:

//   int FrontBackNo;
  
};

#endif
  

#ifndef TopRealSurface_H
#define TopRealSurface_H

#include "RealSurface.h"

class TopRealSurface : public RealSurface {
  
public:
  
  TopRealSurface(const int &iIndex,
		 const int &jIndex,
		 const int &kIndex,
		 const int &Ncx);

  TopRealSurface();
  ~TopRealSurface();
  
  virtual void set_n(double *nn); 
  virtual void get_n();
  virtual void get_t1();
  virtual void get_t2();
  virtual void get_limits(const double *X,
			  const double *Y,
			  const double *Z);
  

// private:
//   int TopBottomNo;
};

#endif
  

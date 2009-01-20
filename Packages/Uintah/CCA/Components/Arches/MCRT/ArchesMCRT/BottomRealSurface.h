#ifndef BottomRealSurface_H
#define BottomRealSurface_H

#include "RealSurface.h"

class BottomRealSurface:public RealSurface{  
public:
 //  BottomRealSurface(int _surfaceIndex,
// 		    int _TopBottomNo, int _VolElementNo);
  BottomRealSurface();
		    
  ~BottomRealSurface();
  void setData(int _surfaceIndex,
	       int _TopBottomNo,
	       int _VolElementNo);
  virtual void set_n(double *nn);
  virtual void get_n();
  virtual void get_t1();
  virtual void get_t2();
  virtual void get_limits(double *VolTable, int &vIndex);
private:
  // double n[3], t1[3], t2[3];   
  double xlow, xup, ylow, yup;
  double zbottom;
  int VolIndex;
  int TopBottomNo, VolElementNo;
};

#endif
  

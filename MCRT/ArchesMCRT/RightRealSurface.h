#ifndef RightRealSurface_H
#define RightRealSurface_H

#include "RealSurface.h"

class RightRealSurface:public RealSurface{
public:
  // RightRealSurface(int _surfaceIndex, int _xno);
  RightRealSurface();
  void setData(int _surfaceIndex, int _xno);
  ~RightRealSurface();
  virtual void set_n(double *nn);
  virtual void get_n();
  virtual void get_t1();
  virtual void get_t2();
  virtual void get_limits(double *VolTable, int &vIndex);
private:
  //double n[3], t1[3], t2[3];   
  double ylow, yup, zlow, zup;
  double xright;
  int VolIndex;
  int xno;
};

#endif
  

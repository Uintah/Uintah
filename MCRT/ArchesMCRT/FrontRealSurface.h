#ifndef FrontRealSurface_H
#define FrontRealSurface_H

#include "RealSurface.h"


class FrontRealSurface:public RealSurface{
public:
//   FrontRealSurface(int _surfaceIndex,
// 		   int _TopBottomNo,int _xno);
  FrontRealSurface();
  ~FrontRealSurface();
  void setData(int _surfaceIndex, int _TopBottomNo, int _xno);
  virtual void set_n(double *nn);
  virtual void get_n();
  virtual void get_t1();
  virtual void get_t2();
  virtual void get_limits(double *VolTable, int &vIndex);
private:
  //double n[3], t1[3], t2[3];   
  double xlow, xup, zlow, zup;
  double yfront;
  int VolIndex;
  int TopBottomNo, xno;
};

#endif
  

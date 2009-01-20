#ifndef BackRealSurface_H
#define BackRealSurface_H

#include "RealSurface.h"


class BackRealSurface:public RealSurface{
public:
//   BackRealSurface(int _surfaceIndex,
// 		  int _TopBottomNo, int _xno);
  BackRealSurface();
  void setData(int _surfaceIndex, int _TopBottomNo, int _xno);
  ~BackRealSurface();
  virtual void set_n(double *nn);
  virtual void get_n();
  virtual void get_t1();
  virtual void get_t2();
  virtual void get_limits(double *VolTable, int &vIndex);
private:
  // double n[3], t1[3], t2[3];   
  double xlow, xup, zlow, zup;
  double yback;
  int VolIndex;
  int TopBottomNo, xno;
};

#endif
  

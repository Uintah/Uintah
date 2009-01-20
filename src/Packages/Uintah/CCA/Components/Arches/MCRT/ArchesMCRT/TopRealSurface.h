#ifndef TopRealSurface_H
#define TopRealSurface_H

#include "RealSurface.h"

class TopRealSurface : public RealSurface {
  
public:
  TopRealSurface();
  // default constructor necessary for dynamics allocation
  //  TopRealSurface(int _surfaceIndex, int _TopBottomNo);
  void setData(int _surfaceIndex, int _TopBottomNo);
  ~TopRealSurface();
  virtual void set_n(double *n); 
  virtual void get_n();
  virtual void get_t1();
  virtual void get_t2();
  virtual void get_limits(double *VolTable, int &vIndex);
  
private:
  //  double n[3], t1[3], t2[3];   
  double xlow, xup, ylow, yup;
  double ztop;
  int VolIndex;
  int TopBottomNo;
};

#endif
  

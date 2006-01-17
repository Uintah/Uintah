#ifndef UINTAH_COMPARE_MMS_MMS_BASE_H
#define UINTAH_COMPARE_MMS_MMS_BASE_H

class MMS {

public:
  virtual double pressure( int x, int y, double time ) = 0;
  virtual double uVelocity( int x, int y, double time ) = 0;

protected:
  static double A_;
  static double viscosity_;
  static double p_ref_;
  
};

#endif

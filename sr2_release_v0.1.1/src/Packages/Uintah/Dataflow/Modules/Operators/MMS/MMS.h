#ifndef UINTAH_COMPARE_MMS_MMS_BASE_H
#define UINTAH_COMPARE_MMS_MMS_BASE_H

class MMS {

public:
  MMS() {}
  virtual ~MMS() {}
  virtual double pressure( double x_pos, double y_pos, double time ) = 0;
  virtual double uVelocity( double x_pos, double y_pos, double time ) = 0;
  virtual double vVelocity( double x_pos, double y_pos, double time ) = 0;

protected:
  static double A_;
  static double viscosity_;
  static double p_ref_;
  
};

#endif

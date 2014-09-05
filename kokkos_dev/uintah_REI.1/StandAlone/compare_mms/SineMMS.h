#ifndef UINTAH_COMPARE_MMS_SineMMS_H
#define UINTAH_COMPARE_MMS_SineMMS_H


#include <Packages/Uintah/StandAlone/compare_mms/MMS.h>

class SineMMS : public MMS {

public:
  SineMMS(double A, double viscosity, double p_ref);
  ~SineMMS();

  double pressure( double x, double y, double z, double time );
  double uVelocity( double x, double y, double z, double time );
  double vVelocity( double x, double y, double z, double time );
  double wVelocity( double x, double y, double z, double time );
  

private:
  double A_;
  double viscosity_;
  double p_ref_;
};

#endif

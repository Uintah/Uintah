#ifndef UINTAH_COMPARE_MMS_ExpMMS_H
#define UINTAH_COMPARE_MMS_ExpMMS_H


#include <Packages/Uintah/StandAlone/tools/compare_mms/MMS.h>

class ExpMMS : public MMS {

public:
  ExpMMS(double A, double viscosity, double p_ref);
  virtual ~ExpMMS();

  double pressure( double x, double y, double z, double time );
  double uVelocity( double x, double y, double z, double time );
  double vVelocity( double x, double y, double z, double time );
  double wVelocity( double x, double y, double z, double time );

private:
  double d_A;
  double d_viscosity;
  double d_p_ref;
  
};

#endif

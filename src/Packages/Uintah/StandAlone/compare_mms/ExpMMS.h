#ifndef UINTAH_COMPARE_MMS_ExpMMS_H
#define UINTAH_COMPARE_MMS_ExpMMS_H


#include <Packages/Uintah/StandAlone/compare_mms/MMS.h>

class ExpMMS : public MMS {

public:
  ExpMMS();
  ~ExpMMS();

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

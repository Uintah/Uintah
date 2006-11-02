#ifndef UINTAH_COMPARE_MMS_LinearMMS_H
#define UINTAH_COMPARE_MMS_LinearMMS_H


#include <Packages/Uintah/StandAlone/compare_mms/MMS.h>

class LinearMMS : public MMS {

public:
  LinearMMS(double cu, double cv, double cw, double cp, double p_ref);
  ~LinearMMS();

  double pressure( double x, double y, double z, double time );
  double uVelocity( double x, double y, double z, double time );
  double vVelocity( double x, double y, double z, double time );
  double wVelocity( double x, double y, double z, double time );
 
  
private:

  double c_u;
  double c_v;
  double c_w;
  double c_p;
  double p_ref_;
};

#endif

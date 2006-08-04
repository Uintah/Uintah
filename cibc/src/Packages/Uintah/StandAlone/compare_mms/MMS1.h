#ifndef UINTAH_COMPARE_MMS_MMS1_H
#define UINTAH_COMPARE_MMS_MMS1_H


#include <Packages/Uintah/StandAlone/compare_mms/MMS.h>

class MMS1 : public MMS {

public:
  virtual double pressure( int x, int y, double time );
  virtual double uVelocity( int x, int y, double time );
  
};

#endif

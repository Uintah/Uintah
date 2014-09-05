#ifndef UINTAH_COMPARE_MMS_MMS1_H
#define UINTAH_COMPARE_MMS_MMS1_H

#include <Packages/Uintah/Dataflow/Modules/Operators/MMS/MMS.h>

class MMS1 : public MMS {

public:
  MMS1() {};
  virtual ~MMS1() {};
  virtual double pressure( int x, int y, double time );
  virtual double uVelocity( int x, int y, double time );
  
};

#endif

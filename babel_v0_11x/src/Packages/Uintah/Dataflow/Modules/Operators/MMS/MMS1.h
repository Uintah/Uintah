#ifndef UINTAH_COMPARE_MMS_MMS1_H
#define UINTAH_COMPARE_MMS_MMS1_H

#include <Packages/Uintah/Dataflow/Modules/Operators/MMS/MMS.h>

class MMS1 : public MMS {

public:
  MMS1() {}
  virtual ~MMS1() {}
  virtual double pressure( double x_pos, double y_pos, double time );
  virtual double uVelocity( double x_pos, double y_pos, double time );
  virtual double vVelocity( double x_pos, double y_pos, double time );
  
};

#endif

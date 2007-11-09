#ifndef UINTAH_COMPARE_MMS_MMS1_H
#define UINTAH_COMPARE_MMS_MMS1_H

#include <Dataflow/Modules/Operators/MMS/MMS.h>

#include <Dataflow/Modules/Operators/MMS/uintahshare.h>

class MMS1 : public MMS {

public:
  MMS1() {}
  virtual ~MMS1() {}
  virtual UINTAHSHARE double pressure( double x_pos, double y_pos, double time );
  virtual UINTAHSHARE double uVelocity( double x_pos, double y_pos, double time );
  virtual UINTAHSHARE double vVelocity( double x_pos, double y_pos, double time );
  
};

#endif

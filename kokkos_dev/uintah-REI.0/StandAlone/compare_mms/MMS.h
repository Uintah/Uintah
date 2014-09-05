#ifndef UINTAH_COMPARE_MMS_MMS_BASE_H
#define UINTAH_COMPARE_MMS_MMS_BASE_H

class MMS {

public:
  //MMS() {}
  //virtual ~MMS() {}

  virtual double pressure( double x, double y, double z, double time ) = 0;
  virtual double uVelocity( double x, double y, double z, double time ) = 0;
  virtual double vVelocity( double x, double y, double z, double time ) = 0;
  virtual double wVelocity( double x, double y, double z, double time ) = 0;
  
};

#endif

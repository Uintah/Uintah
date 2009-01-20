#ifndef VirtualSurface_H
#define VirtualSurface_H

#include "Surface.h"
#include "RNG.h"

class RNG;

class VirtualSurface : public Surface{
public:
  VirtualSurface();
  ~VirtualSurface();
  virtual void getTheta(double &theta, double &random);

  // get sIn
  void get_sIn(double *sIncoming);
  
  //get e1-- e1
  void get_e1(double &random1, double &random2, double &random3);

  //get e2 -- e2
  void get_e2();

  // get scatter_s
  void get_s(RNG &rng, double *sIncoming,
	     double &theta, double &phi,
	     double &random1, double &random2, double &random3,
             double *s);
private:
  double sIn[3], e1[3], e2[3];

};

#endif

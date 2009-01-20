#ifndef VirtualSurface_H
#define VirtualSurface_H

#include "Surface.h"
#include "RNG.h"

class RNG;

class VirtualSurface : public Surface{
public:
  VirtualSurface();
  ~VirtualSurface();
  virtual void getTheta(const double &random);

//   // get sIn
//   void get_sIn(double *sIn);
  
  //get e1-- e1
  void get_e1(const double &random1,
	      const double &random2,
	      const double &random3,
	      const double *sIn);

  //get e2 -- e2
  void get_e2(const double *sIn);

  // get scatter_s
  void get_s(RNG &rng, const double *sIn, double *s);
private:
  double e1[3], e2[3];
};

#endif

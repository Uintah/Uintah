#ifndef RealSurface_H
#define RealSurface_H

#include "Surface.h"
#include "RNG.h" // has to be included
//#include "Consts.h"

class RNG;
class ray;

class RealSurface : public Surface {
public:
  
//   RealSurface(int _surfaceIndex,
// 	      int _BottomStartNo,
// 	      int _BackStartNo, int _FrontStartNo,
// 	      int _LeftStartNo, int _RightStartNo,
// 	      int _TopBottomNo, int _LeftRightNo, int _VolElementNo,
// 	      int _xno);
  
  //RealSurface(int _surfaceIndex);

  RealSurface();
  
  void get_s(RNG &rng,
	     double &theta, double &random1,
	     double &phi, double &random2,
	     double *s);
  
  // put limits and vIndex private data to public data member
  void get_public_limits(double &_alow, double &_aup,
			 double &_blow, double &_bup,
			 double &_constv);
	      
  virtual void getTheta(double &theta, double &random);
  
  virtual void set_n(double *nn) = 0;

  // given surfaceIndex, find limits of that surface element
  virtual void get_limits(double *VolTable, int &vIndex) = 0;
  virtual void get_n() = 0;
  virtual void get_t1() = 0;
  virtual void get_t2() = 0;
  
  friend class ray;
  ~RealSurface();

protected:
  int surfaceIndex;
  double n[3], t1[3], t2[3];
  double alow, aup, blow, bup, constv;
};

#endif 

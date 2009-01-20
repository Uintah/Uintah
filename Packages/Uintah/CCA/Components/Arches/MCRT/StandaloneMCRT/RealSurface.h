#ifndef RealSurface_H
#define RealSurface_H

#include "Surface.h"
#include "RNG.h" // has to be included


class RNG;
class ray;

class RealSurface : public Surface {
public:
  
  RealSurface();
  
  void get_s(RNG &rng, double *s);
  
  virtual void getTheta(const double &random);
  
  virtual void set_n(double *nn) = 0;

  // given surfaceIndex, find limits of that surface element
  virtual void get_limits(const double *X,
			  const double *Y,
			  const double *Z) = 0;

  virtual void get_n() = 0;
  virtual void get_t1() = 0;
  virtual void get_t2() = 0;

  int get_surfaceIndex();
  int get_surfaceiIndex();
  int get_surfacejIndex();
  int get_surfacekIndex();
  double get_xlow();
  double get_xup();
  double get_ylow();
  double get_yup();
  double get_zlow();
  double get_zup();
  
  double SurfaceEmissFlux(const int &i,
			  const double *emiss_surface,
			  const double *T_surface,
			  const double *a_surface);
			 
  double SurfaceEmissFluxBlack(const int &i,
			       const double *T_surface,
			       const double *a_surface);
  
  
  double SurfaceIntensity(const int &i,
			  const double *emiss_surface,
			  const double *T_surface,
			  const double *a_surface);

  
  double SurfaceIntensityBlack(const int &i,
			       const double *T_surface,
			       const double *a_surface);  
  
  friend class ray;
  
  ~RealSurface();

protected:

  double n[3], t1[3], t2[3];
  double xlow, xup, ylow, yup, zlow, zup;
  int surfaceIndex;
  int surfaceiIndex, surfacejIndex, surfacekIndex;  
 
};

#endif 

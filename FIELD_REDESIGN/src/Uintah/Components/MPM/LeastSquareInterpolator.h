#ifndef __LEAST_SQUARE_INTERPOLATOR_H__
#define __LEAST_SQUARE_INTERPOLATOR_H__

class LeastSquareInterpolator {
 public:
  virtual void fromNeighboringParticles(ParticleSet &p) = 0;

};

#endif __LEAST_SQUARE_INTERPOLATOR_H__

// $Log$
// Revision 1.2  2000/03/15 21:58:21  jas
// Added logging and put guards in.
//

#ifndef NODE27_INTERPOLATOR_H
#define NODE27_INTERPOLATOR_H

#include <Packages/Uintah/CCA/Components/MPM/ParticleInterpolator.h>

namespace Uintah {

  class Patch;

  class Node27Interpolator : public ParticleInterpolator {
    
  public:
    
    Node27Interpolator();
    Node27Interpolator(const Patch* patch);
    virtual ~Node27Interpolator();
    
    virtual Node27Interpolator* clone(const Patch*);
    
    virtual void findCellAndWeights(const Point& p,IntVector *ni, 
				    double *S, const Vector& size);
    virtual void findCellAndShapeDerivatives(const Point& pos,
					     IntVector *ni,
					     Vector *d_S,
					     const Vector& size);
    virtual void findCellAndWeightsAndShapeDerivatives(const Point& pos,
						       IntVector *ni,
						       double *S,
						       Vector *d_S,
						       const Vector& size);
    virtual int size();
    
  private:
    const Patch* d_patch;
    int d_size;
    
  };
}

#endif


#ifndef NODE27_INTERPOLATOR_H
#define NODE27_INTERPOLATOR_H

#include <Packages/Uintah/Core/Math/ParticleInterpolator.h>

namespace Uintah {

  class Patch;

  class Node27Interpolator : public ParticleInterpolator {
    
  public:
    
    Node27Interpolator();
    Node27Interpolator(const Patch* patch);
    virtual ~Node27Interpolator();
    
    virtual Node27Interpolator* clone(const Patch*);
    
    virtual void findCellAndWeights(const Point& p,vector<IntVector>& ni, 
				    vector<double>& S, const Vector& size);
    virtual void findCellAndShapeDerivatives(const Point& pos,
					     vector<IntVector>& ni,
					     vector<Vector>& d_S,
					     const Vector& size);
    virtual void findCellAndWeightsAndShapeDerivatives(const Point& pos,
						       vector<IntVector>& ni,
						       vector<double>& S,
						       vector<Vector>& d_S,
						       const Vector& size);
    virtual int size();
    
  private:
    const Patch* d_patch;
    int d_size;
    
  };
}

#endif


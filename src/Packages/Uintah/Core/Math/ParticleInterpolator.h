#ifndef PARTICLE_INTERPOLATOR_H
#define PARTICLE_INTERPOLATOR_H

#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <vector>

namespace Uintah {

  class Patch;
  using SCIRun::Vector;
  using SCIRun::IntVector;
  using SCIRun::Point;
  using std::vector;

  class ParticleInterpolator {
    
  public:
    
    ParticleInterpolator() {};
    virtual ~ParticleInterpolator() {};
    
    virtual ParticleInterpolator* clone(const Patch*) = 0;
    
    virtual void findCellAndWeights(const Point& p,vector<IntVector>& ni, 
				    vector<double>& S,const Vector& size) = 0;
    virtual void findCellAndShapeDerivatives(const Point& pos,
					     vector<IntVector>& ni,
					     vector<Vector>& d_S,
					     const Vector& size) = 0;
    virtual void findCellAndWeightsAndShapeDerivatives(const Point& pos,
						       vector<IntVector>& ni,
						       vector<double>& S,
						       vector<Vector>& d_S,
						       const Vector& size) = 0;
    virtual int size() = 0;

    
  };
}

#endif


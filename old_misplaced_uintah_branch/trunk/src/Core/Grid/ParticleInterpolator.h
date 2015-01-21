#ifndef PARTICLE_INTERPOLATOR_H
#define PARTICLE_INTERPOLATOR_H

#include <SCIRun/Core/Geometry/Point.h>
#include <SCIRun/Core/Geometry/IntVector.h>
#include <SCIRun/Core/Geometry/Vector.h>
#include <vector>

#include <Core/Grid/uintahshare.h>
#include <Core/Grid/Variables/NCVariable.h>
namespace Uintah {

  class Patch;
  class Stencil7;
  using SCIRun::Vector;
  using SCIRun::IntVector;
  using SCIRun::Point;
  using std::vector;

  class UINTAHSHARE ParticleInterpolator {
    
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

    virtual void findCellAndWeights(const Point& p,vector<IntVector>& ni,
                                    vector<double>& S,
                                    constNCVariable<Stencil7>& zoi,
                                    constNCVariable<Stencil7>& zoi_fine,
                                    const bool& getFiner,
                                    int& num_cur,int& num_fine,int& num_coarse,                                     const Vector& size, bool coarse_part,
                                    const Patch* patch) {};

    virtual int size() = 0;
  };
}

#endif


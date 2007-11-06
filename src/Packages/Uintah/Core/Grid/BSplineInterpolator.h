#ifndef BSPLINE_INTERPOLATOR_H
#define BSPLINE_INTERPOLATOR_H

#include <Packages/Uintah/Core/Grid/ParticleInterpolator.h>

#include <Packages/Uintah/Core/Grid/uintahshare.h>
namespace Uintah {

  class Patch;

  class UINTAHSHARE BSplineInterpolator : public ParticleInterpolator {
    
  public:
    
    BSplineInterpolator();
    BSplineInterpolator(const Patch* patch);
    virtual ~BSplineInterpolator();
    
    virtual BSplineInterpolator* clone(const Patch*);
    
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

    void findNodeComponents(const int& idx, int* xn, int& count,
                            const int& low, const int& hi);

    void getBSplineWeights(double* Sd, const int* xn,
                           const int& low, const int& hi,
                           const int& count, const double& cellpos);

    void getBSplineGrads(double* dSd, const int* xn,
                         const int& low, const int& hi, const int& count,
                         const double& cellpos);

    double evalType1BSpline(const double& cp);
    double evalType2BSpline(const double& cp);
    double evalType3BSpline(const double& cp);

    double evalType1BSplineGrad(const double& cp);
    double evalType2BSplineGrad(const double& cp);
    double evalType3BSplineGrad(const double& cp);

  private:
    const Patch* d_patch;
    int d_size;
    
  };
}

#endif

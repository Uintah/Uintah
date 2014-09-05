/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef BSPLINE_INTERPOLATOR_H
#define BSPLINE_INTERPOLATOR_H

#include <Core/Grid/ParticleInterpolator.h>

#include <Core/Grid/uintahshare.h>
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

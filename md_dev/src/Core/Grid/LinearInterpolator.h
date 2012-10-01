/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


#ifndef LINEAR_INTERPOLATOR_H
#define LINEAR_INTERPOLATOR_H

#include <Core/Math/MiscMath.h>
#include <Core/Grid/ParticleInterpolator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <vector>

namespace Uintah {

  using namespace SCIRun;
  using std::vector;

  class LinearInterpolator : public ParticleInterpolator {
    
  public:
    
    LinearInterpolator();
    LinearInterpolator(const Patch* patch);
    virtual ~LinearInterpolator();
    
    virtual LinearInterpolator* clone(const Patch*);
    
    
//__________________________________
//  This version of findCellAndWeights is only used
//  by MPM/HeatConduction/ImplicitHeatConduction.cc
    inline void findCellAndWeights(const Point& pos,
                                   vector<IntVector>& ni, 
                                   vector<double>& S)
      {
        Point cellpos = d_patch->getLevel()->positionToIndex(pos);
        int ix = Floor(cellpos.x());
        int iy = Floor(cellpos.y());
        int iz = Floor(cellpos.z());
        ni[0] = IntVector(ix, iy, iz);
        ni[1] = IntVector(ix, iy, iz+1);
        ni[2] = IntVector(ix, iy+1, iz);
        ni[3] = IntVector(ix, iy+1, iz+1);
        ni[4] = IntVector(ix+1, iy, iz);
        ni[5] = IntVector(ix+1, iy, iz+1);
        ni[6] = IntVector(ix+1, iy+1, iz);
        ni[7] = IntVector(ix+1, iy+1, iz+1);
        double fx = cellpos.x() - ix;
        double fy = cellpos.y() - iy;
        double fz = cellpos.z() - iz;
        double fx1 = 1-fx;
        double fy1 = 1-fy;
        double fz1 = 1-fz;
        S[0] = fx1 * fy1 * fz1;
        S[1] = fx1 * fy1 * fz;
        S[2] = fx1 * fy * fz1;
        S[3] = fx1 * fy * fz;
        S[4] = fx * fy1 * fz1;
        S[5] = fx * fy1 * fz;
        S[6] = fx * fy * fz1;
        S[7] = fx * fy * fz;
      };
    //__________________________________

    virtual void findCellAndWeights(const Point& p,
                                    vector<IntVector>& ni, 
				         vector<double>& S,
                                     const Matrix3& size, 
                                     const Matrix3& defgrad);
                                
    
    //__________________________________
    //  AMRMPM                                
    virtual void findCellAndWeights_CFI(const Point& pos,
                                        vector<IntVector>& ni,
                                        vector<double>& S,
                                        constNCVariable<Stencil7>& zoi);
                                        
    virtual void findCellAndWeightsAndShapeDerivatives_CFI(
                                            const Point& pos,
                                            vector<IntVector>& CFI_ni,
                                            vector<double>& S,
                                            vector<Vector>& d_S,
                                            constNCVariable<Stencil7>& zoi);
    //__________________________________ 
    
    
    inline void findCellAndShapeDerivatives(const Point& pos,
                                            vector<IntVector>& ni,
                                            vector<Vector>& d_S)
      {
        Point cellpos = d_patch->getLevel()->positionToIndex(pos);
        int ix = Floor(cellpos.x());
        int iy = Floor(cellpos.y());
        int iz = Floor(cellpos.z());
        ni[0] = IntVector(ix, iy, iz);
        ni[1] = IntVector(ix, iy, iz+1);
        ni[2] = IntVector(ix, iy+1, iz);
        ni[3] = IntVector(ix, iy+1, iz+1);
        ni[4] = IntVector(ix+1, iy, iz);
        ni[5] = IntVector(ix+1, iy, iz+1);
        ni[6] = IntVector(ix+1, iy+1, iz);
        ni[7] = IntVector(ix+1, iy+1, iz+1);
        double fx = cellpos.x() - ix;
        double fy = cellpos.y() - iy;
        double fz = cellpos.z() - iz;
        double fx1 = 1-fx;
        double fy1 = 1-fy;
        double fz1 = 1-fz;
        d_S[0] = Vector(- fy1 * fz1, -fx1 * fz1, -fx1 * fy1);
        d_S[1] = Vector(- fy1 * fz,  -fx1 * fz,   fx1 * fy1);
        d_S[2] = Vector(- fy  * fz1,  fx1 * fz1, -fx1 * fy);
        d_S[3] = Vector(- fy  * fz,   fx1 * fz,   fx1 * fy);
        d_S[4] = Vector(  fy1 * fz1, -fx  * fz1, -fx  * fy1);
        d_S[5] = Vector(  fy1 * fz,  -fx  * fz,   fx  * fy1);
        d_S[6] = Vector(  fy  * fz1,  fx  * fz1, -fx  * fy);
        d_S[7] = Vector(  fy  * fz,   fx  * fz,   fx  * fy);
      };
    virtual void findCellAndShapeDerivatives(const Point& pos,
					          vector<IntVector>& ni,
					          vector<Vector>& d_S,
					          const Matrix3& size, 
                                             const Matrix3& defgrad);
                                        

    inline void findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                      vector<IntVector>& ni,
                                                      vector<double>& S,
                                                      vector<Vector>& d_S) 
      {
        Point cellpos = d_patch->getLevel()->positionToIndex(pos);
        int ix = Floor(cellpos.x());
        int iy = Floor(cellpos.y());
        int iz = Floor(cellpos.z());
        ni[0] = IntVector(ix, iy, iz);
        ni[1] = IntVector(ix, iy, iz+1);
        ni[2] = IntVector(ix, iy+1, iz);
        ni[3] = IntVector(ix, iy+1, iz+1);
        ni[4] = IntVector(ix+1, iy, iz);
        ni[5] = IntVector(ix+1, iy, iz+1);
        ni[6] = IntVector(ix+1, iy+1, iz);
        ni[7] = IntVector(ix+1, iy+1, iz+1);
        double fx = cellpos.x() - ix;
        double fy = cellpos.y() - iy;
        double fz = cellpos.z() - iz;
        double fx1 = 1-fx;
        double fy1 = 1-fy;
        double fz1 = 1-fz;
        S[0] = fx1 * fy1 * fz1;
        S[1] = fx1 * fy1 * fz;
        S[2] = fx1 * fy * fz1;
        S[3] = fx1 * fy * fz;
        S[4] = fx * fy1 * fz1;
        S[5] = fx * fy1 * fz;
        S[6] = fx * fy * fz1;
        S[7] = fx * fy * fz;
        d_S[0] = Vector(- fy1 * fz1, -fx1 * fz1, -fx1 * fy1);
        d_S[1] = Vector(- fy1 * fz,  -fx1 * fz,   fx1 * fy1);
        d_S[2] = Vector(- fy  * fz1,  fx1 * fz1, -fx1 * fy);
        d_S[3] = Vector(- fy  * fz,   fx1 * fz,   fx1 * fy);
        d_S[4] = Vector(  fy1 * fz1, -fx  * fz1, -fx  * fy1);
        d_S[5] = Vector(  fy1 * fz,  -fx  * fz,   fx  * fy1);
        d_S[6] = Vector(  fy  * fz1,  fx  * fz1, -fx  * fy);
        d_S[7] = Vector(  fy  * fz,   fx  * fz,   fx  * fy);
      };

    inline void findNodes(const Point& pos,
                          vector<IntVector>& cur,
                          const Level* level)
      {
        Point cellpos = level->positionToIndex(pos);
        int ix = Floor(cellpos.x());
        int iy = Floor(cellpos.y());
        int iz = Floor(cellpos.z());
                                                                                
        cur[0] = IntVector(ix, iy, iz);
        cur[1] = IntVector(ix, iy, iz+1);
        cur[2] = IntVector(ix, iy+1, iz);
        cur[3] = IntVector(ix, iy+1, iz+1);
        cur[4] = IntVector(ix+1, iy, iz);
        cur[5] = IntVector(ix+1, iy, iz+1);
        cur[6] = IntVector(ix+1, iy+1, iz);
        cur[7] = IntVector(ix+1, iy+1, iz+1);
      };

    virtual void findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                       vector<IntVector>& ni,
                                                       vector<double>& S,
                                                       vector<Vector>& d_S,
                                                       const Matrix3& size,
                                                       const Matrix3& defgrad);
    virtual int size();

  private:
    const Patch* d_patch;
    int d_size;
  };
}

#endif


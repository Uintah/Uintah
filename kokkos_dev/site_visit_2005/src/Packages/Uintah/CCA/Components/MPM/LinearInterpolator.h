#ifndef LINEAR_INTERPOLATOR_H
#define LINEAR_INTERPOLATOR_H

#include <Core/Math/MiscMath.h>
#include <Packages/Uintah/CCA/Components/MPM/ParticleInterpolator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
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
    
    inline void findCellAndWeights(const Point& pos,vector<IntVector>& ni, 
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
    virtual void findCellAndWeights(const Point& p,vector<IntVector>& ni, 
				    vector<double>& S,const Vector& size);
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
					     const Vector& size);
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


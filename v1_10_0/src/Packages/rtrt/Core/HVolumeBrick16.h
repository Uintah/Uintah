
#ifndef HVOLUMEBRICK16_H
#define HVOLUMEBRICK16_H 1

#include <Packages/rtrt/Core/VolumeBase.h>
#include <Core/Geometry/Point.h>
#include <stdlib.h>
#include <Core/Thread/WorkQueue.h>

namespace rtrt {

using SCIRun::WorkQueue;

struct VMCell16;
//class WorkQueue;

class HVolumeBrick16 : public VolumeBase {
protected:
    Point min;
    Vector datadiag;
    Vector hierdiag;
    Vector ihierdiag;
    Vector sdiag;
    int nx,ny,nz;
    short* indata;
    short* blockdata;
    short datamin, datamax;
    int* xidx;
    int* yidx;
    int* zidx;
    int depth;
    int* xsize;
    int* ysize;
    int* zsize;
    double* ixsize;
    double* iysize;
    double* izsize;
    VMCell16** macrocells;
    int** macrocell_xidx;
    int** macrocell_yidx;
    int** macrocell_zidx;
    WorkQueue* work;
    void brickit(int);
    void parallel_calc_mcell(int);
    char* filebase;
    void calc_mcell(int depth, int ix, int iy, int iz, VMCell16& mcell);
    void isect(int depth, float isoval,double t,
	       double dtdx, double dtdy, double dtdz,
	       double next_x, double next_y, double next_z,
	       int ix, int iy, int iz,
	       int dix_dx, int diy_dy, int diz_dz,
	       int startx, int starty, int startz,
	       const Vector& cellcorner, const Vector& celldir,
	       const Ray& ray, HitInfo& hit,
	       DepthStats* st, PerProcessorContext* ppc);

    //used with cutting planes to color an interior point
    bool interior_value_sublevels(double &ret_val,
				  const Point & where,
				  int depth,
				  int ix, int iy, int iz,
				  const Vector &cellcorner,
				  double ixs, double iys, double izs,
				  bool dbgprint=false);
    
public:
    HVolumeBrick16(Material* matl, VolumeDpy* dpy,
		   char* filebase, int depth, int np);
    HVolumeBrick16(Material* matl, VolumeDpy* dpy, HVolumeBrick16* share);
    virtual ~HVolumeBrick16();
    virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
    virtual Vector normal(const Point&, const HitInfo& hit);
    virtual void compute_bounds(BBox&, double offset);
    virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
    virtual void compute_hist(int nhist, int* hist,
			      float datamin, float datamax);
    virtual void get_minmax(float& min, float& max);
    inline int get_nx() {
	return nx;
    }
    inline int get_ny() {
	return ny;
    }
    inline int get_nz() {
	return nz;
    }
    //used with cutting planes to color the interior
    bool interior_value(double& ret_val, const Ray &r, const double t);
};

} // end namespace rtrt

#endif


#ifndef Grid_H
#define Grid_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/BBox.h>

namespace rtrt {

struct GridTree;
struct BoundedObject;

class Grid : public Object {

protected: 
    // It's ugly, but we should be able to access this data
    // from the derived class as well
    Object* obj;
    BBox bbox;
    int nx, ny, nz;
    Object** grid;
    int* counts;
    int nsides;

public:
    Grid(Object* obj, int nside);
    virtual ~Grid();
    virtual void intersect(const Ray& ray,
			   HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
    virtual Vector normal(const Point&, const HitInfo& hit);
    virtual void light_intersect(Light* light, const Ray& ray,
				 HitInfo& hit, double dist, Color& atten,
				 DepthStats* st, PerProcessorContext*);
    void add(Object* obj);
    inline void calc_se(const BBox& obj_bbox, const BBox& bbox,
			const Vector& diag, int nx, int ny, int nz,
			int &sx, int &sy, int &sz,
			int &ex, int &ey, int &ez);
    virtual void animate(double t, bool& changed);
    virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
    virtual void compute_bounds(BBox&, double offset);
    virtual void collect_prims(Array1<Object*>& prims);
};

inline void Grid::calc_se(const BBox& obj_bbox, const BBox& bbox,
			  const Vector& diag,
			  int nx, int ny, int nz,
			  int& sx, int& sy, int& sz,
			  int& ex, int& ey, int& ez)
{
    Vector s((obj_bbox.min()-bbox.min())/diag);
    Vector e((obj_bbox.max()-bbox.min())/diag);
    sx=(int)(s.x()*nx);
    sy=(int)(s.y()*ny);
    sz=(int)(s.z()*nz);
    ex=(int)(e.x()*nx);
    ey=(int)(e.y()*ny);
    ez=(int)(e.z()*nz);
    if(sx < 0 || ex >= nx){
	cerr << "NX out of bounds!\n";
	cerr << "sx=" << sx << ", ex=" << ex << '\n';
	cerr << "e=" << e << '\n';
	cerr << "obj_bbox=" << obj_bbox.min() << ", " << obj_bbox.max() << '\n';
	cerr << "bbox=" << bbox.min() << ", " << bbox.max() << '\n';
	cerr << "diag=" << diag << '\n';
	exit(1);
    }
    if(sy < 0 || ey >= ny){
	cerr << "NY out of bounds!\n";
	cerr << "sy=" << sy << ", ey=" << ey << '\n';
	exit(1);
    }
    if(sz < 0 || ez >= nz){
	cerr << "NZ out of bounds!\n";
	cerr << "sz=" << sz << ", ez=" << ez << '\n';
	cerr << "e=" << e << '\n';
	cerr << "obj_bbox=" << obj_bbox.min() << ", " << obj_bbox.max() << '\n';
	cerr << "bbox=" << bbox.min() << ", " << bbox.max() << '\n';
	cerr << "diag=" << diag << '\n';
	exit(1);
    }
}

} // end namespace rtrt

#endif

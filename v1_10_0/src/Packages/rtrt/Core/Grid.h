
#ifndef Grid_H
#define Grid_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/pcube.h>

#include <iostream>

namespace rtrt {
  class Grid;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::Grid*&);
}

namespace rtrt {

struct GridTree;

using std::cerr;

extern "C" {
  extern int	
  fast_polygon_intersects_cube(int nverts, const real verts[][3],
			       const real polynormal[3],
			       int already_know_verts_are_outside_cube,
			       int already_know_edges_are_outside_cube);
  extern int
  polygon_intersects_cube(int nverts, const real verts[/* nverts */][3],
			  const real polynormal[3],
			  int already_know_vertices_are_outside_cube,/*unused*/
			  int already_know_edges_are_outside_cube);
}	

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
  Grid() : Object(0) {} // for Pio.
  virtual ~Grid();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Grid*&);
  
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void intersect(Ray& ray,
			 HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual void softshadow_intersect(Light* light, Ray& ray,
				    HitInfo& hit, double dist, Color& atten,
				    DepthStats* st, PerProcessorContext* ppc);
  void add(Object* obj);
  void calc_se(const BBox& obj_bbox, const BBox& bbox,
		      const Vector& diag, int nx, int ny, int nz,
		      int &sx, int &sy, int &sz,
		      int &ex, int &ey, int &ez);
  virtual void animate(double t, bool& changed);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_bounds(BBox&, double offset);
  virtual void collect_prims(Array1<Object*>& prims);
};

} // end namespace rtrt

#endif

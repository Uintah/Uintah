
#ifndef Grid2_H
#define Grid2_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Array1.h>

namespace rtrt {

struct GridTree;
struct BoundedObject;

/*
 * Notes:    logical bounding box can be used to make grid traversal
 *           cheaper. It is currently not used, however. This box will
 *           be expanded whenever an object moves outside the physical
 *           grid.
 *
 *           Ray traversal will start within the logical bounding box,
 *           but not necessarily within the physical grid.
 *
 *           Note that all the extra traversal would not be necessary
 *           if the assumption of a closed environment is made. Moving
 *           objects within a closed environment will therefore be cheaper
 *           than in the general case.
 */

class Grid2 : public Object {
  Object* obj;
  BBox bbox;                         // Grid bounding box
  BBox logical_bbox;                 // Bounding box including objects that have moved outside grid
  int nx, ny, nz;                    // Voxels in physical grid
  int nynz;                          // = ny * nz
  Array1<Object *>* grid;            // Grid is now 1D array of Array1<Object *> arrays
  Array1<Object*> prims;
  int nsides;

public:
  Grid2(Object* obj, int nside);
  virtual ~Grid2();
  virtual void io(SCIRun::Piostream &/*stream*/) 
  { ASSERTFAIL("Pio not supported"); }
  virtual void intersect(Ray& ray,
			 HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual Vector normal(const Point&, const HitInfo& hit);
  void add(Object* obj);
  virtual void animate(double t, bool& changed);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_bounds(BBox&, double offset);
  virtual void collect_prims(Array1<Object*>& prims);
  virtual void remove (Object* obj, const BBox& obj_bbox);              // For dynamic updates
  virtual void insert (Object* obj, const BBox& obj_bbox);              // Idem
  virtual void rebuild ();
  virtual void recompute_bbox ();
  inline virtual double get_shape () {
    return (logical_bbox.diagonal().length() / bbox.diagonal().length());
  }
};

} // end namespace rtrt

#endif // Grid2_H

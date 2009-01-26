/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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

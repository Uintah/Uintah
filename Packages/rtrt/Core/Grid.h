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



#ifndef Grid_H
#define Grid_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/pcube.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

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

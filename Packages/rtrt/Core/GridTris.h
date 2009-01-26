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



#ifndef GridTris_H
#define GridTris_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/BrickArray3.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace rtrt {
using std::string;

struct GridTrisTree;
struct BoundedObject;
struct MCell;

class GridTris : public Object, public Material {
  Material* fallbackMaterial;

  struct Tri {
    float n[3]; // Normal
    int idx[3]; // Vertices
  };
  struct Vert {
    float x[3]; // position
    unsigned char color[3];
  };
  vector<Vert> verts;
  vector<Tri> tris;
  BrickArray3<int> counts;
  vector<int> cells;
  BrickArray3<bool>* macrocells;
  Point min;
  Point max;
  Vector diag;
  Vector inv_diag;
  double inv_ncells;
  int ncells;
  int depth;
  bool preprocessed;
  string filename;

  void isect(int depth, double t,
	     double dtdx, double dtdy, double dtdz,
	     double next_x, double next_y, double next_z,
	     int sx, int sy, int sz, int ix, int iy, int iz,
	     int dix_dx, int diy_dy, int diz_dz,
	     const Vector& cellcorner, const Vector& celldir,
	     const Ray& ray, HitInfo& hit);
  void calc_se(Tri& t, int totalcells,
	       int& sx, int& sy, int& sz,
	       int& ex, int& ey, int& ez);
  void calc_mcell(int depth, int startx, int starty, int startz,
		  bool& mcell, int& ntris);
  bool intersects(const Tri& tri, int totalcells, int x, int y, int z);
  struct HitRecord {
    double u, v;
    int idx;
  };
public:
  GridTris(Material* matl, int nsides, int depth,
	   const std::string& filename);
  void addVertex(float x[3], unsigned char c[3]);
  void addTri(int v1, int v2, int v3);
  inline void clearFallback() {fallbackMaterial=0;}

  virtual ~GridTris();
  virtual void io(SCIRun::Piostream &stream);

  virtual void intersect(Ray& ray,
			 HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void transform(Transform&);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_bounds(BBox&, double offset);
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);

  bool isCached();
};

} // end namespace rtrt

#endif

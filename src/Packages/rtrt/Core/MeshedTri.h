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



#ifndef MESHEDTRI_H
#define MESHEDTRI_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/TriMesh.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Context.h>

namespace rtrt {
class MeshedTri;
class MeshedNormedTri;
class MeshedColoredTri;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::MeshedColoredTri*&);
}

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Point;
using SCIRun::Transform;
using SCIRun::AffineCombination;
using SCIRun::Cross;

class MeshedTri : public Object {
protected:
  TriMesh* tm;
  int p1,p2,p3;
  int n1;
public:
  MeshedTri(Material* matl, TriMesh* tm, 
	    const int& p1, const int& p2, const int& p3, const int& n1);
  virtual ~MeshedTri();

  MeshedTri() : Object(0) {} // for Pio.

  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual Vector normal(const Point&, const HitInfo& hit);
  Vector normal()
    {    
      return tm->norms[n1];
    }

  virtual void compute_bounds(BBox&, double offset);

  Point centroid()
  {
    double one_third = 1./3.;

    return AffineCombination(tm->verts[p1],one_third,
			     tm->verts[p2],one_third,
			     tm->verts[p3],one_third);
  }
  Point pt(int i)
  {
    if (i==0)
      return tm->verts[p1];
    else if (i==1)
      return tm->verts[p2];
    else 
      return tm->verts[p3];
  }
};	       


class MeshedNormedTri : public MeshedTri {
  int n2,n3;

 public:
  MeshedNormedTri(Material* matl, TriMesh* tm, 
		  const int& p1, const int& p2, const int& p3,
		  const int& _vn1, const int& _vn2, const int& _vn3);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual Vector normal() 
    {
      Vector v1(tm->verts[p2]-tm->verts[p1]);
      Vector v2(tm->verts[p3]-tm->verts[p1]);
      Vector n=Cross(v1, v2);

      return n;
    }
};

class MeshedColoredTri : public MeshedTri, public Material {

 public:
  MeshedColoredTri(TriMesh* tm, 
		   const int& p1, const int& p2, const int& p3,
		   const int& n1);
  MeshedColoredTri() { tm = 0; } // for Pio
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, MeshedColoredTri*&);
};

} // end namespace rtrt

#endif

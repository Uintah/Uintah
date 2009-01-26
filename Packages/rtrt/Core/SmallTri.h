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



#ifndef SMALLTRI_H
#define SMALLTRI_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Vector.h>

namespace rtrt {
class SmallTri;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::SmallTri*&);
}

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Point;
using SCIRun::Transform;
using SCIRun::AffineCombination;
using SCIRun::Cross;

class Rect;

class SmallTri : public Object {
  Point p1, p2, p3;
  Vector vn1, vn2, vn3;
  double d;
  bool bad;
public:
  inline bool isbad() {
    return bad;
  }
  SmallTri(Material* matl, const Point& p1, const Point& p2, const Point& p3);
  SmallTri(Material* matl, const Point& p1, const Point& p2, const Point& p3,
      const Vector& vn1, const Vector& vn2, const Vector& vn3,
      bool check_badness = true);
  virtual ~SmallTri();

  SmallTri() : Object(0) {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, SmallTri*&);

  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual Vector normal(const Point&, const HitInfo& hit);
  Vector normal()
    {    
      Vector v1(p2-p1);
      Vector v2(p3-p1);
      Vector n=Cross(v2, v1);

      double l = n.length2();
      if (l > 1.e-16) {
	bad = false;
	n *= 1/sqrt(l);
      }

      return n;
    }

  virtual void compute_bounds(BBox&, double offset);

  // returns a new rect that combines me and tri if we form a rect, else NULL
  Rect * pairMeUp( SmallTri * tri );

  Point centroid()
  {
    double one_third = 1./3.;

    return AffineCombination(p1,one_third,
			     p2,one_third,
			     p3,one_third);
  }
  Point pt(int i)
  {
    if (i==0)
      return p1;
    else if (i==1)
      return p2;
    else 
      return p3;
  }
	       
  inline SmallTri copy_transform(Transform& T)
  {

    Point tp1 = T.project(p1);
    Point tp2 = T.project(p2);
    Point tp3 = T.project(p3);
	  
    Vector tvn1 = T.project_normal(vn1);
    Vector tvn2 = T.project_normal(vn2);
    Vector tvn3 = T.project_normal(vn3);

    if (!isbad()) {
      tvn1.normalize();
      tvn2.normalize();
      tvn3.normalize();
    }

    return SmallTri(this->get_matl(),
		    tp1,tp2,tp3,
		    tvn1,tvn2,tvn3,false);
  }

  void transform(Transform& T)
  {
    *this = copy_transform(T);
  }
};

} // end namespace rtrt

#endif

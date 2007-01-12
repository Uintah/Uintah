
#ifndef TEXTUREDTRI_H
#define TEXTUREDTRI_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/UV.h>

namespace rtrt {
class TexturedTri;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::TexturedTri*&);
}

namespace rtrt {

class Rect;

class TexturedTri : public Object, public UVMapping {

 protected:
  Vector ngu,ngv;   // the normalized 2D basis for this triangle's
  // object space
  Vector ngungv;    // ngu cross ngv
  double lngu,lngv; // length of the above basis vectors
  Vector ntu,ntv;   // the normalized 2D basis for this triangle's
  // texture space
  double lntu,lntv; // length of the above basis vectors
  Point p1, p2, p3; // object space vertices
  Vector vn1, vn2, vn3; // vertex normals
  Point t1, t2, t3; // texture vertices (map to p1, p2, and p3 respectively)
  Vector n;         // the normal vector
  double d;
  Vector e1p, e2p, e3p;
  Vector e1, e2, e3;
  double e1l, e2l, e3l;
  bool bad;
 public:
  inline bool isbad() {
    return bad;
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
 
  TexturedTri(Material* matl, const Point&, const Point&, const Point&);
  TexturedTri(Material* matl, const Point& p1, const Point& p2, 
	      const Point& p3, const Vector& vn1, const Vector& vn2, 
	      const Vector& vn3);
  virtual ~TexturedTri();

  TexturedTri() : Object(0), UVMapping() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, TexturedTri*&);

  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual void light_intersect(Ray& ray, HitInfo& hit, Color& atten,
			       DepthStats* st, PerProcessorContext* ppc);
  virtual void softshadow_intersect(Light* light, Ray& ray,
				    HitInfo& hit, double dist, Color& atten,
				    DepthStats* st, PerProcessorContext* ppc);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void uv(UV& uv, const Point&, const HitInfo& hit);
  virtual void set_texcoords(const Point&, const Point&, const Point&);
  
  // returns a new rect that combines me and tri if we form a rect, else NULL
  Rect * pairMeUp( TexturedTri * tri );
  inline TexturedTri copy_transform(Transform& T)
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
      
      TexturedTri copy_tri(this->get_matl(),
                           tp1,tp2,tp3,
                           tvn1,tvn2,tvn3);

      copy_tri.set_texcoords(t1,t2,t3);

      return copy_tri;
    }
  
  void transform(Transform& T)
    {
      *this = copy_transform(T);
      this->set_uvmapping(this);
    }
};

} // end namespace rtrt

#endif



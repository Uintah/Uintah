
#include <Packages/rtrt/Core/MeshedTri.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Rect.h>

#include <Core/Math/MiscMath.h>
#include <Core/Thread/Mutex.h>

#include <iostream>
#include <vector>

#include <stdio.h>

using namespace rtrt;
using namespace SCIRun;
using std::cerr;
using std::vector;

MeshedTri::MeshedTri(Material* matl, TriMesh* tm, 
		     const int& p1, const int& p2, const int& p3,
		     const int& _vn1)
  : Object(matl), tm(tm), p1(p1), p2(p2), p3(p3), n1(_vn1)
{
}

MeshedTri::~MeshedTri()
{
}

// I changed the epsilon to 1e-9 to avoid holes in the bunny -- Bill

void MeshedTri::intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*)
{
    st->tri_isect++;
    
    Point anchor = tm->verts[p1];
    Vector e1(tm->verts[p2]-anchor);
    Vector e2(tm->verts[p3]-anchor);
    Vector dir(ray.direction());
    Vector o(anchor-ray.origin());

    Vector e1e2(Cross(e1, e2));
    double det=Dot(e1e2, dir);
    if(det>1.e-9 || det < -1.e-9){
	double idet=1./det;

	Vector DX(Cross(dir, o));
	double A=-Dot(DX, e2)*idet;
	if(A>0.0 && A<1.0){
	    double B=Dot(DX, e1)*idet;
	    if(B>0.0 && A+B<1.0){
		double t=Dot(e1e2, o)*idet;
		if (hit.hit(this, t)) {
		  double* uv = (double *)hit.scratchpad;
		  uv[0] = B;
		  uv[1] = A;
		}
		st->tri_hit++;
	    }
	}
    }
}

void MeshedTri::light_intersect(Ray& ray, HitInfo& hit, Color&,
			  DepthStats* st, PerProcessorContext*)
{
  st->tri_isect++;

  Point anchor = tm->verts[p1];
  Vector e1(tm->verts[p2]-anchor);
  Vector e2(tm->verts[p3]-anchor);
  Vector dir(ray.direction());
  Vector o(anchor-ray.origin());

  Vector e1e2(Cross(e1, e2));
  double det=Dot(e1e2, dir);
  if(det>1.e-9 || det < -1.e-9){
    double idet=1./det;
    double t=Dot(e1e2, o)*idet;
    if(t<hit.min_t){
      Vector DX(Cross(dir, o));
      double A=-Dot(DX, e2)*idet;
      if(A>0.0 && A<1.0){
	double B=Dot(DX, e1)*idet;
	if(B>0.0 && A+B<1.0){
	  hit.shadowHit(this, t);
	  st->tri_hit++;
	}
      }
    }
  }
}

Vector MeshedTri::normal(const Point&, const HitInfo&)
{
  return tm->norms[n1];
}

void MeshedTri::compute_bounds(BBox& bbox, double /*offset*/)
{
    bbox.extend(tm->verts[p1]);
    bbox.extend(tm->verts[p2]);
    bbox.extend(tm->verts[p3]);
}

MeshedNormedTri::MeshedNormedTri(Material* matl, TriMesh* tm, 
			   const int& p1, const int& p2, const int& p3,
			   const int& _vn1, const int& _vn2, const int& _vn3)
  : MeshedTri(matl,tm,p1,p2,p3,_vn1), n2(_vn2), n3(_vn3)
{
}

Vector MeshedNormedTri::normal(const Point&, const HitInfo& hitinfo)
{
  double *uv = (double *)hitinfo.scratchpad;
  double beta = uv[1];
  double gamma = uv[0];

  Vector norm((1.-beta-gamma)*tm->norms[n1] + 
	      beta*tm->norms[n2] + 
	      gamma*tm->norms[n3]);

  norm.normalize();

  return norm;
}

SCIRun::Persistent* meshedtri_maker() {
  return new MeshedColoredTri();
}

// initialize the static member type_id
SCIRun::PersistentTypeID MeshedColoredTri::type_id("MeshedColoredTri", "Object", meshedtri_maker);

const int MESHEDCOLOREDTRI_VERSION = 1;

void 
MeshedColoredTri::io(SCIRun::Piostream &str)
{
  str.begin_class("Instance", MESHEDCOLOREDTRI_VERSION);
  Object::io(str);
  Material::io(str);
//    SCIRun::Pio(str, tm);   // IS this right????
  SCIRun::Pio(str, p1);
  SCIRun::Pio(str, p2);
  SCIRun::Pio(str, p3);
  SCIRun::Pio(str, n1);
  str.end_class();
}

namespace SCIRun {

void Pio( Piostream& stream, rtrt::MeshedColoredTri*& obj )
{
  Persistent* pobj=obj;
  stream.io(pobj, rtrt::MeshedColoredTri::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::MeshedColoredTri*>(pobj);
    ASSERT(obj != 0)
  }
}
}

MeshedColoredTri::MeshedColoredTri(TriMesh* tm, 
				   const int& p1, const int& p2, const int& p3,
				   const int& n1)
  : MeshedTri(this,tm,p1,p2,p3,n1)
{
  
}

void MeshedColoredTri::shade(Color& result, const Ray& ray,
			     const HitInfo& hit, int depth,
			     double atten, const Color& accumcolor,
			     Context* cx) 
{
  // extract barycoords
  double* uv = (double *)hit.scratchpad;
  double a = uv[0];
  double b = uv[1];
  // blend colors;
  
  Color diff_color = 
    (1.-a-b)*tm->colors[p1] + 
    a*tm->colors[p2] + 
    b*tm->colors[p3];
  
  Color spec_color(.2,.2,.2);
  phongshade(result,diff_color,spec_color,80,0,ray,hit,depth,atten,
	     accumcolor,cx);
}


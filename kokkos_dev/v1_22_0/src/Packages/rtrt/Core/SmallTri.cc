
#include <Packages/rtrt/Core/SmallTri.h>
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

Persistent* tri_maker() {
  return new SmallTri();
}

// initialize the static member type_id
PersistentTypeID SmallTri::type_id("SmallTri", "Object", tri_maker);


SmallTri::SmallTri(Material* matl, const Point& p1, const Point& p2,
	 const Point& p3)
    : Object(matl), p1(p1), p2(p2), p3(p3)
{
    Vector v1(p2-p1);
    Vector v2(p3-p1);
    Vector n=Cross(v2, v1);

    double l = n.length2();
    if (l > 1.e-16) {
      bad = false;
      n *= 1/sqrt(l);
    } else {
      //	printf("BAD NORMAL!\n");
      bad = true;
      return;
    }
    vn1 = n;
    vn2 = n;
    vn3 = n;
    d=Dot(n, p1);
}

SmallTri::SmallTri(Material* matl, const Point& p1, const Point& p2,
	 const Point& p3,
	 const Vector& _vn1, const Vector& _vn2, const Vector& _vn3,
	 bool check_badness)
  : Object(matl), p1(p1), p2(p2), p3(p3)
{
    Vector v1(p2-p1);
    Vector v2(p3-p1);
    Vector n=Cross(v1, v2);
    double l = n.length2();
    if ((!check_badness && l > 0) || l > 1.e-16) {
      bad = false;
      n *= 1/sqrt(l);
    } else {
      //	printf("BAD NORMAL!\n");
      bad = true;
      return;
    }
    d=Dot(n, p1);

    if (_vn1.length2() > 1E-12 &&
	_vn2.length2() > 1E-12 && 
	_vn3.length2() > 1E-12) {

      vn1 = _vn1.normal();
      if (Dot(vn1,n) < 0)
	vn1 = -vn1;
      vn2 = _vn2.normal();
      if (Dot(vn2,n) < 0)
	vn2 = -vn2;
      vn3 = _vn3.normal();
      if (Dot(vn3,n) < 0)
	vn3 = -vn3;
    } else {
      vn1 = vn2 = vn3 = n;
    }
}

SmallTri::~SmallTri()
{
}

// I changed the epsilon to 1e-9 to avoid holes in the bunny -- Bill

void SmallTri::intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*)
{
    st->tri_isect++;
    Vector e1(p2-p1);
    Vector e2(p3-p1);
    Vector dir(ray.direction());
    Vector o(p1-ray.origin());

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

Rect *
SmallTri::pairMeUp( SmallTri * tri )
{
  if( tri == NULL ) return NULL;

  if (isbad() || tri->isbad())
    return NULL;

  double dotVal = Dot( tri->normal(), normal());
  if( dotVal > 0.999 || dotVal < -.999 ) { // we are coplanar

    double s1_t1 = (p1 - p2).length2();
    double s2_t1 = (p1 - p3).length2();
    double s3_t1 = (p2 - p3).length2();

    double s1_t2 = (tri->p1 - tri->p2).length2();
    double s2_t2 = (tri->p1 - tri->p3).length2();
    double s3_t2 = (tri->p2 - tri->p3).length2();

    vector<Point *> notDiagT1;
    vector<Point *> notDiagT2;
    double lenT1Hyp = 0;
    double lenT1Others = 0;
    double lenT2Hyp = 0;

    Point centerHypT1; // Center of Hypotenuse
    Point centerSaT1;  // Center of Side A
    Point centerSbT1;  // Center of Side B

    if( s1_t1 > s2_t1 || s1_t1 > s3_t1 ) {
      notDiagT1.push_back( &p3 );
      lenT1Hyp = s1_t1;
      lenT1Others = s2_t1 + s3_t1;
      centerHypT1 = p1 + (p2 - p1)/2;
      centerSaT1  = p3 + (p1 - p3)/2;
      centerSbT1  = p3 + (p2 - p3)/2;
    } else if( s2_t1 < s3_t1 ) {
      notDiagT1.push_back( &p2 );
      lenT1Hyp = s2_t1;
      lenT1Others = s1_t1 + s3_t1;
      centerHypT1 = p1 + (p3 - p1)/2;
      centerSaT1  = p2 + (p1 - p2)/2;
      centerSbT1  = p2 + (p3 - p2)/2;
    } else {
      notDiagT1.push_back( &p1 );
      lenT1Hyp = s3_t1;
      lenT1Others = s1_t1 + s2_t1;
      centerHypT1 = p2 + (p3 - p2)/2;
      centerSaT1  = p1 + (p2 - p1)/2;
      centerSbT1  = p1 + (p3 - p1)/2;
    }

    if( s1_t2 > s2_t2 || s1_t2 > s3_t2 ) {
      notDiagT2.push_back( &p3 );
      lenT2Hyp = s1_t2;
    } else if( s2_t2 < s3_t2 ) {
      notDiagT2.push_back( &p2 );
      lenT2Hyp = s2_t2;
    } else {
      notDiagT2.push_back( &p1 );
      lenT2Hyp = s3_t2;
    }

#if 0
    cout << "lengths t1: " << s1_t1 << ", " 
	 << s2_t1 << ", " << s3_t1 << "\n";
    cout << "lengths t2: " << s1_t2 << ", " 
	 << s2_t2 << ", " << s3_t2 << "\n";
    cout << "hyps: " << lenT1Hyp << ", " << lenT2Hyp << "\n";
    cout << "len others: " << lenT1Others << "\n";
#endif

    // If Hypotenuses are the same length 
    if( fabs( lenT1Hyp - lenT2Hyp ) < 0.0001 ) {
      //cout << "same\n";
      // And if triangle(s) is a right triangle (ie: H^2 = A^2 + B^2).
      if( fabs( lenT1Others - lenT1Hyp ) < 0.0001 ) {
	return new Rect( get_matl(), centerHypT1, 
			 centerHypT1 - centerSaT1,
			 centerHypT1 - centerSbT1 );
      }
    }
  }
  return NULL;
}

void SmallTri::light_intersect(Ray& ray, HitInfo& hit, Color&,
			  DepthStats* st, PerProcessorContext*)
{
  st->tri_isect++;
  Vector e1(p2-p1);
  Vector e2(p3-p1);
  Vector dir(ray.direction());
  Vector o(p1-ray.origin());

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

Vector SmallTri::normal(const Point&, const HitInfo& hitinfo)
{
  double *uv = (double *)hitinfo.scratchpad;
  double beta = uv[1];
  double gamma = uv[0];

//    printf("beta: %lf gamma: %lf\n",beta,gamma);

  if (isbad())
    return Vector(1,0,0);

  Vector norm((1.-beta-gamma)*vn1 + beta*vn2 + gamma*vn3);

  norm.normalize();

  return norm;
}

void SmallTri::compute_bounds(BBox& bbox, double /*offset*/)
{
#if 0
    Vector e1(p3-p2);
    Vector e2(p1-p3);
    Vector e3(p2-p1);
    e1.normalize();
    e2.normalize();
    e3.normalize();
    double sina3=Abs(Cross(e1, e2).length());
    double sina2=Abs(Cross(e3, e1).length());
    double sina1=Abs(Cross(e2, e3).length());
    Point p3p(p3+(e1-e2)*(offset/sina3));
    Point p2p(p2+(e3-e1)*(offset/sina2));
    Point p1p(p1+(e2-e3)*(offset/sina1));
    Vector dz(n*offset*0);
    bbox.extend(p3p+dz);
    bbox.extend(p3p-dz);
    bbox.extend(p2p+dz);
    bbox.extend(p2p-dz);
    bbox.extend(p1p+dz);
    bbox.extend(p1p-dz);
    if(isnan(p1.z()) || isnan(p2.z()) || isnan(p3.z())
       || isnan(p1p.z()) || isnan(p2p.z()) || isnan(p3p.z())){
      cerr << "p1=" << p1 << ", p2=" << p2 << ", p3=" << p3 << '\n';
      cerr << "p1p=" << p1p << ", p2p=" << p2p << ", p3p=" << p3p << '\n';
      cerr << "bbox=" << bbox.min() << ", " << bbox.max() << '\n';
    }
#else
    bbox.extend(p1);
    bbox.extend(p2);
    bbox.extend(p3);
#endif
}

const int TRI_VERSION = 1;

void 
SmallTri::io(SCIRun::Piostream &str)
{
  str.begin_class("SmallTri", TRI_VERSION);
  Object::io(str);
  Pio(str, p1);
  Pio(str, p2);
  Pio(str, p3);
  Pio(str, vn1);
  Pio(str, vn2);
  Pio(str, vn3);
  Pio(str, d);
  Pio(str, bad);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::SmallTri*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::SmallTri::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::SmallTri*>(pobj);
    ASSERT(obj != 0)
  }
}
} // end namespace SCIRun

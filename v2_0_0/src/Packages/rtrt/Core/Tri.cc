
#include <Packages/rtrt/Core/Tri.h>
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

Persistent* rtrttri_maker() {
  return new Tri();
}

// initialize the static member type_id
PersistentTypeID Tri::type_id("Tri", "Object", rtrttri_maker);


Tri::Tri(Material* matl, const Point& p1, const Point& p2,
	 const Point& p3)
    : Object(matl), p1(p1), p2(p2), p3(p3)
{
    Vector v1(p2-p1);
    Vector v2(p3-p1);
    n=Cross(v1, v2);

#if 1
    double l = n.length2();
    if (l > 1.e-16) {
      bad = false;
      n *= 1/sqrt(l);
    } else {
//  	printf("BAD NORMAL!\n");
      bad = true;
      return;
    }
    

#else
    double l=n.normalize();
    if(l<1.e-8){
	cerr << "Bad normal? " << n << '\n';
	cerr << "l=" << l << '\n';
	cerr << "before: " << Cross(v1, v2) << ", after: " << n << '\n';
	cerr << "p1=" << p1 << ", p2=" << p2 << ", p3=" << p3 << '\n';
	bad=true;
    } else {
	bad=false;
    }
#endif
    vn1 = n;
    vn2 = n;
    vn3 = n;

    d=Dot(n, p1);
    e1=p3-p2;
    e2=p1-p3;
    e3=p2-p1;
    e1l=e1.normalize();
    e2l=e2.normalize();
    e3l=e3.normalize();
    e1p=Cross(e1, n);
    e2p=Cross(e2, n);
    e3p=Cross(e3, n);
}

Tri::Tri(Material* matl, const Point& p1, const Point& p2,
	 const Point& p3,
	 const Vector& _vn1, const Vector& _vn2, const Vector& _vn3,
	 bool check_badness)
  : Object(matl), p1(p1), p2(p2), p3(p3)
{
    Vector v1(p2-p1);
    Vector v2(p3-p1);
    n=Cross(v1, v2);
#if 1
    double l = n.length2();
    if ((!check_badness && l > 0) || l > 1.e-16) {
      bad = false;
      n *= 1/sqrt(l);
    } else {
      //	printf("BAD NORMAL!\n");
      bad = true;
      return;
    }
#else
    double l=n.normalize();
    if(l<1.e-8){
	cerr << "Bad normal? " << n << '\n';
	cerr << "l=" << l << '\n';
	cerr << "before: " << Cross(v1, v2) << ", after: " << n << '\n';
	cerr << "p1=" << p1 << ", p2=" << p2 << ", p3=" << p3 << '\n';
	bad=true;
    } else {
	bad=false;
    }
#endif
    d=Dot(n, p1);
    e1=p3-p2;
    e2=p1-p3;
    e3=p2-p1;
    e1l=e1.normalize();
    e2l=e2.normalize();
    e3l=e3.normalize();
    e1p=Cross(e1, n);
    e2p=Cross(e2, n);
    e3p=Cross(e3, n);

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

Tri::~Tri()
{
}

// I changed the epsilon to 1e-9 to avoid holes in the bunny -- Bill

void Tri::intersect(Ray& ray, HitInfo& hit, DepthStats* st,
		    PerProcessorContext*)
{
  if( bad )
    {
      cout << "warning, intersect called on bad tri\n";
      return;
    }
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
Tri::pairMeUp( Tri * tri )
{
  if( tri == NULL ) return NULL;

  double dotVal = Dot( tri->n, n );
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

void Tri::light_intersect(Ray& ray, HitInfo& hit, Color&,
			  DepthStats* st, PerProcessorContext*)
{
  if( bad )
    {
      cout << "warning, light intersect called on bad tri\n";
      return;
    }

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

Vector Tri::normal(const Point&, const HitInfo& hitinfo)
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

// I changed epsilon to 1e-9 to avoid holes in the bunny! -- Bill

void Tri::softshadow_intersect(Light* light, Ray& ray,
			       HitInfo&, double dist, Color& atten,
			       DepthStats* st, PerProcessorContext*)
{
  if( bad )
    {
      cout << "warning, softshadow intersect called on bad tri\n";
      return;
    }

    st->tri_light_isect++;
    Vector dir(ray.direction());
    Point orig(ray.origin());
    double dt=Dot(dir, n);
    if(dt < 1.e-9 && dt > -1.e-9)
	return;
    double t=(d-Dot(n, orig))/dt;
    if(t<1.e-9)
	return;
    if(t>dist)
	return;
    Point p(orig+dir*t);

    double delta=light->radius*t/dist/2;
    if(delta < .0001){
	return;
    }

    Vector pp1(p-p2);
    double c1=Dot(pp1, e1);
    if(c1<-delta || c1>e1l+delta)
	return;
    double d1=Dot(pp1, e1p);

    Vector pp2(p-p3);
    double c2=Dot(pp2, e2);
    if(c2<-delta || c2>e2l+delta)
	return;
    double d2=Dot(pp2, e2p);

    Vector pp3(p-p1);
    double c3=Dot(pp3, e3);
    if(c3<-delta || c3>e3l+delta)
	return;
    double d3=Dot(pp3, e3p);

#if 0

    if(d1>delta || d2>delta || d3>delta)
    	return;
    if(d1<-delta && d2<-delta && d3<-delta){
	atten = Color(0,0,0);
	return;
    }
    

#define MODE 1
#if MODE==0
    double sum=0;
    if(d1>0)
	sum+=d1;
    if(d2>0)
	sum+=d2;
    if(d3>0)
	sum+=d3;
    if(sum>delta)
	return;
    double gg=sum/delta;
#else
#if MODE==1
    double sum=0;
    if(d1>0)
	sum+=d1*d1;
    if(d2>0)
	sum+=d2*d2;
    if(d3>0)
	sum+=d3*d3;
    if(sum>delta*delta)
	return;
    double gg=sqrt(sum)/delta;
#else
#if MODE==2
    double sum=d1;
    if(d2>sum)
	sum=d2;
    if(d3>sum)
	sum=d3;
    double gg=sum/delta;
#else
#if MODE==4
    atten=Color(0,0, 0);
    return;
#else
#error "Illegal mode"
#endif
#endif
#endif
#endif
#else
    double tau;
    if(d1>0){
	if(c1<0){
	    tau=(p-p2).length();
	} else if(c1>e1l){
	    tau=(p-p3).length();
	} else {
	    tau=d1;
	}
    } else if(d2>0){
	if(c2<0){
	    tau=(p-p3).length();
	} else if(c2>e2l){
	    tau=(p-p1).length();
	} else {
	    tau=d2;
	}
    } else if(d3>0){
	if(c3<0){
	    tau=(p-p1).length();
	} else if(c3>e3l){
	    tau=(p-p2).length();
	} else {
	    tau=d3;
	}
    } else {
	// Inside
	atten=Color(0,0,0);
	st->tri_light_hit++;
	return;
    }
    double gg=tau/delta;
    if(gg>1){
	return;
    }
#endif

    st->tri_light_penumbra++;
    double g=3*gg*gg-2*gg*gg*gg;
    atten=g<atten.luminance()?Color(g,g,g):atten;
}

void Tri::compute_bounds(BBox& bbox, double /*offset*/)
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
Tri::io(SCIRun::Piostream &str)
{
  str.begin_class("Tri", TRI_VERSION);
  Object::io(str);
  Pio(str, p1);
  Pio(str, p2);
  Pio(str, p3);
  Pio(str, vn1);
  Pio(str, vn2);
  Pio(str, vn3);
  Pio(str, n);
  Pio(str, d);
  Pio(str, e1p);
  Pio(str, e2p);
  Pio(str, e3p);
  Pio(str, e1);
  Pio(str, e2);
  Pio(str, e3);
  Pio(str, e1l);
  Pio(str, e2l);
  Pio(str, e3l);
  Pio(str, bad);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Tri*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Tri::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Tri*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun

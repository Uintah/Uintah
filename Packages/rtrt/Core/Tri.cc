
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/MiscMath.h>
#include <Core/Thread/Mutex.h>
#include <Packages/rtrt/Core/Stats.h>
#include <iostream>

using namespace rtrt;
using std::cerr;

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
      bad = true;
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
}

Tri::~Tri()
{
}

// I changed the epsilon to 1e-9 to avoid holes in the bunny -- Bill

void Tri::intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
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
		hit.hit(this, t);
		st->tri_hit++;
	    }
	}
    }
}

Vector Tri::normal(const Point&, const HitInfo&)
{
    return n;
}

// I changed epsilon to 1e-9 to avoid holes in the bunny! -- Bill

void Tri::light_intersect(Light* light, const Ray& ray,
			  HitInfo&, double dist, Color& atten,
			  DepthStats* st, PerProcessorContext*)
{
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

void Tri::compute_bounds(BBox& bbox, double offset)
{
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
}

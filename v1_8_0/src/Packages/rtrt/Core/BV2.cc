
#include <Packages/rtrt/Core/BV2.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <iostream>

using namespace rtrt;
using namespace std;

using SCIRun::Thread;
using SCIRun::Time;

BV2::BV2(Object* obj)
    : Object(0), obj(obj)
{
}

BV2::~BV2()
{
}

namespace rtrt {
  struct BV2Tree {
    double bv[6];
    BV2Tree* left;
    BV2Tree* right;
    Object* obj;
    BV2Tree(Object* obj, double maxradius);
    BV2Tree(BV2Tree* left, BV2Tree* right);
    void intersect(Ray& ray, const Point& orig, const Vector& idir, HitInfo& hit,
		   DepthStats* st, PerProcessorContext* ppc);
    ~BV2Tree();
  };
} // end namespace rtrt

BV2Tree::BV2Tree(Object* obj, double maxradius)
    : left(0), right(0), obj(obj)
{
    if (obj == 0) ASSERTFAIL("Trying to create a BV1 with no objects");
    BBox bbox;
    obj->compute_bounds(bbox, maxradius);
    Point min(bbox.min());
    Point max(bbox.max());
    bv[0]=min.x()-1.e-6;
    bv[1]=min.y()-1.e-6;
    bv[2]=min.z()-1.e-6;
    bv[3]=max.x()+1.e-6;
    bv[4]=max.y()+1.e-6;
    bv[5]=max.z()+1.e-6;
}

BV2Tree::BV2Tree(BV2Tree* left, BV2Tree* right)
    : left(left), right(right), obj(0)
{
    bv[0]=Min(left->bv[0], right->bv[0]);
    bv[1]=Min(left->bv[1], right->bv[1]);
    bv[2]=Min(left->bv[2], right->bv[2]);
    bv[3]=Max(left->bv[3], right->bv[3]);
    bv[4]=Max(left->bv[4], right->bv[4]);
    bv[5]=Max(left->bv[5], right->bv[5]);
}

BV2Tree* BV2::make_tree(int nprims, Object** prims, double maxradius)
{
    if(nprims==1){
	return new BV2Tree(prims[0], maxradius);
    } else {
	int n1=nprims/2;
	int n2=nprims-n1;
	return new BV2Tree(make_tree(n1, prims, maxradius),
			   make_tree(n2, prims+n1, maxradius));
    }
}

void BV2::preprocess(double maxradius, int& pp_offset, int& scratchsize)
{
    obj->preprocess(maxradius, pp_offset, scratchsize);
    double time=Time::currentSeconds();

    Array1<Object*> prims;
    obj->collect_prims(prims);
    cerr << "Collect prims took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();
    if(prims.size()==0){
	top=0;
	return;
    }
    top=make_tree(prims.size(), &prims[0], maxradius);
    if(maxradius == 0){
	top_light=top;
    } else {
	top_light=make_tree(prims.size(), &prims[0], maxradius);
    }
}

void BV2Tree::intersect(Ray& ray, const Point& orig, const Vector& idir,
			HitInfo& hit,
			DepthStats* st, PerProcessorContext* ppc)
{
    st->bv_total_isect++;
    //Point orig(ray.origin());
    double MINa=1.e-6, MAXa=hit.min_t;
    double x0a, x1a;
    if(idir.x() > 0){
	x0a=idir.x()*(bv[0]-orig.x());
	x1a=idir.x()*(bv[3]-orig.x());
    } else {
	x0a=idir.x()*(bv[3]-orig.x());
	x1a=idir.x()*(bv[0]-orig.x());
    }
    if(x0a>MINa)
	MINa=x0a;
    if(x1a<MAXa)
	MAXa=x1a;
    if(MAXa<MINa){
	return;
    }
    double y0a, y1a;
    if(idir.y() > 0){
	y0a=idir.y()*(bv[1]-orig.y());
	y1a=idir.y()*(bv[4]-orig.y());
    } else {
	y0a=idir.y()*(bv[4]-orig.y());
	y1a=idir.y()*(bv[1]-orig.y());
    }
    if(y0a>MINa)
	MINa=y0a;
    if(y1a<MAXa)
	MAXa=y1a;
    if(MAXa<MINa){
	return;
    }
    double z0a, z1a;
    if(idir.z() > 0){
	z0a=idir.z()*(bv[2]-orig.z());
	z1a=idir.z()*(bv[5]-orig.z());
    } else {
	z0a=idir.z()*(bv[5]-orig.z());
	z1a=idir.z()*(bv[2]-orig.z());
    }
    if(z0a>MINa)
	MINa=z0a;
    if(z1a<MAXa)
	MAXa=z1a;
    if(MAXa > MINa){
	if(MAXa > 1.e-6){
	    if(obj){
		st->bv_prim_isect++;		
		obj->intersect(ray, hit, st, ppc);
	    } else {
		left->intersect(ray, orig, idir, hit, st, ppc);
		right->intersect(ray, orig, idir, hit, st, ppc);
	    }
	}
    }
}

void BV2::intersect(Ray& ray, HitInfo& hit,
		    DepthStats* st, PerProcessorContext* ppc)
{
    Point orig(ray.origin());    
    Vector dir(ray.direction());
    Vector idir(1./dir.x(), 1./dir.y(), 1./dir.z());
    top->intersect(ray, orig, idir, hit, st, ppc);
}

void BV2::light_intersect(Ray&, HitInfo&, Color&,
			  DepthStats*, PerProcessorContext*)
{
    cerr << "BV2::light_intersect not finished\n";
}

void BV2::softshadow_intersect(Light*, Ray&,
			  HitInfo&, double, Color&,
			  DepthStats*, PerProcessorContext*)
{
    cerr << "BV2::softshadow_intersect not finished\n";
}

void BV2::animate(double t, bool& changed)
{
    obj->animate(t, changed);
}

void BV2::collect_prims(Array1<Object*>& prims)
{
    prims.add(this);
}

void BV2::compute_bounds(BBox& bbox, double offset)
{
    obj->compute_bounds(bbox, offset);
}

Vector BV2::normal(const Point&, const HitInfo&)
{
    cerr << "Error: BV2 normal should not be called!\n";
    return Vector(0,0,0);
}

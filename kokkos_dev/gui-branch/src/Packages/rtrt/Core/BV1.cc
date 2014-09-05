
#include "BV1.h"
#include "BBox.h"
#include "Array1.h"
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include "Ray.h"
#include "Stats.h"
#include "HitInfo.h"
#include <iostream>

using namespace rtrt;
using SCIRun::Thread;
using SCIRun::Time;

namespace rtrt {
  struct BV1Tree {
    double* slabs;
    int primStart;
    BBox bbox;
    Array1<Object*> prims;
    ~BV1Tree();
  };
} // end namespace rtrt

BV1Tree::~BV1Tree()
{
    delete slabs;
}

BV1::BV1(Object* obj)
    : Object(0), obj(obj)
{
}

BV1::~BV1()
{
    delete obj;
    if(normal_tree != light_tree)
	delete light_tree;
    delete normal_tree;
}

void BV1::preprocess(double maxradius, int& pp_offset, int& scratchsize)
{
    obj->preprocess(maxradius, pp_offset, scratchsize);
    normal_tree=make_tree(0);
    if(maxradius == 0)
	light_tree=normal_tree;
    else
	light_tree=make_tree(maxradius);
}

BV1Tree* BV1::make_tree(double maxradius)
{
    BV1Tree* tree=new BV1Tree();
    double time=Time::currentSeconds();

    obj->collect_prims(tree->prims);
    cerr << "Collect prims took " << Time::currentSeconds()-time << " seconds\n";
    if(tree->prims.size() == 1){
	cerr << "BV1 does not work with a single primitive!\n";
	Thread::exitAll(1);
    }
    time=Time::currentSeconds();
    int nnodes=2*tree->prims.size()-1;
    tree->slabs=new double[6*nnodes];

    tree->primStart=nnodes-tree->prims.size();
    double* slab=tree->slabs+6*tree->primStart;
    for(int i=0;i<tree->prims.size();i++){
	Object* obj=tree->prims[i];
	BBox bbox;
	obj->compute_bounds(bbox, maxradius);
	Point min(bbox.min());
	Point max(bbox.max());
	slab[0]=min.x()-1.e-6;
	slab[1]=min.y()-1.e-6;
	slab[2]=min.z()-1.e-6;
	slab[3]=max.x()+1.e-6;
	slab[4]=max.y()+1.e-6;
	slab[5]=max.z()+1.e-6;
	slab+=6;
    }
    cerr << "Compute bounds took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();

    cerr << "There are " << tree->prims.size() << " prims\n";
    if(tree->prims.size()==0)
	return tree;

    make_tree(tree->prims.size(), &tree->prims[0], tree->slabs+6*tree->primStart);
    cerr << "Make tree took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();

    finishit(tree->slabs, tree->prims, tree->primStart);
    cerr << "Finish(1) took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();
    return tree;
}

void BV1::finishit(double* slabs, Array1<Object*>& prims,
		     int primStart)
{
    int nnodes=2*prims.size()-1;
    int split=1;
    while(split<prims.size())
	split*=2;
    int start=nnodes-split+1;
    int idx=0;
    Array1<Object*> tmp(prims);
    double* tmpslabs=new double[6*prims.size()];
    for(int i=0;i<6*prims.size();i++)
	tmpslabs[i]=slabs[i+6*primStart];
    int idx2=6*primStart;
    for(int i=start;i<prims.size();i++){
	prims[idx++]=tmp[i];
	for(int j=0;j<6;j++)
	    slabs[idx2++]=tmpslabs[6*i+j];
    }
    for(int i=0;i<start;i++){
	prims[idx++]=tmp[i];
	for(int j=0;j<6;j++)
	    slabs[idx2++]=tmpslabs[6*i+j];
    }
    delete[] tmpslabs;
    for(int i=nnodes-2;i>=1;i-=2){
	int parent=i/2;
	double* s1=&slabs[6*i];
	double* s2=s1+6;
	double* ps=&slabs[6*parent];
	ps[0]=Min(s1[0], s2[0]);
	ps[1]=Min(s1[1], s2[1]);
	ps[2]=Min(s1[2], s2[2]);
	ps[3]=Max(s1[3], s2[3]);
	ps[4]=Max(s1[4], s2[4]);
	ps[5]=Max(s1[5], s2[5]);
    }
}

inline bool compare(double* s1, double* s2, int compare_axis)
{
    double d1=s1[compare_axis]+s1[compare_axis+3];
    double d2=s2[compare_axis]+s2[compare_axis+3];
    return d1<d2;
}

static void heapify(Object** odata, double* sdata, int n, int i, int compare_axis)
{
    int l=2*i+1;
    int r=l+1;
    int largest=i;
    if(l<n && compare(sdata+l*6, sdata+i*6, compare_axis))
	largest=l;
    if(r<n && compare(sdata+r*6, sdata+largest*6, compare_axis))
	largest=r;
    if(largest != i){
	Object* otmp=odata[i];
	odata[i]=odata[largest];
	odata[largest]=otmp;
	for(int j=0;j<6;j++){
	    double stmp=sdata[i*6+j];
	    sdata[i*6+j]=sdata[largest*6+j];
	    sdata[largest*6+j]=stmp;
	}
	heapify(odata, sdata, n, largest, compare_axis);
    }
}

static void my_qsort(Object** odata, double* sdata, int n, int compare_axis)
{
    // Sort it...
    // Build the heap...
    for(int i=n/2-1;i >= 0;i--){
	heapify(odata, sdata, n, i, compare_axis);
    }
    // Sort
    for(int i=n-1;i>0;i--){
	// Exchange 1 and i
	Object* otmp=odata[i];
	odata[i]=odata[0];
	odata[0]=otmp;
	for(int j=0;j<6;j++){
	    double stmp=sdata[i*6+j];
	    sdata[i*6+j]=sdata[j];
	    sdata[j]=stmp;
	}
	heapify(odata, sdata, i, 0, compare_axis);
    }
}

void BV1::make_tree(int nprims, Object** subprims, double* slabs){
    if(nprims==1){
	return;
    } else {
	int i;	
	int split=(nprims+1)/2;
	// Sort by X...
	my_qsort(&subprims[0], slabs, nprims, 0);
	BBox xbox1;
	for(i=0;i<split;i++)
	    xbox1.extend(&slabs[6*i]);
	BBox xbox2;
	for(i=split;i<nprims;i++)
	    xbox2.extend(&slabs[6*i]);
	    
	// Sort by Y...
	my_qsort(&subprims[0], slabs, nprims, 1);
	BBox ybox1;
	for(i=0;i<split;i++)
	    ybox1.extend(&slabs[6*i]);
	BBox ybox2;
	for(i=split;i<nprims;i++)
	    ybox2.extend(&slabs[6*i]);
	
	// Sort by Z...
	my_qsort(&subprims[0], slabs, nprims, 2);
	BBox zbox1;
	for(i=0;i<split;i++)
	    zbox1.extend(&slabs[6*i]);
	BBox zbox2;
	for(i=split;i<nprims;i++)
	    zbox2.extend(&slabs[6*i]);

	Point x1(Max(xbox1.min(), xbox2.min()));
	Point x2(Min(xbox2.max(), xbox2.max()));
	Vector ox(x2-x1);
	double overx=ox.minComponent();
	
	Point y1(Max(ybox1.min(), ybox2.min()));
	Point y2(Min(ybox2.max(), ybox2.max()));
	Vector oy(y2-y1);
	double overy=oy.minComponent();

	Point z1(Max(zbox1.min(), zbox2.min()));
	Point z2(Min(zbox2.max(), zbox2.max()));
	Vector oz(z2-z1);
	double overz=oz.minComponent();

	if(overx < overy && overx < overz){
	    my_qsort(&subprims[0], slabs, nprims, 0);
	} else if(overy < overz){
	    my_qsort(&subprims[0], slabs, nprims, 1);
	} else {
	    my_qsort(&subprims[0], slabs, nprims, 2);
	}
	make_tree(split, subprims, slabs);
	make_tree(nprims-split, subprims+split, slabs+split*6);
    }
}


inline void isect_bbox(const Point& orig, const Vector& idir,
		       double* slabs, bool& lhit, bool& rhit,
		       double maxt) {
    double MINa=1.e-6, MAXa=maxt;
    double MINb=1.e-6, MAXb=maxt;
    double x0a, x1a;
    double x0b, x1b;
    if(idir.x() > 0){
	x0a=idir.x()*(slabs[0]-orig.x());
	x1a=idir.x()*(slabs[3]-orig.x());
	x0b=idir.x()*(slabs[6]-orig.x());
	x1b=idir.x()*(slabs[9]-orig.x());
    } else {
	x0a=idir.x()*(slabs[3]-orig.x());
	x1a=idir.x()*(slabs[0]-orig.x());
	x0b=idir.x()*(slabs[9]-orig.x());
	x1b=idir.x()*(slabs[6]-orig.x());
    }
    if(x0a>MINa)
	MINa=x0a;
    if(x1a<MAXa)
	MAXa=x1a;
    if(x0b>MINb)
	MINb=x0b;
    if(x1b<MAXb)
	MAXb=x1b;
    if(MAXa<MINa && MAXb<MINb){
	lhit=rhit=false;
	return;
    }
    double y0a, y1a;
    double y0b, y1b;
    if(idir.y() > 0){
	y0a=idir.y()*(slabs[1]-orig.y());
	y1a=idir.y()*(slabs[4]-orig.y());
	y0b=idir.y()*(slabs[7]-orig.y());
	y1b=idir.y()*(slabs[10]-orig.y());
    } else {
	y0a=idir.y()*(slabs[4]-orig.y());
	y1a=idir.y()*(slabs[1]-orig.y());
	y0b=idir.y()*(slabs[10]-orig.y());
	y1b=idir.y()*(slabs[7]-orig.y());
    }
    if(y0a>MINa)
	MINa=y0a;
    if(y1a<MAXa)
	MAXa=y1a;
    if(y0b>MINb)
	MINb=y0b;
    if(y1b<MAXb)
	MAXb=y1b;
    if(MAXa<MINa && MAXb<MINb){
	lhit=rhit=false;
	return;
    }
    double z0a, z1a;
    double z0b, z1b;
    if(idir.z() > 0){
	z0a=idir.z()*(slabs[2]-orig.z());
	z1a=idir.z()*(slabs[5]-orig.z());
	z0b=idir.z()*(slabs[8]-orig.z());
	z1b=idir.z()*(slabs[11]-orig.z());
    } else {
	z0a=idir.z()*(slabs[5]-orig.z());
	z1a=idir.z()*(slabs[2]-orig.z());
	z0b=idir.z()*(slabs[11]-orig.z());
	z1b=idir.z()*(slabs[8]-orig.z());
    }
    if(z0a>MINa)
	MINa=z0a;
    if(z1a<MAXa)
	MAXa=z1a;
    if(z0b>MINb)
	MINb=z0b;
    if(z1b<MAXb)
	MAXb=z1b;
    if(MAXa > MINa){
	if(MINa > 1.e-6){
	    lhit=true;
	} else if(MAXa > 1.e-6){
	    lhit=true;
	} else {
	    lhit=false;
	}
    } else {
	lhit=false;
    }
    if(MAXb > MINb){
	if(MINb > 1.e-6){
	    rhit=true;
	} else if(MAXb > 1.e-6){
	    rhit=true;
	} else {
	    rhit=false;
	}
    } else {
	rhit=false;
    }
}

void BV1::intersect(const Ray& ray, HitInfo& hit,
		    DepthStats* st, PerProcessorContext* ppc)
{
    int idx=0;
    int sp=0;
    double* slabs=normal_tree->slabs;
    int primStart=normal_tree->primStart;
    Object** prims=&normal_tree->prims[0];
#define BSTACK_SIZE 100
    int bstack[BSTACK_SIZE];
    Point orig(ray.origin());
    Vector dir(ray.direction());
    Vector idir(1./dir.x(), 1./dir.y(), 1./dir.z());
    for(;;){
	int L=2*idx+1;
	int R=L+1;
	bool lhit, rhit;
	isect_bbox(orig, idir, &slabs[L*6], lhit, rhit, hit.min_t);

	st->bv_total_isect+=2;
	if(lhit){
	    if(rhit){
		if(L>=primStart){
		    st->bv_prim_isect+=2;
		    prims[L-primStart]->intersect(ray, hit, st, ppc);
		    prims[R-primStart]->intersect(ray, hit, st, ppc);
		    if(--sp<0)
			break;
		    idx=bstack[sp];
		} else if(R>=primStart){
		    st->bv_prim_isect++;
		    prims[R-primStart]->intersect(ray, hit, st, ppc);
		    idx=L;
		} else {
		    bstack[sp++]=R;
		    if(sp>=BSTACK_SIZE){
			cerr << "BSTACK OVERFLOW!\n";
			Thread::exitAll(-1);
		    }
		    idx=L;
		}
	    } else {
		if(L>=primStart){
		    st->bv_prim_isect++;
		    prims[L-primStart]->intersect(ray, hit, st, ppc);
		    if(--sp<0)
			break;
		    idx=bstack[sp];
		} else {
		    idx=L;
		}
	    }
	} else {
	    if(rhit){
		if(R>=primStart){
		    st->bv_prim_isect++;
		    prims[R-primStart]->intersect(ray, hit, st, ppc);
		    if(--sp<0)
			break;
		    idx=bstack[sp];
		} else {
		    idx=R;
		}
	    } else {
		if(--sp<0)
		    break;
		idx=bstack[sp];
	    }
	}
    }
}

void BV1::light_intersect(Light* light, const Ray& lightray,
			  HitInfo& hit, double dist, Color& atten,
			  DepthStats* st, PerProcessorContext* ppc)
{
    double* slabs=light_tree->slabs;
    int primStart=light_tree->primStart;
    Object** prims=&light_tree->prims[0];
#if 0
    if(shadow_cache){
	shadow_cache->light_intersect(light, lightray, hit, dist, atten, st);
	st->shadow_cache_try++;
	if(atten.luminance()<1.e-6)
	    return;
	shadow_cache=0;
	st->shadow_cache_miss++;
    }
    if(!soft_shadows){
	intersect(obj, lightray, hit, st);
	return;
    }
#endif

     int idx=0;
     int sp=0;
     int bstack[BSTACK_SIZE];
     Point orig(lightray.origin());
     Vector dir(lightray.direction());
     Vector idir(1./dir.x(), 1./dir.y(), 1./dir.z());
     for(;;){
	 int L=2*idx+1;
	 int R=L+1;
	 bool lhit, rhit;
	 isect_bbox(orig, idir, &slabs[L*6], lhit, rhit, hit.min_t);
	 st->bv_total_isect_light+=2;
	 if(lhit){
	     if(rhit){
		 if(L>=primStart){
		     prims[L-primStart]->light_intersect(light, lightray, hit, dist, atten, st, ppc);
		     st->bv_prim_isect_light++;
#if 0
		     if(atten.luminance()<1.e-6){
			 shadow_cache=prims[L-primStart];
			 break;
		     }
#endif
		     prims[R-primStart]->light_intersect(light, lightray, hit, dist, atten, st, ppc);
		     st->bv_prim_isect_light++;
#if 0
		     if(atten.luminance()<1.e-6){
			 shadow_cache=prims[R-primStart];
			 break;
		     }
#endif
		     if(--sp<0)
			 break;
		     idx=bstack[sp];
		 } else if(R>=primStart){
		     prims[R-primStart]->light_intersect(light, lightray, hit, dist, atten, st, ppc);
		     st->bv_prim_isect_light++;
#if 0
		     if(atten.luminance()<1.e-6){
			 shadow_cache=prims[R-primStart];
			 break;
		     }
#endif
		     idx=L;
		 } else {
		     bstack[sp++]=R;
		     st->bv_prim_isect_light++;
		     if(sp>=BSTACK_SIZE){
			 cerr << "BSTACK OVERFLOW!\n";
			 Thread::exitAll(-1);
		     }
		     idx=L;
		 }
	     } else {
		 if(L>=primStart){
		     prims[L-primStart]->light_intersect(light, lightray, hit, dist, atten, st, ppc);
		     st->bv_prim_isect_light++;
#if 0
		     if(atten.luminance()<1.e-6){
			 shadow_cache=prims[L-primStart];
			 break;
		     }
#endif
		     if(--sp<0)
			 break;
		     idx=bstack[sp];
		 } else {
		     idx=L;
		 }
	     }
	 } else {
	     if(rhit){
		 if(R>=primStart){
		     prims[R-primStart]->light_intersect(light, lightray, hit, dist, atten, st, ppc);
		     st->bv_prim_isect_light++;
#if 0
		     if(atten<1.e-6){
			 shadow_cache=prims[R-primStart];
			 break;
		     }
#endif
		     if(--sp<0)
			 break;
		     idx=bstack[sp];
		 } else {
		     idx=R;
		 }
	     } else {
		 if(--sp<0)
		     break;
		 idx=bstack[sp];
	     }
	 }
     }
}


void BV1::animate(double t, bool& changed)
{
    obj->animate(t, changed);
}

void BV1::collect_prims(Array1<Object*>& prims)
{
    prims.add(this);
}

void BV1::compute_bounds(BBox& bbox, double offset)
{
    obj->compute_bounds(bbox, offset);
}

Vector BV1::normal(const Point&, const HitInfo&)
{
    cerr << "Error: BV1 normal should not be called!\n";
    return Vector(0,0,0);
}

void BV1::multi_light_intersect(Light* light, const Point& orig,
				const Array1<Vector>& dirs,
				const Array1<Color>& in_attens,
				double dist,
				DepthStats* st, PerProcessorContext* ppc)
{
    int idx=0;
    int sp=0;
    double* slabs=normal_tree->slabs;
    int primStart=normal_tree->primStart;
    Object** prims=&normal_tree->prims[0];
#define BSTACK_SIZE 100
    int bstack[BSTACK_SIZE];
    Vector idirs[100];
    for(int i=0;i<dirs.size();i++){
	const Vector& dir=dirs[i];
	Vector idir(1./dir.x(), 1./dir.y(), 1./dir.z());
	idirs[i]=idir;
    }
    Color* attens=&in_attens[0];
    int ndirs=dirs.size();
    for(;;){
	int L=2*idx+1;
	int R=L+1;
	bool lhit=false;
	bool rhit=false;
	for(int i=0;i<ndirs;i++){
	    if(attens[i].luminance() != 0){
		bool lh, rh;
		isect_bbox(orig, idirs[i], &slabs[L*6], lh, rh, 1.e30);
		lhit|=lh;
		rhit|=rh;
		if(lhit && rhit)
		    break;
	    }
	}
	st->bv_total_isect+=2;
	if(lhit){
	    if(rhit){
		if(L>=primStart){
		    st->bv_prim_isect+=2;
		    prims[L-primStart]->multi_light_intersect(light, orig,
							      dirs, in_attens,
							      dist, st, ppc);
		    prims[R-primStart]->multi_light_intersect(light, orig,
							      dirs, in_attens,
							      dist, st, ppc);
		    if(--sp<0)
			break;
		    idx=bstack[sp];
		} else if(R>=primStart){
		    st->bv_prim_isect++;
		    prims[R-primStart]->multi_light_intersect(light, orig,
							      dirs, in_attens,
							      dist, st, ppc);
		    idx=L;
		} else {
		    bstack[sp++]=R;
		    if(sp>=BSTACK_SIZE){
			cerr << "BSTACK OVERFLOW!\n";
			Thread::exitAll(-1);
		    }
		    idx=L;
		}
	    } else {
		if(L>=primStart){
		    st->bv_prim_isect++;
		    prims[L-primStart]->multi_light_intersect(light, orig,
							      dirs, in_attens,
							      dist, st, ppc);
		    if(--sp<0)
			break;
		    idx=bstack[sp];
		} else {
		    idx=L;
		}
	    }
	} else {
	    if(rhit){
		if(R>=primStart){
		    st->bv_prim_isect++;
		    prims[R-primStart]->multi_light_intersect(light, orig,
							      dirs, in_attens,
							      dist, st, ppc);
		    if(--sp<0)
			break;
		    idx=bstack[sp];
		} else {
		    idx=R;
		}
	    } else {
		if(--sp<0)
		    break;
		idx=bstack[sp];
	    }
	}
    }    
}



#include <Packages/rtrt/Core/Group.h>
#include <iostream>

using namespace rtrt;

Group::Group()
    : Object(0)
{
}

Group::~Group()
{
}

void Group::light_intersect(Light* light, const Ray& ray, HitInfo& hit,
			    double dist, Color& atten, DepthStats* st,
			    PerProcessorContext* ppc)
{
    for(int i=0;i<objs.size();i++){
	objs[i]->light_intersect(light, ray, hit, dist, atten, st, ppc);
    }
}

void Group::intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
		      PerProcessorContext* ppc)
{
    for(int i=0;i<objs.size();i++){
	objs[i]->intersect(ray, hit, st, ppc);
    }
}

void Group::multi_light_intersect(Light* light, const Point& orig,
				  const Array1<Vector>& dirs,
				  const Array1<Color>& attens,
				  double dist,
				  DepthStats* st, PerProcessorContext* ppc)
{
    for(int i=0;i<objs.size();i++){
	objs[i]->multi_light_intersect(light, orig, dirs, attens,
				       dist, st, ppc);
    }
}

Vector Group::normal(const Point&, const HitInfo&)
{
    cerr << "Error: Group normal should not be called!\n";
    return Vector(0,0,0);
}


void Group::add(Object* obj)
{
    objs.add(obj);
}

void Group::animate(double t, bool& changed)
{
    for(int i=0;i<objs.size();i++){
	objs[i]->animate(t, changed);
    }
}

void Group::collect_prims(Array1<Object*>& prims)
{
    for(int i=0;i<objs.size();i++){
	objs[i]->collect_prims(prims);
    }
}

void Group::preprocess(double maxradius, int& pp_offset, int& scratchsize)
{
    for(int i=0;i<objs.size();i++){
	objs[i]->preprocess(maxradius, pp_offset, scratchsize);
    }
}

void Group::compute_bounds(BBox& bbox, double offset)
{
    for(int i=0;i<objs.size();i++){
	objs[i]->compute_bounds(bbox, offset);
    }
}

void Group::prime(int n)
{
    int nobjs=objs.size();
    objs.resize(n);
    objs.resize(nobjs);
}


#include <Packages/rtrt/Core/MIPGroup.h>
#include <Packages/rtrt/Core/HitInfo.h>

using namespace rtrt;

MIPGroup::MIPGroup()
{
}

MIPGroup::~MIPGroup()
{
}

void MIPGroup::intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext* ppc)
{
    double maxval=0;
    Object* maxobj=0;
    double maxt;
    for(int i=0;i<objs.size();i++){
	HitInfo newhit;
	objs[i]->intersect(ray, newhit, st, ppc);
	if(newhit.was_hit){
	    double* newval=(double*)newhit.scratchpad;
	    if(*newval > maxval){
		maxval=*newval;
		maxobj=newhit.hit_obj;
		maxt=newhit.min_t;
	    }
	}
    }
    if(maxobj){
	hit.hit(maxobj, maxt);
	double* val=(double*)hit.scratchpad;
	*val=maxval;
    }
}

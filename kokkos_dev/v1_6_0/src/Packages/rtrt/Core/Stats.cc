
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Color.h>
#include <strings.h>
#include <iostream>

using namespace rtrt;

extern "C" {
#if HAVE_PERFEX
         int start_counters( int e0, int e1 );
         int read_counters( int e0, long long *c0, int e1, long long *c1);
         int print_counters( int e0, long long c0, int e1, long long c1);
         int print_costs( int e0, long long c0, int e1, long long c1);
         int load_costs(char *CostFileName);
#endif
}


Stats::Stats(int maxstats)
    : maxstats(maxstats)
{
    buf=new double[4*maxstats];
    n=0;
}

Stats::~Stats()
{
    delete[] buf;
}

int Stats::nstats()
{
    return n;
}

double Stats::time(int idx)
{
    return buf[idx*4];
}

double* Stats::color(int idx)
{
    return &buf[idx*4+1];
}

void Stats::reset()
{
    n=0;
    bzero(ds, sizeof(ds));
}

void Stats::add(double time, const Color& c)
{
    int idx=4*n;
    n++;
    if(n>=maxstats){
      //	cerr << "STATS OVERFLOW! n=" << n << "\n";
    } else {
	buf[idx++]=time;
	buf[idx++]=c.red();
	buf[idx++]=c.green();
	buf[idx++]=c.blue();
    }
}


void DepthStats::addto(DepthStats& ds)
{
    nrays+=ds.nrays;
    nbg+=ds.nbg;
    nrefl+=ds.nrefl;
    ntrans+=ds.ntrans;
    nshadow+=ds.nshadow;
    inshadow+=ds.inshadow;
    shadow_cache_try+=ds.shadow_cache_try;
    shadow_cache_miss+=ds.shadow_cache_miss;
    bv_prim_isect+=ds.bv_prim_isect;
    bv_total_isect+=ds.bv_total_isect;
    bv_prim_isect_light+=ds.bv_prim_isect_light;
    bv_total_isect_light+=ds.bv_total_isect_light;
    
    sphere_isect+=ds.sphere_isect;
    sphere_hit+=ds.sphere_hit;
    sphere_light_isect+=ds.sphere_light_isect;
    sphere_light_hit+=ds.sphere_light_hit;
    sphere_light_penumbra+=ds.sphere_light_penumbra;
    
    tri_isect+=ds.tri_isect;
    tri_hit+=ds.tri_hit;
    tri_light_isect+=ds.tri_light_isect;
    tri_light_hit+=ds.tri_light_hit;
    tri_light_penumbra+=ds.tri_light_penumbra;

    rect_isect+=ds.rect_isect;
    rect_hit+=ds.rect_hit;
    rect_light_isect+=ds.rect_light_isect;
    rect_light_hit+=ds.rect_light_hit;
    rect_light_penumbra+=ds.rect_light_penumbra;

    parallelogram_isect+=ds.parallelogram_isect;
    parallelogram_hit+=ds.parallelogram_hit;
    parallelogram_light_isect+=ds.parallelogram_light_isect;
    parallelogram_light_hit+=ds.parallelogram_light_hit;
    parallelogram_light_penumbra+=ds.parallelogram_light_penumbra;

}

Counters::Counters(int ncounters, int ic0, int ic1)
    : ncounters(ncounters), c0(ic0), c1(ic1)
{
    if(ncounters){
	if(ncounters==1){
	    if(c0<16)
		c1=16;
	    else
		c1=0;
	}
#if HAVE_PERFEX
	start_counters(c0, c1);
#endif
    }
}

void Counters::end_frame()
{
#if 0
    if(ncounters){
#if HAVE_PERFEX
	read_counters(c0, &val0, c1, &val1);
	start_counters(c0, c1);
#endif
    }
#endif
}


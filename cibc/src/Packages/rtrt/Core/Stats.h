
#ifndef STATS_H
#define STATS_H 1

#include <Packages/rtrt/Core/params.h>

namespace rtrt {
  
class Color;

struct DepthStats {
    int nrays;
    int nbg;
    int nrefl;
    int ntrans;
    int nshadow;
    int inshadow;
    int shadow_cache_try;
    int shadow_cache_miss;
    int bv_prim_isect;
    int bv_total_isect;
    int bv_prim_isect_light;
    int bv_total_isect_light;

    int sphere_isect;
    int sphere_hit;
    int sphere_light_isect;
    int sphere_light_hit;
    int sphere_light_penumbra;

    int   box_isect;
    int box_hit;
    int box_light_isect;
    int box_light_hit;
    int box_light_penumbra;

    int tri_isect;
    int tri_hit;
    int tri_light_isect;
    int tri_light_hit;
    int tri_light_penumbra;

    int rect_isect;
    int rect_hit;
    int rect_light_isect;
    int rect_light_hit;
    int rect_light_penumbra;

    int parallelogram_isect;
    int parallelogram_hit;
    int parallelogram_light_isect;
    int parallelogram_light_hit;
    int parallelogram_light_penumbra;

    void addto(DepthStats& st);
};

class Counters {
protected:
    char pad[128];
    int ncounters;
    int c0, c1;
    long long val0, val1;
    friend class Stats;
public:
    Counters(int ncounters, int c0, int c1);
    void end_frame();
    inline long long count0() {
	return val0;
    }
    inline long long count1() {
	return val1;
    }
    char pad2[128];

};

class Stats {
    char pad[128];
    int n;
    int maxstats;
    double* buf;
public:
    Stats(int maxstats);
    ~Stats();
    int nstats();
    double time(int idx);
    double* color(int idx);

    void reset();
    void add(double time, const Color&);
    int npixels;
    DepthStats ds[MAXDEPTH];
    char pad2[128];
};

} // end namespace rtrt

#endif

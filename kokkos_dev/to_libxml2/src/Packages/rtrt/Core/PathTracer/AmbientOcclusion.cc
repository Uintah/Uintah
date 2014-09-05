
#include <Packages/rtrt/Core/PathTracer/AmbientOcclusion.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Background.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace rtrt;
using namespace std;

AmbientOccContext::AmbientOccContext(Object* geom,
                                     const Point& min, const Point& max,
                                     int nx, int ny, int nz,
                                     int num_samples,
                                     Background* bg) :
  geometry(geom),
  min(min), max(max),
  background(bg),
  ambient_values(nx, ny, nz)
{
  // Initialize all the values to make sure we have a resonable value.
  ambient_values.initialize(0);
  
  // Create a list of directions to be used by the threads
  directions.resize(num_samples);
  double prev_phi = 0;
  for(int k = 1; k <= num_samples; k++) {
    double hk = -1 + 2*(k-1)/(num_samples-1);
    double theta = acos(hk);
    double hk2 = hk*hk;
    double phi;
    if (hk2 < 1)
      phi = prev_phi + 3.6/sqrt((double)num_samples)/sqrt(1-hk2);
    else
      phi = 0;
    prev_phi = phi;
    // Now get the three coordinates
    double x=cos(phi)*sin(theta);
    double z=sin(phi)*sin(theta);
    double y=cos(theta);
    Vector dir(x,y,z);
    directions[k-1] = dir.normal();
  }

  // Determine the scratch size needed
  float bvscale=0.3;
  pp_size=0;
  pp_scratchsize=0;
  geometry->preprocess(bvscale, pp_size, pp_scratchsize);
}

AmbientOccWorker::AmbientOccWorker(AmbientOccContext* aoc) :
  aoc(aoc)
{
  // Allocate the DepthStats and PerProcessor context
  ppc = new PerProcessorContext(aoc->pp_size, aoc->pp_scratchsize);
  depth_stats = new DepthStats();
}

AmbientOccWorker::~AmbientOccWorker()
{
  if (ppc) delete ppc;
  if (depth_stats) delete depth_stats;
}

int AmbientOccWorker::setWork(int minx_in, int miny_in, int minz_in,
                              int maxx_in, int maxy_in, int maxz_in) {
  char *me = "AmbientOccWorker::setWork";
  
  // check to see if anyone is negative
  if (minx_in < 0) { cerr << me << ": ERROR: minx_in < 0\n"; return 1; }
  if (miny_in < 0) { cerr << me << ": ERROR: miny_in < 0\n"; return 1; }
  if (minz_in < 0) { cerr << me << ": ERROR: minz_in < 0\n"; return 1; }

  if (maxx_in < 0) { cerr << me << ": ERROR: maxx_in < 0\n"; return 1; }
  if (maxy_in < 0) { cerr << me << ": ERROR: maxy_in < 0\n"; return 1; }
  if (maxz_in < 0) { cerr << me << ": ERROR: maxz_in < 0\n"; return 1; }

  // Check the max extents
  if (maxx_in > aoc->nx) { cerr << me <<": ERROR: maxx_in("<<maxx_in<<") > nx("<<aoc->nx<<")\n"; return 1; }
  if (maxy_in > aoc->ny) { cerr << me <<": ERROR: maxy_in("<<maxy_in<<") > ny("<<aoc->ny<<")\n"; return 1; }
  if (maxz_in > aoc->nz) { cerr << me <<": ERROR: maxz_in("<<maxz_in<<") > nz("<<aoc->nz<<")\n"; return 1; }
  
  // check the sizes
  int nx_in = maxx_in - minx_in;
  int ny_in = maxy_in - miny_in;
  int nz_in = maxz_in - minz_in;
  if (nx_in <=0) { cerr << me << ": ERROR: nx_in ("<<nx_in<<") is bad\n"; return 1; }
  if (ny_in <=0) { cerr << me << ": ERROR: ny_in ("<<ny_in<<") is bad\n"; return 1; }
  if (nz_in <=0) { cerr << me << ": ERROR: nz_in ("<<nz_in<<") is bad\n"; return 1; }

  return 0;
}

void AmbientOccWorker::run() {
  for(int z_index = minz; z_index < maxz; z_index++)
    for(int y_index = minx; y_index < maxy; y_index++)
      for(int x_index = minx; x_index < maxx; x_index++) {
        Point origin = (Vector(x_index, y_index, z_index) * (aoc->max-aoc->min)
          + aoc->min.asVector()).asPoint();
        float result = 0;
        for(int dir_index = 0; dir_index < aoc->directions.size(); dir_index++)
          {
	    Ray ray(origin, aoc->directions[dir_index]);

	    // Trace ray into geometry
	    HitInfo hit;
	    aoc->geometry->intersect(ray, hit, depth_stats, ppc);

	    if (!hit.was_hit) {
	      // Accumulate bg color?
	      Color bgcolor;
	      aoc->background->color_in_direction(ray.direction(), bgcolor);
	      result += bgcolor.luminance();
	    }
	  } // end foreach directions[i]
        // Save the result back
        aoc->ambient_values(x_index, y_index, z_index) = result;
      }
}


#include <Core/Packages/rtrt/Core/PathTrace/AmbientOcclusion.h>
#include <Core/Packages/rtrt/Core/Object.h>
//#include <Core/Packages/rtrt/Core/HitInfo.h>
//#include <Core/Packages/rtrt/Core/Ray.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace rtrt;
using namespace std;

AmbientOccContext::AmbientOccContext()
{
  // Determine the scratch size needed
  float bvscale=0.3;
  pp_size=0;
  pp_scratchsize=0;
  geometry->preprocess(bvscale, pp_size, pp_scratchsize);
}

AmbientOccWorker::AmbientOccWorker()
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

  ambient_values.resize(nx_in, ny_in, nz_in);

  return 0;
}

void AmbientOccWorker::run() {
  for(int z_index = minz; z_index < maxz; z_index++)
    for(int y_index = minx; y_index < maxy; y_index++)
      for(int x_index = minx; x_index < maxx; x_index++) {
        Point origin = Vector(x_index, y_index, z_index) * (aoc->max-aoc->min)
          + aoc->min.asVector();
        float result = 0;
        for(int dir_index = 0; dir_index < aoc->directions.size(); dir_index++)
          {
	    Ray ray(origin, directions[dir_index]);

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
        aoc->(x_index, y_index, z_index) = result;
      }
}

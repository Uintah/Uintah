

#include <Packages/rtrt/Core/PathTracer/PathTraceEngine.h>

#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Point2D.h>
#include <Packages/rtrt/Core/HitInfo.h>

#include <Core/Thread/Runnable.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <math.h>

using namespace rtrt;
using namespace SCIRun;
using namespace std;

PathTraceContext::PathTraceContext(Color &color, Object* geometry,
				   int num_samples):
      color(color), geometry(geometry), num_samples(num_samples)
{
  // Fix num_samples to be a complete square
  num_samples_root = (int)(ceil(sqrt((double)num_samples)));
  int new_num_samples = num_samples_root * num_samples_root;
  if (new_num_samples != num_samples) {
    cerr << "Changing the number of samples from "<<num_samples<<" to "<<new_num_samples<<"\n";
    num_samples = new_num_samples;
  }

  // Determine the scratch size needed
  float bvscale = 0.3;
  pp_size = 0;
  pp_scratchsize = 0;
  geometry->preprocess(bvscale, pp_size, pp_scratchsize);
}

PathTraceWorker::PathTraceWorker():
  rng(10)
{
  // Generate our group of random samples
  double inc = 1.0/ptc->num_samples_root;
  for(int sgindex = 0; sgindex < NUM_SAMPLE_GROUPS; sgindex++) {
    // This is our sample index
    int index = 0;
    // These are offsets into the various pixel regions
    double x = 0;
    for (int i = 0; i < ptc->num_samples_root; i++) {
      ASSERT(index < ptc->num_samples);
      double y = 0;
      for (int j = 0; j < ptc->num_samples_root; j++) {
	sample_points[sgindex][index] = Point2D(x + rng() * inc,
						y + rng() * inc);
	index++;
	y += inc;
      }
      x += inc;
    }
  }

  // Allocate the DepthStats and PerProcessor context
  ppc = new PerProcessorContext(ptc->pp_size, ptc->pp_scratchsize);
  depth_stats = new DepthStats();
}

PathTraceWorker::~PathTraceWorker() {
  if (ppc) delete ppc;
  if (depth_stats) delete depth_stats;
}

void PathTraceWorker::run() {
  
  // Iterate over our spheres
  for(int sindex = 0; sindex < local_spheres->objs.size(); sindex++) {
    TextureSphere *sphere  =
      dynamic_cast<TextureSphere*>(local_spheres->objs[sindex]);
    if (!sphere) {
      // Something has gone horribly wrong
      cerr << "Object in local list (index="<<sindex<<") is not a TextureSphere\n";
      continue;
    }
    int width = sphere->texture.dim1();
    int height = sphere->texture.dim2();
    // Iterate over each texel
    for(int y = 0; y < height; y++)
      for(int x = 0; x < width; x++)
	{
	  int sgindex = (int)(rng()*(NUM_SAMPLE_GROUPS-1));
	
	  for(int sample = 0; sample < ptc->num_samples; sample++) {
	    Point2D sample_point = sample_points[sgindex][sample];
	    // Project 2D point onto sphere
	    Point ray_origin;
	    // Pick a random direction on the hemisphere
	    Vector normal;
	    Vector ray_dir;
	    // Trace ray into our group
	    Ray ray(ray_origin, ray_dir);
	    HitInfo local_hit;
	    local_spheres->intersect(ray, local_hit, depth_stats, ppc);
	    // Trace ray into our geometry
	    HitInfo global_hit;
	    ptc->geometry->intersect(ray, global_hit, depth_stats, ppc);
	  }
	}
      
  } // end sphere
}

TextureSphere::TextureSphere(Point cen, double radius, int tex_res):
  Sphere(0, cen, radius), texture(tex_res, tex_res)
{
  texture.initialize(0);
}


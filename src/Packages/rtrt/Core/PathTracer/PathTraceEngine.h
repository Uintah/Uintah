#ifndef RTRT_PATH_TRACE_ENGINE_H
#define RTRT_PATH_TRACE_ENGINE_H 1

#include <Core/Thread/Runnable.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/MusilRNG.h>
#include <Packages/rtrt/Core/Point2D.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Array2.h>

namespace rtrt {

  using SCIRun::Runnable;

  class PerProcessorContext;
  class DepthStats;

  class PathTraceContext {
  public:
    PathTraceContext(Color &color, Object* geometry, int num_samples);

    // Light source
    
    // Object color
    Color color;
    // Acceleration geometry
    Object* geometry;
    // Number of samples to perform for each texel
    int num_samples;
    int num_samples_root;

    // Stuff for PerProcessorContext
    int pp_size;
    int pp_scratchsize;
  };

#define NUM_SAMPLE_GROUPS 100
  
  class PathTraceWorker: public Runnable {
    // This has all the global information
    PathTraceContext* ptc;
    // Textures
    Group* local_spheres;
    // Random number generator
    MusilRNG rng;
    // Our group of random samples
    Array1<Point2D> sample_points[NUM_SAMPLE_GROUPS];

    // Used for rendering
    PerProcessorContext* ppc;
    DepthStats* depth_stats;
  public:
    PathTraceWorker();
    ~PathTraceWorker();
    // This does all the work
    void run();
  };
  
  class TextureSphere: public Sphere {
  protected:
    // Pointer to its texture
    Array2<Color> texture;
    friend class PathTraceWorker;
  public:
    TextureSphere(Point cen, double radius, int tex_res);
  };
  
} // end namespace rtrt

#endif

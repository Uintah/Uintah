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

  class PathTraceLight {
  public:
    Point center;
    double radius;
    double area;
    Color color;

    PathTraceLight(const Point &cen, double radius, const Color &c);
    Point random_point(double r1, double r2, double r3);
  };

  class PathTraceContext {
  public:
    PathTraceContext(const Color &color, const PathTraceLight &light,
		     Object* geometry, int num_samples, int max_depth);

    // Light source
    PathTraceLight light;
    // Object color
    Color color;
    // Acceleration geometry
    Object* geometry;
    // Number of samples to perform for each texel
    int num_samples;
    int num_samples_root;
    // Maximum ray depth
    int max_depth;

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
    // Base filename for textures
    char *basename;
    // Random number generator
    MusilRNG rng;
    // Our group of random samples
    Array1<Point2D> sample_points[NUM_SAMPLE_GROUPS];

    // Used for rendering
    PerProcessorContext* ppc;
    DepthStats* depth_stats;
  public:
    PathTraceWorker(Group *group, PathTraceContext *ptc, char *texname);
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
    TextureSphere(const Point &cen, double radius, int tex_res=16);
    void writeTexture(char* basename, int index);
  };

} // end namespace rtrt

#endif

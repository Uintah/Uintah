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

namespace SCIRun {
  class Semaphore;
}

namespace rtrt {

  using SCIRun::Runnable;
  using SCIRun::Semaphore;

  class PerProcessorContext;
  class DepthStats;
  class Background;

  class PathTraceLight {
  public:
    Point center;
    double radius;
    double area;
    Color color;

    PathTraceLight(const Point &cen, double radius, const Color &c);
    Vector random_vector(double r1, double r2, double r3);
    Point random_point(double r1, double r2, double r3);
    Point point_from_normal(const Vector& dir);
    Vector normal(const Point &point);
  };

  class PathTraceContext {
  public:
    PathTraceContext(const Color &color, const PathTraceLight &light,
		     Object* geometry, Background *background,
		     int num_samples, int max_depth,
		     Semaphore *semi = 0);

    // Light source
    PathTraceLight light;
    // Object color
    Color color;
    // Acceleration geometry
    Object* geometry;
    // Background
    Background *background;
    // Number of samples to perform for each texel
    int num_samples;
    int num_samples_root;
    // Maximum ray depth
    int max_depth;

    // Semephore for threading
    Semaphore *semi;

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

    size_t offset;
  public:
    PathTraceWorker(Group *group, PathTraceContext *ptc, char *texname,
		    size_t offset=0);
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
    ~TextureSphere() {}
    
    void writeTexture(char* basename, size_t index);
    void writeData(FILE *outfile);
  };

} // end namespace rtrt

#endif

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
    float luminance;
    
    PathTraceLight(const Point &cen, double radius, float lum);
    Vector random_vector(double r1, double r2, double r3);
    Point random_point(double r1, double r2, double r3);
    Point point_from_normal(const Vector& dir);
    Vector normal(const Point &point);
  };
  
  class PathTraceContext {
  public:
    PathTraceContext(float luminance, const PathTraceLight &light,
		     Object* geometry, Background *background,
		     int num_samples, int num_sample_divs,
		     int max_depth, bool dilate,
		     int support, int use_weighted_ave,
		     float threshold, Semaphore* sem);

    // Light source
    PathTraceLight light;
    // Surface luminance
    float luminance;
    // Lighting parameters
    bool compute_shadows; // defaults to true
    bool compute_directlighting; // defaults to true
    // Acceleration geometry
    Object* geometry;
    // Background
    Background *background;
    // This is the size of num_samples and num_samples_root
    int num_sample_divs;
    // Number of samples to perform for each texel
    Array1<int> num_samples; 
    Array1<int> num_samples_root;
    // Maximum ray depth
    int max_depth;
    // Stuff for texture dilation
    bool dilate;
    int support;
    int use_weighted_ave;
    float threshold;

    // Semephore for threading
    Semaphore *sem;

    // Stuff for PerProcessorContext
    int pp_size;
    int pp_scratchsize;

    void shadowsOff() { compute_shadows = false; }
    void directLightingOff() {
      compute_shadows = false;
      compute_directlighting = false;
    }
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
    // Should be of dimesntions
    //   [NUM_SAMPLE_GROUPS, num_sample_divs, num_samples]
    Array2<Array1<Point2D> > sample_points;
    
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
    Array2<float> texture;
    Array2<float> inside;
    friend class PathTraceWorker;
  public:
    TextureSphere(const Point &cen, double radius, int tex_res=16);
    ~TextureSphere() {}
    
    void dilateTexture(size_t index, PathTraceContext* ptc);
    void writeTexture(char* basename, size_t index, PathTraceContext* ptc);
    void writeData(FILE *outfile, size_t index, PathTraceContext* ptc);
  };
  
} // end namespace rtrt

#endif

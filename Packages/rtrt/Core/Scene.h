
#ifndef SCENE_H
#define SCENE_H 1

#include <Core/Thread/WorkQueue.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Plane.h>
#include <Packages/rtrt/Core/Background.h>
#include <Packages/rtrt/Core/Material.h>
#include <stdio.h>

namespace rtrt {

using SCIRun::WorkQueue;

class Object;
class Camera;
class Image;
class Light;
class Ray;
class HitInfo;
class DpyBase;
struct DepthStats;
struct PerProcessorContext;

class Scene {
  Object* obj;
  Camera* camera0;
  Camera* camera1;
  Image* image0;
  Image* image1;
  Background* background;
  Color cup;           // color above groundplane
  Color cdown;         // color in direction of groundplane
  Plane groundplane;   // the groundplane for ambient hack
                       // distance guage is based on normal length
  
  Array1<Light*> lights;
  RTRT *rtrt_engine;
  Array1<DpyBase*> displays;
  
  double ambientscale;
  bool hotspots;
  Array1<Material*> materials;
  friend class Dpy;
public:
  Scene(Object*, const Camera&, Image*, Image*, const Color& bgcolor,
	const Color& cdown, const Color& cup, const Plane& groundplane,
	double ambientscale);
  Scene(Object*, const Camera&, const Color& bgcolor,
	const Color& cdown, const Color& cup, const Plane& groundplane,
	double ambientscale);
  ~Scene();
  
  inline Image* get_image(int which) const {
    return which==0?image0:image1;
  }
  inline void set_image(int which, Image* i) {
    if(which==0)
      image0=i;
    else
      image1=i;
  }
  
  inline Camera* get_camera(int which) const {
    return which==0?camera0:camera1;
  }
  
  inline Material* get_material(int which) {
    return materials[which];
  }

  inline int nmaterials() {
    return materials.size();
  }

  inline void set_materials(const Array1<Material*> &copy) {
    materials=copy;
  }

  void copy_camera(int which);
  
  inline Object* get_object() const {
    return obj;
  }
  inline Object* get_shadow_object() const {
    return shadowobj;
  }
  inline void set_object(Object* new_obj) {
    obj=new_obj;
  }
  
  inline const Plane& get_groundplane() const {
    return groundplane;
  }
  inline void set_groundplane(const Plane& p) {
    groundplane=p;
  }
  
  inline void get_bgcolor( const Vector& v, Color& result ) const {
    background->color_in_direction( v, result );
  }
  
  inline const Color& get_average_bg( ) const {
    return background->average( );
  }
  
  inline void set_background_ptr( Background* ptr ) {
    background = ptr;
  }
  
  inline void set_bgcolor(const Color& c) {
    background = new ConstantBackground(c);
  }
  
  inline const Color& get_cdown() const {
    return cdown;
  }
  inline void set_cdown(const Color& c) {
    cdown=c;
  }
  
  inline const Color& get_cup() const {
    return cup;
  }
  inline void set_cup(const Color& c) {
    cup=c;
  }
  
  inline double get_ambientscale() const {
    return ambientscale;
  }
  
  inline int nlights() {
    return lights.size();
  }
  
  inline Light* light(int i) {
    return lights[i];
  }

  inline void set_rtrt_engine(RTRT* _rtrt) {
    rtrt_engine = _rtrt;
  }

  inline RTRT* get_rtrt_engine() {
    return rtrt_engine;
  }
  
  void add_light(Light*);
  int nprims();
  
  WorkQueue work;
  void refill_work(int which, int nworkers);
  void waitForEmpty(int which);
  
  
  void light_intersect(Object* obj, Light* light, const Ray& ray,
		       HitInfo& hitinfo, double dist, double& atten,
		       PerProcessorContext* ppc);
  void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  int shadow_mode;
  int maxdepth;
  float base_threshold;
  float full_threshold;
  double xoffset;
  double yoffset;
  
  int xtilesize;
  int ytilesize;
  bool no_aa;
  bool ambient_hack;
  Object* shadowobj;
  bool stereo;
  bool animate;
  
  bool logframes;
  int frameno;
  FILE* frametime_fp;
  double lasttime;
  bool followpath;  
  FILE* path_fp;

  bool doHotSpots() {
    return hotspots;
  }

  void attach_display(DpyBase *dpy);
};

} // end namespace rtrt

#endif

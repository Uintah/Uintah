
#ifndef SCENE_H
#define SCENE_H 1

#include <Core/Thread/WorkQueue.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Containers/LockingHandle.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Plane.h>
#include <Packages/rtrt/Core/Background.h>
#include <Packages/rtrt/Core/Shadows/ShadowBase.h>
#include <Core/Thread/Mutex.h>

#include <stdio.h>

#include <vector>
#include <string>

namespace rtrt {

using SCIRun::WorkQueue;
using std::vector;
using std::string;

class Object;
class Camera;
class Image;
class Light;
class Ray;
class HitInfo;
class DpyBase;
class Gui;
class Material;
class ShadowBase;
class Group;
class Scene;
class Trigger;
#if !defined(linux)
class SoundThread;
class Sound;
#endif

struct DepthStats;
struct PerProcessorContext;

enum AmbientType { Global_Ambient = 0, Constant_Ambient, Arc_Ambient, 
		   Sphere_Ambient };

class Scene : public SCIRun::Persistent {
  
public:
  Scene(Object*, const Camera&, const Color& bgcolor,
	const Color& cdown, const Color& cup, const Plane& groundplane,
	double ambientscale, AmbientType ambient_mode=Constant_Ambient);
  Scene(); // for Pio.
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

  void set_object(Object* new_obj); 
  
  inline const Plane& get_groundplane() const {
    return groundplane;
  }
  inline void set_groundplane(const Plane& p) {
    groundplane=p;
  }
  
  inline void get_bgcolor( const Vector& v, Color& result ) const {
    background->color_in_direction( v, result );
  }

  // This function is only called to replace the old background that
  // was created in init.
  inline void set_background_ptr( Background* ptr ) {
    orig_background = background = ptr;
  }
  
  inline void set_ambient_environment_map ( EnvironmentMapBackground *ptr ) {
    ambient_environment_map = ptr;
  }

  inline void get_ambient_environment_map_color (const Vector& v,
						  Color &c) {
    if (ambient_environment_map) 
      ambient_environment_map->color_in_direction(v, c);
    else c=Color(0,1,0);
  }

  inline void set_bgcolor(const Color& c) {
    background = new ConstantBackground(c);
  }
  
  inline void set_original_bg() {
    background = orig_background;
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
  
  inline const Color & getAmbientColor() const {
    return ambientColor_;
  }

  inline void setBaseAmbientColor(const Color &c) {
    origAmbientColor_ = c;
    ambientColor_ = origAmbientColor_ * ambientScale_;
  }

  inline double getAmbientLevel() { return ambientScale_; }

  inline void setAmbientLevel( float scale ) {
    if( scale > 1.0 ) scale = 1.0;
    else if( scale < 0.0 ) scale = 0.0;

    ambientScale_ = scale;

    cup = origCup_ * scale;
    cdown = origCDown_ * scale;

    ambientColor_ = origAmbientColor_ * scale;

    background->updateAmbient( scale );
  }
  inline const vector<string> & getRoutes() const {
    return routeNames_;
  }
  
  // Render a sphere in the scene for each light.
  void renderLights( bool on ); 

  // left is % left on... ranging from 1.0 to 0.0 to turn it off.
  void turnOffAllLights( double left ); 
  void turnOnAllLights(); // Put all lights back in the active light list.

  inline int nlights() {
    return lights.size();
  }
  inline int nPerMatlLights() {
    return per_matl_lights.size();
  }
  
  inline Light* light(int i) {
    return lights[i];
  }
  inline Light* per_matl_light(int i) {
    return per_matl_lights[i];
  }
  inline int nlightBits() {
    return lightbits;
  }

  void add_light(Light*);
  void add_permanent_light(Light*);
  void add_per_matl_light(Light*);
  void add_perm_per_matl_light(Light*);

  inline void set_rtrt_engine(RTRT* _rtrt) {
    rtrt_engine = _rtrt;
  }

  inline RTRT* get_rtrt_engine() {
    return rtrt_engine;
  }
  
  int nprims();
  
  WorkQueue work;
  void refill_work(int which, int nworkers);
  void waitForEmpty(int which);

#if !defined(linux)
  // Used mainly by make_scene to register sounds with the application.
  void             addSound( Sound * sound ) { sounds_.push_back( sound ); }
  vector<Sound*> & getSounds() { return sounds_; }
  int              soundVolume() const { return soundVolume_; }
#endif

  void               addTrigger( Trigger * trigger );
  vector<Trigger*> & getTriggers() { return triggers_; }

  // These are mostly used by multi-scene to update the main scene...
  Array1<Object*>   &  getObjectsOfInterest() { return objectsOfInterest_; }
  Array1<Object*>   &  getAnimateObjects() { return animateObjects_; }
  Array1<Object*>   &  getDynBBoxObjs() { return dynamicBBoxObjects_; }
  Array1<DpyBase*>  &  getDisplays() { return displays; }
  Array1<DpyBase*>  &  getAuxDisplays() { return aux_displays; }
  Array1<Material*> &  getMaterials() { return materials; }
  vector<string>    &  getRouteNames() { return routeNames_; }
  vector<string>    &  getRoomsForRoutes() { return roomsForRoutes_; }
  Array1<ShadowBase*> & getShadows() { return shadows; }
  
  void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  Array1<ShadowBase*> shadows;
  int maxdepth;
  float base_threshold;
  float full_threshold;
  double xoffset;
  double yoffset;
  
  int xtilesize;
  int ytilesize;
  bool no_aa;
  Object* shadowobj;
  bool stereo;
  bool animate;

  int     ambient_mode;
  double  ambientScale_;
  double  orig_ambientScale_;

  int frameno;
  FILE* frametime_fp;
  double lasttime;
  int ref_cnt;
  SCIRun::Mutex lock;

  inline int getHotSpotsMode() const { return hotSpotMode_; }

  // Display image as a "transmission".  Ie: turn off every other scan line.
  // (Net effect of this is to double the frame rate.)
  inline bool doTransmissionMode() const { return transmissionMode_; }

  // Any object that the GUI should allow "direct" interaction with
  // or that needs to be animated should notify the scene of this
  // via this call.  If the "name_" of the object is "", then the
  // Gui will not display the object.  Also, animate will (now) only 
  // be called on objects that have been added through this function
  // with the "animate" flag set to true.  If you have an object that needs
  // to change its bounding box on the fly, then you need to set the
  // remakebbox flag to true.
  void addObjectOfInterest( Object * obj, bool animate = false, bool remakebbox = false );
  void addObjectOfInterest( const string& name, Object * obj, bool animate = false, bool remakebbox = false );

  void attach_display(DpyBase *dpy);
  void attach_auxiliary_display(DpyBase *dpy);
  void hide_auxiliary_displays();
  void show_auxiliary_displays();

  void init(const Camera& cam, const Color& bgcolor);
  void add_shadowmode(const char* name, ShadowBase* s);
  void select_shadow_mode( ShadowType st );

  inline bool lit(const Point& hitpos, Light* light,
		  const Vector& light_dir, double dist, Color& shadow_factor,
		  int depth, Context* cx) {
    return shadows[shadow_mode]->lit(hitpos, light, light_dir, dist,
				     shadow_factor, depth, cx);
  }

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);

  // public for testing.
  int shadow_mode;

  // Tell the scene to automatically load the specified route file.
  //   Associate that route file with the given room.
  void addRouteName( const string & filename, const string & room );

private:

//  friend class LumiDpy;
  friend class Dpy;
  friend class Gui;

#if !defined(linux)
  // This is just used as a pass through so the make_scene can get
  // sounds to the sound thread;
  vector<Sound*>   sounds_;
#endif
  int              soundVolume_;

  // List of triggers that we need to constantly check to see
  // if they have fired.
  vector<Trigger*> triggers_;
  
  // Points to either mainGroup_ or mainGroupWithLights_;
  Object * obj;

  Group  * mainGroup_;
  Group  * mainGroupWithLights_;
  Group  * lightsGroup_;
  Group  * permanentLightsGroup_;

  // Objects that are to be presented to the GUI for user interaction.
  Array1<Object*> objectsOfInterest_;

  // Objects that will have their animate() function called:
  Array1<Object*> animateObjects_;

  // Objects that, when they have their animate() function called, must recompute bbox
  Array1<Object*> dynamicBBoxObjects_;
  
  Camera* camera0;
  Camera* camera1;
  Image* image0;
  Image* image1;
  Background* orig_background;
  Background* background;
  Background* ambient_environment_map;

  Color origCup_;      // initialized cup value
  Color origCDown_;    // initialized cdown value

  Color cup;           // color above groundplane
  Color cdown;         // color in direction of groundplane
  Plane groundplane;   // the groundplane for ambient hack
                       // distance guage is based on normal length
  //  int shadow_mode;

  int lightbits;

  // Lights that are on.
  Array1<Light*> lights;
  Array1<Light*> per_matl_lights;

  // Lights that have been turned off.
  Array1<Light*> nonActiveLights_;
  Array1<Light*> nonActivePerMatlLights_;

  RTRT *rtrt_engine;
  Array1<DpyBase*> displays;
  Array1<DpyBase*> aux_displays;
  
  Color  ambientColor_;
  Color  origAmbientColor_;

  int  hotSpotMode_; // 0 == off, 1 == normal, 2 == half screen
  bool transmissionMode_;

  Array1<Material*> materials;

  // If a scene wished to register route names, then these routes
  // will be automatically loaded by the GUI.
  vector< string > routeNames_;
  vector< string > roomsForRoutes_;

}; // end class Scene

typedef SCIRun::LockingHandle<Scene> SceneHandle;

} // end namespace rtrt

#endif

#include <Packages/rtrt/Core/Scene.h>

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Image.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/DpyBase.h>
#include <Packages/rtrt/Core/Shadows/NoShadows.h>
#include <Packages/rtrt/Core/Shadows/HardShadows.h>
#include <Packages/rtrt/Core/Shadows/SingleSampleSoftShadows.h>
#include <Packages/rtrt/Core/Shadows/MultiSampleSoftShadows.h>
#include <Packages/rtrt/Core/Shadows/ScrewyShadows.h>
#include <Packages/rtrt/Core/Shadows/UncachedHardShadows.h>

#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Core/Math/MinMax.h>

#include <iostream>
#include <stdlib.h>
#include <unistd.h>

using namespace rtrt;
using namespace SCIRun;
using namespace std;

Persistent* scene_maker() {
  return new Scene();
}

// initialize the static member type_id
PersistentTypeID Scene::type_id("Scene", "Persistent", scene_maker);

Scene::Scene() : 
  work("frame tiles"),
  ref_cnt(0),
  lock("rtrt::Scene lock")
{}

Scene::Scene(Object* ob, const Camera& cam, Image* image0, Image* image1,
	     const Color& bgcolor,
             const Color& cdown,
             const Color& cup,
	     const Plane& groundplane, 
	     double ambientscale,
	     AmbientType ambient_mode) : 
  work("frame tiles"),
  ambient_mode(ambient_mode),
  ambientScale_(ambientscale),
  ref_cnt(0),
  lock("rtrt::Scene lock"),
  soundVolume_(50),
  obj(ob), 
  mainGroup_(ob), 
  camera0(camera0), 
  image0(image0), 
  image1(image1),
  origCup_(cup), 
  origCDown_(cdown),
  cup(cup), 
  cdown(cdown), 
  groundplane(groundplane),
  transmissionMode_(false)
{
  init(cam, bgcolor);
}

void Scene::init(const Camera& cam, const Color& bgcolor)
{
  lightsGroup_ = new Group;
  mainGroupWithLights_ = lightsGroup_;

  work.refill(0,0,8);
  shadow_mode = Hard_Shadows;
  camera0=new Camera(cam);
  camera1=new Camera(cam);
  maxdepth=2;
  xtilesize=32;
  ytilesize=2;
  shadowobj=0;
  background = new ConstantBackground( bgcolor );
  animate=true;
  hotSpotMode_ = 0;
  frameno=0;
  frametime_fp=0;
  lasttime=0;
  ambient_environment_map=0;

  add_shadowmode(ShadowBase::shadowTypeNames[0],new NoShadows());
  add_shadowmode(ShadowBase::shadowTypeNames[1],new SingleSampleSoftShadows());
  add_shadowmode(ShadowBase::shadowTypeNames[2],new HardShadows());
  add_shadowmode(ShadowBase::shadowTypeNames[3],new ScrewyShadows());
  add_shadowmode(ShadowBase::shadowTypeNames[4],new MultiSampleSoftShadows());
  add_shadowmode(ShadowBase::shadowTypeNames[5],new UncachedHardShadows());

  select_shadow_mode( Hard_Shadows );

  origAmbientColor_ = Color(1,1,1);
  ambientColor_     = origAmbientColor_;
  setAmbientLevel( ambientScale_ );
  orig_ambientScale_ = ambientScale_;
}

void
Scene::set_object(Object* new_obj) 
{
  obj        = new_obj;
  mainGroup_ = new_obj;
  
  mainGroupWithLights_ = new Group;
  mainGroupWithLights_->add( new_obj );
  mainGroupWithLights_->add( lightsGroup_ );
}

void
Scene::add_shadowmode(const char* name, ShadowBase* s)
{
  s->setName(name);
  shadows.add(s);
}

void
Scene::select_shadow_mode( ShadowType st )
{
  shadow_mode = st;
}

Scene::Scene(Object* ob, const Camera& cam, const Color& bgcolor,
             const Color& cdown,
             const Color& cup,
	     const Plane& groundplane,
	     double ambientscale,
	     AmbientType ambient_mode) :     
  work("frame tiles"),
  ambient_mode(ambient_mode),
  ambientScale_(ambientscale),
  ref_cnt(0),
  lock("rtrt::Scene lock"),
  soundVolume_(50),
  obj(ob), 
  mainGroup_(ob ),
  camera0(camera0), 
  image0(0), 
  image1(0),
  origCup_(cup), 
  origCDown_(cdown),
  cup(cup), 
  cdown(cdown), 
  groundplane(groundplane), 
  transmissionMode_(false)
{
  init(cam, bgcolor);
}

Scene::~Scene()
{
  cerr << "Scene destroyed!\n";
    delete lightsGroup_;
    delete mainGroup_;
    delete mainGroupWithLights_;
    delete camera0;
    delete camera1;
    delete image0;
    delete image1;
}

void Scene::refill_work(int which, int nworkers)
{
    int xres = get_image(which)->get_xres();
    int yres = get_image(which)->get_yres();
    int nx   = (xres+xtilesize-1)/xtilesize;
    int ny   = (yres+ytilesize-1)/ytilesize;
    int nass = nx*ny;

    work.refill(nass, nworkers, 40);
}

void Scene::add_light(Light* light)
{
  if( light->isMoodLight() ) light->turnOff();

  lightsGroup_->add( light->getSphere() );
  lights.add(light);
}

void Scene::add_per_matl_light( Light* light )
{
  if( light->isMoodLight() ) light->turnOff();

  lightsGroup_->add( light->getSphere() );
  per_matl_lights.add(light);
}

void Scene::preprocess(double bvscale, int& pp_offset, int& scratchsize)
{
  int i=0;
  for(;i<lights.size();i++)
    lights[i]->setIndex(i);
  for(;i<lights.size()+per_matl_lights.size(); i++)
    per_matl_lights[i-lights.size()]->setIndex(i);
  lightbits = 0;
  i=lights.size()+per_matl_lights.size();
  while(i){
    lightbits++;
    i>>=1;
  }
  double maxradius=0;

  for(i=0;i<lights.size();i++)
    if(lights[i]->radius > maxradius)
      maxradius=lights[i]->radius;
  for(;i<lights.size()+per_matl_lights.size(); i++)
    if(per_matl_lights[i-lights.size()]->radius > maxradius)
      maxradius=per_matl_lights[i-lights.size()]->radius;
  maxradius*=bvscale;
  double time=SCIRun::Time::currentSeconds();
  obj->preprocess(maxradius, pp_offset, scratchsize);
  if(shadowobj)
    shadowobj->preprocess(maxradius, pp_offset, scratchsize);
  else
    shadowobj=obj;
  for(i=0;i<shadows.size();i++)
    shadows[i]->preprocess(this, pp_offset, scratchsize);
  cerr << "Preprocess took " << SCIRun::Time::currentSeconds()-time << " seconds\n";
}

void Scene::copy_camera(int which)
{
    if(which==0){
	*camera1=*camera0;
    } else {
	*camera0=*camera1;
    }
}

int Scene::nprims()
{
    Array1<Object*> prims;
    obj->collect_prims(prims);
    return prims.size();
}

#if 0
void Scene::waitForEmpty(int which)
{
    work[which].waitForEmpty();
}
#endif

void Scene::attach_display(DpyBase *dpy) {
  displays.add(dpy);
  dpy->set_scene(this);
}

void Scene::attach_auxiliary_display(DpyBase *dpy) {
  aux_displays.add(dpy);
}

void Scene::hide_auxiliary_displays() {
  for(int i=0;i<aux_displays.size();i++)
    aux_displays[i]->Hide();
}
void Scene::show_auxiliary_displays() {
  for(int i=0;i<aux_displays.size();i++)
    aux_displays[i]->Show();
}

void
Scene::turnOffAllLights( double left )
{
  static double beginScale;
  if (left == 1.0) beginScale = ambientScale_;

  int numLights = lights.size();
  int numAllLights = numLights + per_matl_lights.size();

  for( int cnt = 0; cnt < numAllLights; cnt++ ) {
    Light * light;
    if( cnt < numLights )
      light = lights[cnt];
    else
      light = per_matl_lights[cnt-numLights];

    if (left>0.0) {

      if( light->isMoodLight() ) {
	light->updateIntensity( 1.0 - left );
	light->turnOn();
      } else {
	light->modifyCurrentIntensity( left );
      }
    } else {
      if( !light->isMoodLight() ){
	light->turnOff();
      }
    }
  }
  // Scale down from current ambient level to minimum ambient of
  // 1/4th of original ambient.
  setAmbientLevel( (beginScale*left) + (orig_ambientScale_ / 2.0) );
}

void
Scene::turnOnAllLights()
{
  int numLights = lights.size();
  int numAllLights = numLights + per_matl_lights.size();

  for( int cnt = 0; cnt < numAllLights; cnt++ ) {
    Light * light;
    if( cnt < numLights )
      light = lights[cnt];
    else
      light = per_matl_lights[cnt-numLights];

    if( light->isMoodLight() ) {
      light->turnOff();
      lights[cnt]->reset(); 
    } else {
      lights[cnt]->reset(); 
      light->turnOn();
    }
  }

  setAmbientLevel(orig_ambientScale_);
}

void
Scene::renderLights( bool on )
{
  if( on ){ 
    // Draw spheres for all the lights
    obj = mainGroupWithLights_;
  } else {
    // Remove the spheres for all the lights
    obj = mainGroup_;
  }
}


const int SCENE_VERSION = 1;

void 
Scene::io(SCIRun::Piostream &stream) {
  cerr << "in Scene Pio" << endl;
  stream.begin_class("Scene", SCENE_VERSION);
  //Pio(stream, s.work);

  SCIRun::Pio(stream, shadows);
  SCIRun::Pio(stream, maxdepth);
  SCIRun::Pio(stream, base_threshold);
  SCIRun::Pio(stream, full_threshold);
  SCIRun::Pio(stream, xoffset);
  SCIRun::Pio(stream, yoffset);
  SCIRun::Pio(stream, xtilesize);
  SCIRun::Pio(stream, ytilesize);
  SCIRun::Pio(stream, no_aa);
  SCIRun::Pio(stream, shadowobj);
  SCIRun::Pio(stream, stereo);
  SCIRun::Pio(stream, animate);
  SCIRun::Pio(stream, ambient_mode);
  SCIRun::Pio(stream, frameno);
  //  SCIRun::Pio(stream, frametime_fp);
  SCIRun::Pio(stream, lasttime);
  SCIRun::Pio(stream, obj);
  SCIRun::Pio(stream, mainGroup_);
  SCIRun::Pio(stream, mainGroupWithLights_);
  SCIRun::Pio(stream, camera0);
  SCIRun::Pio(stream, camera1);
  SCIRun::Pio(stream, image0);
  SCIRun::Pio(stream, image1);
  SCIRun::Pio(stream, background);
  SCIRun::Pio(stream, ambient_environment_map);
  SCIRun::Pio(stream, origCup_);
  SCIRun::Pio(stream, origCDown_);
  SCIRun::Pio(stream, cup);
  SCIRun::Pio(stream, cdown);
  SCIRun::Pio(stream, groundplane);
  SCIRun::Pio(stream, shadow_mode);
  SCIRun::Pio(stream, lightbits);
  SCIRun::Pio(stream, lights);
  SCIRun::Pio(stream, per_matl_lights);
  SCIRun::Pio(stream, nonActiveLights_);
  SCIRun::Pio(stream, nonActivePerMatlLights_);
  //  SCIRun::Pio(stream, rtrt_engine);
  //  SCIRun::Pio(stream, displays);
  SCIRun::Pio(stream, ambientColor_);
  SCIRun::Pio(stream, origAmbientColor_);
  SCIRun::Pio(stream, hotSpotMode_);
  SCIRun::Pio(stream, materials);
  stream.end_class();
}

// Animate will only be called on objects added through this function.
void
Scene::addObjectOfInterest( Object * obj, bool animate /* = false */ )
{
  if( animate )
    animateObjects_.add( obj );
  if( obj->name_ != "" )
    objectsOfInterest_.add( obj );
}

// For adding single route names
void
Scene::addRouteName( const string & filename, const string & room )
{
  routeNames_.push_back( filename );
  roomsForRoutes_.push_back( room );
}

void
Scene::addTrigger( Trigger * trigger )
{
  triggers_.push_back( trigger );
}

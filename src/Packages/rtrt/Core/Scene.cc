#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Image.h>
#include <Packages/rtrt/Core/Light.h>
#include <Core/Math/MinMax.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/DpyBase.h>
#include <Packages/rtrt/Core/Shadows/NoShadows.h>
#include <Packages/rtrt/Core/Shadows/HardShadows.h>
#include <Packages/rtrt/Core/Shadows/SingleSampleSoftShadows.h>
#include <Packages/rtrt/Core/Shadows/MultiSampleSoftShadows.h>
#include <Packages/rtrt/Core/Shadows/ScrewyShadows.h>
#include <Packages/rtrt/Core/Shadows/UncachedHardShadows.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>

using namespace rtrt;
using namespace SCIRun;
using std::cerr;

Scene::Scene(Object* obj, const Camera& cam, Image* image0, Image* image1,
	     const Color& bgcolor,
             const Color& cdown,
             const Color& cup,
	     const Plane& groundplane,
	     double ambientscale)
    : obj(obj), camera0(camera0), image0(image0), image1(image1),
      groundplane(groundplane),
      cup(cup), cdown(cdown), origCup_(cup), origCDown_(cdown),
      work("frame tiles")
{
  origAmbientColor_ = Color(1,1,1) * ambientscale;
  ambientColor_     = origAmbientColor_;

  init(cam, bgcolor);
}

void Scene::init(const Camera& cam, const Color& bgcolor)
{
  work.refill(0,0,8);
  shadow_mode = Hard_Shadows;
  camera0=new Camera(cam);
  camera1=new Camera(cam);
  maxdepth=2;
  xtilesize=32;
  ytilesize=2;
  ambient_hack=true;
  shadowobj=0;
  background = new ConstantBackground( bgcolor );
  animate=true;
  hotspots=false;
  frameno=0;
  frametime_fp=0;
  lasttime=0;

  add_shadowmode(ShadowBase::shadowTypeNames[0],new NoShadows());
  add_shadowmode(ShadowBase::shadowTypeNames[1],new SingleSampleSoftShadows());
  add_shadowmode(ShadowBase::shadowTypeNames[2],new HardShadows());
  add_shadowmode(ShadowBase::shadowTypeNames[3],new ScrewyShadows());
  add_shadowmode(ShadowBase::shadowTypeNames[4],new MultiSampleSoftShadows());
  add_shadowmode(ShadowBase::shadowTypeNames[5],new UncachedHardShadows());

  select_shadow_mode( Hard_Shadows );
}

void Scene::add_shadowmode(const char* name, ShadowBase* s)
{
  s->setName(name);
  shadows.add(s);
}

void
Scene::select_shadow_mode( ShadowType st )
{
  shadow_mode = st;
}

Scene::Scene(Object* obj, const Camera& cam, const Color& bgcolor,
             const Color& cdown,
             const Color& cup,
	     const Plane& groundplane,
	     double ambientscale)
    : obj(obj), camera0(camera0), image0(0), image1(0),
      groundplane(groundplane),
      cdown(cdown), cup(cup), origCup_(cup), origCDown_(cdown),
      work("frame tiles")
{
  origAmbientColor_ = Color(1,1,1) * ambientscale;
  ambientColor_     = origAmbientColor_;

  init(cam, bgcolor);
}

Scene::~Scene()
{
    delete obj;
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
    lights.add(light);
}

void Scene::add_per_matl_light(Light* light)
{
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

#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Image.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/MinMax.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Stats.h>
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
      cup(cup), cdown(cdown),
      ambientscale(ambientscale), work("frame tiles")
{
  work.refill(0,0,8);
  shadow_mode=3;
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
  logframes=false;
  frameno=0;
  frametime_fp=0;
  lasttime=0;
}

Scene::Scene(Object* obj, const Camera& cam, const Color& bgcolor,
             const Color& cdown,
             const Color& cup,
	     const Plane& groundplane,
	     double ambientscale)
    : obj(obj), camera0(camera0), image0(0), image1(0),
      groundplane(groundplane),
      cdown(cdown),
      cup(cup),
      ambientscale(ambientscale), work("frame tiles")
{
  work.refill(0,0,8);
  shadow_mode=3;
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
  logframes=false;
  frameno=0;
  frametime_fp=0;
  lasttime=0;
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
    int xres=get_image(which)->get_xres();
    int yres=get_image(which)->get_yres();
    int nx=(xres+xtilesize-1)/xtilesize;
    int ny=(yres+ytilesize-1)/ytilesize;
    int nass=nx*ny;
    work.refill(nass, nworkers, 40);
}

void Scene::add_light(Light* light)
{
    lights.add(light);
}

void Scene::preprocess(double bvscale, int& pp_offset, int& scratchsize)
{
    double maxradius=0;
    for(int i=0;i<lights.size();i++)
	if(lights[i]->radius > maxradius)
	    maxradius=lights[i]->radius;
    maxradius*=bvscale;
    double time=Time::currentSeconds();
    obj->preprocess(maxradius, pp_offset, scratchsize);
    if(shadowobj)
	shadowobj->preprocess(maxradius, pp_offset, scratchsize);
    else
	shadowobj=obj;
    cerr << "Preprocess took " << Time::currentSeconds()-time << " seconds\n";
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

/*
 *  SimpleScene.cc:  Scene for the Real Time Ray Tracer renderer
 *
 *  This module creates a scene for the real time ray tracer.
 *
 *  Written by:
 *   James Bigler
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

// rtrt Core stuff
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
// all the module stuff
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/rtrt/Dataflow/Ports/ScenePort.h>
// general libs
#include <iostream>

namespace rtrt {

using namespace SCIRun;
using namespace std;

class SimpleScene : public Module {
public:
  SimpleScene(GuiContext *ctx);
  virtual ~SimpleScene();
  virtual void execute();
  void tcl_command(GuiArgs& args, void* userdata);
private:
  PhongMaterial *sphere_matl_;
  GuiDouble color_r_;
  GuiDouble color_g_;
  GuiDouble color_b_;
  GuiDouble color_a_;
  GuiDouble reflectance_;
  GuiDouble shininess_;

  void update_sphere_material();
  Scene* make_scene();
  SceneContainerHandle sceneHandle_;
};

DECLARE_MAKER(SimpleScene)

SimpleScene::SimpleScene(GuiContext* ctx)
  : Module("SimpleScene", ctx, Filter, "Scenes", "rtrt"),
    sphere_matl_(0),
    color_r_(ctx->subVar("color-r")),
    color_g_(ctx->subVar("color-g")),
    color_b_(ctx->subVar("color-b")),
    color_a_(ctx->subVar("color-a")),
    reflectance_(ctx->subVar("reflectance")),
    shininess_(ctx->subVar("shininess"))
{
}

SimpleScene::~SimpleScene()
{
}

void SimpleScene::update_sphere_material() 
{
  reset_vars();
  double r=color_r_.get();
  double g=color_g_.get();
  double b=color_b_.get();
  double opacity=color_a_.get();
  double reflectance=reflectance_.get();
  double shininess=shininess_.get();
  if (sphere_matl_) {
    sphere_matl_->set_diffuse(Color(r,g,b));
    sphere_matl_->set_opacity(opacity);
    sphere_matl_->set_reflectance(reflectance);
    sphere_matl_->set_shininess(shininess);
  } else {
    sphere_matl_ = 
      new PhongMaterial(Color(r,g,b), opacity, reflectance, shininess);
  }
}

Scene* SimpleScene::make_scene()
{
  update_sphere_material();

  // materials for the scene
  Material *gray_metal = new MetalMaterial(Color(0.5, 0.5, 0.5));

  // geometry for the scene
  Group *obj_group = new Group();
  for( int i = 0; i < 3; i++ )
    for( int j = 0; j < 3; j++ )
      obj_group->add(new Sphere(sphere_matl_, Point((i-1)*10,(j-1)*10,0), 1));
  obj_group->add(new Rect(gray_metal, Point(0,0,-1.3), 
			  Vector(12,0,0), Vector(0,12,0)));

  // set up all of the parameters for the Scene constructor
  Camera cam(Point(30.678, 2.8381, 16.9925),
	     Point(0, 0, 0),
	     Vector(-0.55671, -0.0136153, 0.830595),
	     47.239);
  double ambient_scale=1.0;
  Color bgcolor(0.1, 0.2, 0.45);
  Color cdown(0.82, 0.62, 0.62);
  Color cup(0.1, 0.3, 0.8);
  Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 1) );
  Scene* scene=new Scene(obj_group, cam, bgcolor, cdown, cup, groundplane,
			 ambient_scale, Arc_Ambient);

  // add a named light
  Light *l = new Light(Point(5,-3,3), Color(1,1,.8)*2, 0);
  l->name_="Spot";
  scene->add_light(l);

  // set the background
  scene->set_background_ptr( new LinearBackground(Color(0.2, 0.4, 0.9),
						  Color(0.0,0.0,0.0),
						  Vector(0,0,1)) );
  // set the shadow mode
  scene->select_shadow_mode( Hard_Shadows );
  return scene;
}

void SimpleScene::execute()
{
  // Get the output port
  SceneOPort *scene_out_port = (SceneOPort *) get_oport("Scene");
  if (!scene_out_port) {
    error("No output port");
    return;
  }
  Scene *scene = make_scene();
  SceneContainer *container = scinew SceneContainer();
  container->put_scene(scene);
  sceneHandle_ = container;
  scene_out_port->send(sceneHandle_);
}

// This is called when the tcl code explicity calls a function other than
// needexecute.
void SimpleScene::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2) {
    args.error("Streamline needs a minor command");
    return;
  } else if (args[1] == "update_sphere_material") {
    update_sphere_material();
  } else {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace rtrt

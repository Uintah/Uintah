/*
 *  GeoProbeScene.cc:  Scene for the Real Time Ray Tracer renderer
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
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/CutMaterial.h>
#include <Packages/rtrt/Core/ColorMap.h>
#include <Packages/rtrt/Core/GeoProbeReader.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Packages/rtrt/Core/HVolume.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/CutVolumeDpy.h>

// all the module stuff
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/rtrt/Dataflow/Ports/ScenePort.h>
#include <Dataflow/Ports/ColorMapPort.h>

// general libs
#include <iostream>

namespace rtrt {

using namespace SCIRun;
using namespace std;

class GeoProbeScene : public Module {
public:
  GeoProbeScene(GuiContext *ctx);
  virtual ~GeoProbeScene();
  virtual void execute();
  void tcl_command(GuiArgs& args, void* userdata);
private:
  VolumeDpy *vdpy;
  int first_execute_;
  int cmap_generation_;
  string execute_string_;
  GuiDouble xa_;
  GuiDouble xb_;
  GuiDouble ya_;
  GuiDouble yb_;
  GuiDouble za_;
  GuiDouble zb_;
  GuiString gpfilename_;
  GuiDouble iso_min_;
  GuiDouble iso_max_;
  GuiDouble iso_val_;

  // Colors for the isosurface
  GuiDouble gui_color_r_;
  GuiDouble gui_color_g_;
  GuiDouble gui_color_b_;
  //  Material *surfmat;
  void update_isosurface_color();

  Object *hvol;
  
  Scene* make_scene(Object *obj);
  SceneContainerHandle sceneHandle_;

  void update_isosurface_material() {}
  void update_isosurface_value();
};

DECLARE_MAKER(GeoProbeScene)

GeoProbeScene::GeoProbeScene(GuiContext* ctx)
  : Module("GeoProbeScene", ctx, Filter, "Scenes", "rtrt"),
    vdpy(0),
    first_execute_(1),
    cmap_generation_(-1),
    execute_string_(""),
    xa_(ctx->subVar("xa")),
    xb_(ctx->subVar("xb")),
    ya_(ctx->subVar("ya")),
    yb_(ctx->subVar("yb")),
    za_(ctx->subVar("za")),
    zb_(ctx->subVar("zb")),
    gui_color_r_(ctx->subVar("color-r")),
    gui_color_g_(ctx->subVar("color-g")),
    gui_color_b_(ctx->subVar("color-b")),
    //    surfmat(0),
    hvol(0),
    gpfilename_(ctx->subVar("gpfilename")),
    iso_min_(ctx->subVar("iso_min")),
    iso_max_(ctx->subVar("iso_max")),
    iso_val_(ctx->subVar("iso_val"))
{
}

GeoProbeScene::~GeoProbeScene()
{
}

Scene* GeoProbeScene::make_scene(Object *obj)
{
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
  Scene* scene=new Scene(obj, cam, bgcolor, cdown, cup, groundplane,
			 ambient_scale, Arc_Ambient);

  // This object needs to be animated
  scene->addObjectOfInterest(obj, true);
  
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

void GeoProbeScene::execute()
{
  // Get the output port
  ColorMapIPort *cmap_iport = (ColorMapIPort *) get_iport("Colormap");
  if (!cmap_iport) {
    error("No colormap input port");
  }

  ColorMapHandle cmH;
  if (!cmap_iport->get(cmH) || !cmH.get_rep()) {
    error("No valid colormap input");
    return;
  }

  SceneOPort *scene_oport = (SceneOPort *) get_oport("Scene");
  if (!scene_oport) {
    error("No scene output port");
    return;
  }

  if (first_execute_) {
    int nx, ny, nz;
    Point min, max;
    unsigned char datamin, datamax;
    Array3<unsigned char> data;
    cerr << "input file = "<<gpfilename_.get()<<"\n";
    if (!read_geoprobe(gpfilename_.get().c_str(), nx, ny, nz, min, max, 
		       datamin, datamax, data)) {
      error("Could not read GeoProbe input file");
      return;
    }
    // Print to the console some information about the data
    printf("dim = (%d, %d, %d)\n", nx, ny, nz);
    cout << "min = "<<min<<", max = "<<max<<"\n";
    cout << "datamin = "<<(int)datamin<<", datamax = "<<(int)datamax<<"\n";
    iso_min_.set(datamin);
    iso_max_.set(datamax);
    double xa = xa_.get();
    double xb = xb_.get();
    double ya = ya_.get();
    double yb = yb_.get();
    double za = za_.get();
    double zb = zb_.get();
    xa = max.x()*xa+min.x()*(1-xa);
    xb = max.x()*xb+min.x()*(1-xb);
    ya = max.y()*ya+min.y()*(1-ya);
    yb = max.y()*yb+min.y()*(1-yb);
    za = max.z()*za+min.z()*(1-za);
    zb = max.z()*zb+min.z()*(1-zb);
    Group *g = new Group;
    Material *surfmat = new LambertianMaterial(Color(gui_color_r_.get(),
						     gui_color_g_.get(),
						     gui_color_b_.get()));
    //    ColorMap *cmap =new ColorMap("/opt/SCIRun/data/Geometry/volumes/vol_cmap");
    //    CutPlaneDpy *cpdpy = new CutPlaneDpy(Vector(1,0,0), Point(xa,0,0));
    //    Material *cutmat = new CutMaterial(surfmat, cmap, cpdpy);
    iso_val_.set(82.5);
    //    CutVolumeDpy *cvdpy = new CutVolumeDpy(iso_val_.get(), cmap);
    vdpy = new VolumeDpy(iso_val_.get());
    vdpy->set_minmax(datamin, datamax);
    hvol = new HVolume<unsigned char, BrickArray3<unsigned char>, 
      BrickArray3<VMCell<unsigned char> > >
      (surfmat, vdpy, 3 /*depth*/, 2 /*np*/, nx, ny, nz, 
       min, max, datamin, datamax, data);
    g->add(hvol);
#if 0
    BBox temp;
    hvol->compute_bounds(temp, 0);
    cout <<"hvol.compte_bounds.min = "<<temp.min()<<", max = "<<temp.max()<<"\n";
    //    cout <<"hvol.min = "<<hvol->min<<", datadiag = "<<hvol->datadiag<<"\n";
    temp.reset();
    g->compute_bounds(temp, 0);
    cout <<"group.min = "<<temp.min()<<", max = "<<temp.max()<<"\n";
#endif
    Scene *scene = make_scene(g);
    SceneContainer *container = scinew SceneContainer();
    container->put_scene(scene);
    sceneHandle_ = container;
    scene_oport->send(sceneHandle_);

    // do everything
    // ...
    first_execute_ = 0;
    cmap_generation_ = cmH->generation;
    execute_string_ = "";
    return;
  }


  if (cmH->generation != cmap_generation_) {
    // need to update the colormap
    // ...
    cmap_generation_ = cmH->generation;
    execute_string_ = "";
  }

  if (execute_string_ == "") return; // nothing to do

  if (execute_string_ == "newfile") {
    // read in new file
    return;
  }

  if (execute_string_ == "newplanes") {
    // set the new cutting plane positions
    return;
  }
}

// This is called when the tcl code explicity calls a function other than
// needexecute.
void GeoProbeScene::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2) {
    args.error("Streamline needs a minor command");
    return;
  } else if (args[1] == "update_plane") {
    want_to_execute();
  } else if (args[1] == "update_isosurface_material") {
    update_isosurface_color();
  } else if (args[1] == "update_isosurface_value") {
    update_isosurface_value();
  } else {
    Module::tcl_command(args, userdata);
  }
}

void GeoProbeScene::update_isosurface_color() {
  if (first_execute_)
    // haven't initialized anything yet
    return;
  
  reset_vars();
  
  // Get a pointer to the old material, so that we can delete it.
  Material *oldmat = hvol->get_matl();
  // Create the new material
  Material *newmat = new LambertianMaterial(Color(gui_color_r_.get(),
						  gui_color_g_.get(),
						  gui_color_b_.get()));
  // Now set the material for the object
  hvol->set_matl(newmat);
  // Delete the old material
  if (oldmat) delete oldmat;
}

void GeoProbeScene::update_isosurface_value() {
  if (first_execute_)
    // haven't initialized anything yet
    return;
  
  reset_vars();
  vdpy->change_isoval((float)(iso_val_.get()));
}


} // End namespace rtrt

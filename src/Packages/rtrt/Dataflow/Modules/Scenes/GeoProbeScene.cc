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
#include <Packages/rtrt/Core/PhongColorMapMaterial.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/CutMaterial.h>
#include <Packages/rtrt/Core/ColorMap.h>
#include <Packages/rtrt/Core/GeoProbeReader.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Packages/rtrt/Core/HVolume.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/CutVolumeDpy.h>
#include <Packages/rtrt/Core/PlaneDpy.h>
#include <Packages/rtrt/Core/CutPlane.h>

// all the module stuff
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/rtrt/Dataflow/Ports/ScenePort.h>
#include <Dataflow/Ports/ColorMapPort.h>

// General SCIRun stuff
#include <Core/Geom/ColorMap.h>
#include <Core/Datatypes/Color.h>

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
  CutPlane *xacp;
  CutPlane *xbcp;
  CutPlane *yacp;
  CutPlane *ybcp;
  CutPlane *zacp;
  CutPlane *zbcp;
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

  // Some state variables
  GuiInt xa_act_;
  GuiInt xa_mat_;
  GuiInt xb_act_;
  GuiInt xb_mat_;
  GuiInt ya_act_;
  GuiInt ya_mat_;
  GuiInt yb_act_;
  GuiInt yb_mat_;
  GuiInt za_act_;
  GuiInt za_mat_;
  GuiInt zb_act_;
  GuiInt zb_mat_;
  
  // Colors for the isosurface
  GuiDouble gui_color_r_;
  GuiDouble gui_color_g_;
  GuiDouble gui_color_b_;
  //  Material *surfmat;
  void update_isosurface_color();

  Object *hvol;
  // The physical extents of the data
  Point min, max;
  
  Scene* make_scene(Object *obj);
  SceneContainerHandle sceneHandle_;

  void update_isosurface_material() {}
  void update_isosurface_value();
  void update_cutting_plane(string which, float val);
  void update_active_cut(string which);
  void update_usemat_cut(string which);

  Material *gen_color_map(ColorMapHandle, Object *obj, float min, float max);
};

DECLARE_MAKER(GeoProbeScene)

GeoProbeScene::GeoProbeScene(GuiContext* ctx)
  : Module("GeoProbeScene", ctx, Filter, "Scenes", "rtrt"),
    vdpy(0),
    first_execute_(1),
    cmap_generation_(-1),
    execute_string_(""),
    xacp(0), xbcp(0), yacp(0), ybcp(0), zacp(0), zbcp(0),
    xa_(ctx->subVar("xa")),
    xb_(ctx->subVar("xb")),
    ya_(ctx->subVar("ya")),
    yb_(ctx->subVar("yb")),
    za_(ctx->subVar("za")),
    zb_(ctx->subVar("zb")),
    gui_color_r_(ctx->subVar("color-r")),
    gui_color_g_(ctx->subVar("color-g")),
    gui_color_b_(ctx->subVar("color-b")),
    xa_act_(ctx->subVar("xa-active")),
    xa_mat_(ctx->subVar("xa-usemat")),
    xb_act_(ctx->subVar("xb-active")),
    xb_mat_(ctx->subVar("xb-usemat")),
    ya_act_(ctx->subVar("ya-active")),
    ya_mat_(ctx->subVar("ya-usemat")),
    yb_act_(ctx->subVar("yb-active")),
    yb_mat_(ctx->subVar("yb-usemat")),
    za_act_(ctx->subVar("za-active")),
    za_mat_(ctx->subVar("za-usemat")),
    zb_act_(ctx->subVar("zb-active")),
    zb_mat_(ctx->subVar("zb-usemat")),
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
  rtrt::Color bgcolor(0.1, 0.2, 0.45);
  rtrt::Color cdown(0.82, 0.62, 0.62);
  rtrt::Color cup(0.1, 0.3, 0.8);
  Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 1) );
  Scene* scene=new Scene(obj, cam, bgcolor, cdown, cup, groundplane,
			 ambient_scale, Arc_Ambient);

  // This object needs to be animated
  scene->addObjectOfInterest(obj, true);
  
  // add a named light
  Light *l = new Light(Point(10,-3,3), rtrt::Color(1,1,.8)*1, 0);
  l->name_="Spot";
  scene->add_light(l);

  // set the background
  scene->set_background_ptr( new LinearBackground(rtrt::Color(0.2, 0.4, 0.9),
						  rtrt::Color(0.0,0.0,0.0),
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
    Group *all_cuts = new Group;
    Material *surfmat = new LambertianMaterial(rtrt::Color(gui_color_r_.get(),
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
    // Add the cutting planes
    Object *obj = hvol;
    // Here is the material
    Material *cut_mat = gen_color_map(cmH, hvol, datamin, datamax);
    // The X clipping planes
    obj = xacp =new CutPlane(obj, Vector(1/(max.x()-min.x()),0,0), xa_.get());
    //    xacp->set_matl(new LambertianMaterial(rtrt::Color(0.1,0.1,0.9)));
    xacp->set_matl(cut_mat);
    
    obj = xbcp =new CutPlane(obj, Vector(1/(min.x()-max.x()),0,0), -xb_.get());
    //xbcp->set_matl(new LambertianMaterial(rtrt::Color(0.9,0.9,0.1)));
    xbcp->set_matl(cut_mat);

    // The Y clipping planes
    obj = yacp =new CutPlane(obj, Vector(0,1/(max.y()-min.y()),0), ya_.get());
    //    yacp->set_matl(new LambertianMaterial(rtrt::Color(0.1,0.5,0.2)));
    yacp->set_matl(cut_mat);
    
    obj = ybcp =new CutPlane(obj, Vector(0,1/(min.y()-max.y()),0), -yb_.get());
    //ybcp->set_matl(new LambertianMaterial(rtrt::Color(0.9,0.2,0.5)));
    ybcp->set_matl(cut_mat);

    // The Z clipping planes
    obj = zacp =new CutPlane(obj, Vector(0,0,1/(max.z()-min.z())), za_.get());
    //    zacp->set_matl(new LambertianMaterial(rtrt::Color(0.9,0.1,0.9)));
    zacp->set_matl(cut_mat);
    
    obj = zbcp =new CutPlane(obj, Vector(0,0,1/(min.z()-max.z())), -zb_.get());
    //zbcp->set_matl(new LambertianMaterial(rtrt::Color(0.2,0.9,0.9)));
    zbcp->set_matl(cut_mat);

    all_cuts->add(obj);
#if 0
    BBox temp;
    hvol->compute_bounds(temp, 0);
    cout <<"hvol.compte_bounds.min = "<<temp.min()<<", max = "<<temp.max()<<"\n";
    //    cout <<"hvol.min = "<<hvol->min<<", datadiag = "<<hvol->datadiag<<"\n";
    temp.reset();
    all_cuts->compute_bounds(temp, 0);
    cout <<"group.min = "<<temp.min()<<", max = "<<temp.max()<<"\n";
#endif
    Scene *scene = make_scene(all_cuts);
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
#if 1
    // need to update the colormap
    Material *cut_mat = gen_color_map(cmH, hvol,
				      iso_min_.get(), iso_max_.get());
    // We know that they all share the same material
    Material *old_mat = xacp->get_matl();
    // Now reset the colormap for all the objects
    xacp->set_matl(cut_mat);
    xbcp->set_matl(cut_mat);
    yacp->set_matl(cut_mat);
    ybcp->set_matl(cut_mat);
    zacp->set_matl(cut_mat);
    zbcp->set_matl(cut_mat);
    // Now delete old_mat.  We should get a handle on all the inside
    // stuff to be sure to delete that stuff too.
    delete old_mat;
#endif
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
  } else if (args[1] == "update_cut") {
    update_cutting_plane(args[2], atof(args[3].c_str()));
  } else if (args[1] == "update_active") {
    update_active_cut(args[2]);
  } else if (args[1] == "update_usemat") {
    update_usemat_cut(args[2]);
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
  Material *newmat = new LambertianMaterial(rtrt::Color(gui_color_r_.get(),
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

void GeoProbeScene::update_cutting_plane(string which, float val) {
  if (first_execute_)
    // haven't initialized anything yet
    return;
  
  //  cout << "Updating cutting plane ";
  CutPlane *plane = 0;
  if (which == "xa") {
    //    cout << "xa";
    plane = xacp;
  } else if (which == "xb") {
    //    cout << "xb";
    plane = xbcp;
    val = -val;
  } else if (which == "ya") {
    //    cout << "ya";
    plane = yacp;
  } else if (which == "yb") {
    //    cout << "yb";
    plane = ybcp;
    val = -val;
  } else if (which == "za") {
    //    cout << "za";
    plane = zacp;
  } else if (which == "zb") {
    //    cout << "zb";
    plane = zbcp;
    val = -val;
  }
  //  cout <<" with value "<<val<<"\n";
  if (plane) {
    if (val == 0 || val == -1)
      val += 1e-3;
    if (val == 1)
      val -= 1e-3;
    plane->update_displacement(val);
  }
}

void GeoProbeScene::update_active_cut(string which) {
  if (first_execute_)
    // haven't initialized anything yet
    return;

  CutPlane *cp = 0;
  bool state = true;
  
  reset_vars();
  if (which == "xa") {
    cp = xacp;
    state = xa_act_.get() == 1;
  } else if (which == "xb") {
    cp = xbcp;
    state = xb_act_.get() == 1;
  } else if (which == "ya") {
    cp = yacp;
    state = ya_act_.get() == 1;
  } else if (which == "yb") {
    cp = ybcp;
    state = yb_act_.get() == 1;
  } else if (which == "za") {
    cp = zacp;
    state = za_act_.get() == 1;
  } else if (which == "zb") {
    cp = zbcp;
    state = zb_act_.get() == 1;
  }

  if (cp) {
    cp->update_active_state(state);
  }
}

void GeoProbeScene::update_usemat_cut(string which) {
  if (first_execute_)
    // haven't initialized anything yet
    return;
  
  CutPlane *cp = 0;
  bool state = true;
  
  reset_vars();
  if (which == "xa") {
    cp = xacp;
    state = xa_mat_.get() == 1;
  } else if (which == "xb") {
    cp = xbcp;
    state = xb_mat_.get() == 1;
  } else if (which == "ya") {
    cp = yacp;
    state = ya_mat_.get() == 1;
  } else if (which == "yb") {
    cp = ybcp;
    state = yb_mat_.get() == 1;
  } else if (which == "za") {
    cp = zacp;
    state = za_mat_.get() == 1;
  } else if (which == "zb") {
    cp = zbcp;
    state = zb_mat_.get() == 1;
  }

  if (cp) {
    cp->update_usemat_state(state);
  }
 
}

// 
Material *GeoProbeScene::gen_color_map(ColorMapHandle cmap, Object *obj,
				       float min, float max) {
  // Need to generate the ScalarTransform1D for color and alpha
  int size = 256;
  Array1<rtrt::Color> *colors = new Array1<rtrt::Color>(256);
  Array1<float> *alphas = new Array1<float>(256);
  for (int i = 0; i < size; i++) {
    double index = (double)i/(size-1);
    SCIRun::Color sc = cmap->getColor(index);
    rtrt::Color rc(sc.r(), sc.g(), sc.b());
    (*colors)[i] = rc;
    (*alphas)[i] = cmap->getAlpha(index);
#if 1
    cout << "colors["<<i<<"] = "<<(*colors)[i];
    cout << ", alpha = "<<(*alphas)[i]<<endl;
#endif
  }
  
  ScalarTransform1D<float, rtrt::Color> *color_trans;
  color_trans = new ScalarTransform1D<float, rtrt::Color>(colors);
  color_trans->scale(min, max);
  
  ScalarTransform1D<float, float> *alpha_trans;
  alpha_trans = new ScalarTransform1D<float, float>(alphas);
  alpha_trans->scale(min, max);

  return new PhongColorMapMaterial(obj, color_trans, alpha_trans);
}

} // End namespace rtrt



#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/PhongColorMapMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Core/Math/MinMax.h>
#include <Core/Geometry/Transform.h>

#include <Core/Thread/Thread.h>
#include <Packages/rtrt/Core/SelectableGroup.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/HVolume.h>
#include <Packages/rtrt/Core/HVolumeBrick16.h>
#include <Packages/rtrt/Core/MIPHVB16.h>
#include <Packages/rtrt/Core/CutVolumeDpy.h>
#include <Packages/rtrt/Core/CutPlaneDpy.h>
#include <Packages/rtrt/Core/CutPlane.h>
#include <Packages/rtrt/Core/ColorMap.h>
#include <Packages/rtrt/Core/ColorMapDpy.h>
#include <Packages/rtrt/Core/CutMaterial.h>
#include <Packages/rtrt/Core/CutGroup.h>
#include <Packages/rtrt/Core/Instance.h>
#include <Packages/rtrt/Core/InstanceWrapperObject.h>
#include <Packages/rtrt/Core/SpinningInstance.h>
#include <Packages/rtrt/Core/DynamicInstance.h>

using namespace rtrt;
using namespace std;
using SCIRun::Thread;

Array1<DpyBase*> dpys;
Array1<Object*> objects_of_interest;

//#define DATA_TYPE float
//#define DATA_LOCATION "/usr/sci/data/Geometry/volumes2/CSAFE"

typedef unsigned char datatype;
#define DATA_LOCATION "/usr/sci/data/CSAFE/quantize"

void add_fire2(Group *g, int nworkers) {
  // Create a default color map.
  int size = 5;
  Array1<ColorPos> colors(size);
  for(int i = 0; i < size; i++) {
    colors[i].x = (float)i/(size-1);
    colors[i].val = 1;
  }
  colors[0].c = Color(1,0,0);
  colors[1].c = Color(1,1,0);
  colors[2].c = Color(0,1,0);
  colors[3].c = Color(0,1,1);
  colors[4].c = Color(0,0,1);

  ColorMapDpy *fcdpy = new ColorMapDpy(colors);
  dpys.add(fcdpy);
  Material *temp_matl = new Phong(Color(1, 0.7, 0.8), Color(1,1,1), 100);
  VolumeDpy *temp_dpy = new VolumeDpy(1000);
  // Don't add the display unless you plan on rendering the temperature field
  //  dpys.add(temp_dpy);
  VolumeDpy *vel_dpy = new VolumeDpy(5.6);
  dpys.add(vel_dpy);

  ////////////////////////////////////////////////////////////////
  //  Load up a group
  
  int fstart = 32;
  int fend = 152;
  //  int fstart = 64;
  //  int fend = 80;
  int finc = 8; // never less than 8, must be a multiple of 8
  //  int finc = 16; // 0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160
  //  int finc = 24; // 0, 24, 48, 72, 96, 120, 144, 168
  //  int finc = 32; // 0, 32, 64, 96, 128, 160
  //  int finc = 40; // 0, 40, 80, 120, 160
  SelectableGroup *fire_geom = new SelectableGroup(0.5);
  fire_geom->set_name("CSAFE Fire Time Step Selector");
  //  TimeObj *fire_time = new TimeObj(5);
  for(int f = fstart; f <= fend; f+= finc) {
    char buf[1000];
    // load in the temperature field to color by
    sprintf(buf, DATA_LOCATION"/h300_%04d.uchar.raw", f);
    //    cout << "Reading "<<buf<<endl;
    // Send 1 for depth since we don't need to render it and preprocess
    // goes much quickley.
    HVolume<datatype, BrickArray3<datatype>, BrickArray3<VMCell<datatype> > > *tempt = 
      new HVolume<datatype, BrickArray3<datatype>, BrickArray3<VMCell<datatype> > >
      (temp_matl, temp_dpy, buf, 1, nworkers);

    // Get min and max and attach to ColorMapDpy.
    float min, max;
    tempt->get_minmax(min, max);
    fcdpy->attach(min,max);

    // Create the Material
    Material *temp_color_map = new PhongColorMapMaterial
      (tempt,
       fcdpy->get_color_transfer_pointer(),
       fcdpy->get_alpha_transfer_pointer());

    // Load the velocity field that will be used to isosurface.
    sprintf(buf, DATA_LOCATION"/heptane300_velmag_%04d.uchar.raw", f);
    HVolume<datatype, BrickArray3<datatype>, BrickArray3<VMCell<datatype> > > *vel =
      new HVolume<datatype, BrickArray3<datatype>, BrickArray3<VMCell<datatype> > >
      (temp_color_map, vel_dpy, buf, 3, nworkers);


    fire_geom->add(vel);
  }

#if 0
  // Can't do cutting plane on time dependant data just yet.  There's a
  // CycleMaterial that may be able to work, but I haven't checked it out.
  CutPlaneDpy* pd=new CutPlaneDpy(Vector(0,1,0), Point(0,0,0));
  dpys.add(pd);
  //  (new Thread(pd, "Cutting plane display thread"))->detach();
  Object *obj = new CutPlane(fire_geom, pd);
  obj->set_matl(hmat);
#endif
  g->add(fire_geom);
  objects_of_interest.add(fire_geom);
}

void add_fire(Group *g, int nworkers) {
  Array1<float> *opacity = new Array1<float>(5);
  Array1<Color> *color = new Array1<Color>(5);
  for(int i = 0; i < 5; i++)
    (*opacity)[i] = 1;
  //  (*opacity)[0] = 0.5;
  //  (*opacity)[3] = 0.1;
  (*color)[0] = Color(1,0,0);
  (*color)[1] = Color(1,1,0);
  (*color)[2] = Color(0,1,0);
  (*color)[3] = Color(0,1,1);
  (*color)[4] = Color(0,0,1);

  ScalarTransform1D<float, float> *opacity_transform =
    new ScalarTransform1D<float, float>(opacity);
  ScalarTransform1D<float, Color> *color_transform =
    new ScalarTransform1D<float, Color>(color);
  // need to the get the min and max


  //82.5 for dave
  //  CutVolumeDpy* hcvdpy = new CutVolumeDpy(11000.0, hcmap);
  Material *firematl = new Phong(Color(1, 0.7, 0.8), Color(1,1,1), 100);
  VolumeDpy *fire_dpy = new VolumeDpy(1000);
  //  VolumeDpy *fire_dpy = new VolumeDpy(11000);
  //  (new Thread(fire_dpy, "Fire VolumeDpy Thread"))->detach();
  //  dpys.add(fire_dpy);

  // load the temperature field for coloring
  HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > > *fire =
    new HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > >
    (firematl, fire_dpy,
     "/usr/sci/data/Geometry/volumes2/CSAFE/h300_0064f.raw",
     1, nworkers);
  
  float min, max;
  fire->get_minmax(min, max);
  opacity_transform->scale(min,max);
  color_transform->scale(min,max);
  Material *hmat = new PhongColorMapMaterial(fire, color_transform,
					     opacity_transform);

  // Load the velocity field that will be used to isosurface.
  VolumeDpy *vel_dpy = new VolumeDpy(1);
  dpys.add(vel_dpy);
  HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > > *vel =
    new HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > >
    (hmat, vel_dpy,
     "/usr/sci/data/Geometry/volumes2/CSAFE/heptane300_velmag_0064.raw",
     3, nworkers);

  CutPlaneDpy* pd=new CutPlaneDpy(Vector(0,1,0), Point(0,0,0));
  dpys.add(pd);
  //  (new Thread(pd, "Cutting plane display thread"))->detach();
  Object *obj = new CutPlane(vel, pd);
  obj->set_matl(hmat);
  g->add(obj);
  objects_of_interest.add(obj);
}

void add_head(Group *g, int nworkers) {

  //82.5 for dave
  //  CutVolumeDpy* hcvdpy = new CutVolumeDpy(11000.0, hcmap);
  Material *headmatl = new Phong(Color(1, 0.7, 0.8), Color(1,1,1), 100);
  VolumeDpy *head_dpy = new VolumeDpy(300);
  //  VolumeDpy *head_dpy = new VolumeDpy(11000);
  //  (new Thread(head_dpy, "Head VolumeDpy Thread"))->detach();
  dpys.add(head_dpy);

  Object* head=new HVolumeBrick16(headmatl, head_dpy,
					  //    "/usr/sci/data/Geometry/volumes2/dave",
				  //					  "/usr/sci/data/Geometry/volumes2/gk2-anat-US.raw",
				  "/usr/sci/data/CSAFE/heptane300_3D_NRRD/h300_0072_short.raw",
					      3, nworkers);
  
  Array1<float> *opacity = new Array1<float>(5);
  Array1<Color> *color = new Array1<Color>(5);
  for(int i = 0; i < 5; i++)
    (*opacity)[i] = 1;
  (*opacity)[0] = 0.5;
  (*opacity)[3] = 0.1;
  (*color)[0] = Color(1,0,0);
  (*color)[1] = Color(1,1,0);
  (*color)[2] = Color(0,1,0);
  (*color)[3] = Color(0,1,1);
  (*color)[4] = Color(0,0,1);

  ScalarTransform1D<float, float> *opacity_transform =
    new ScalarTransform1D<float, float>(opacity);
  ScalarTransform1D<float, Color> *color_transform =
    new ScalarTransform1D<float, Color>(color);
  Material *hmat = new PhongColorMapMaterial(head, color_transform,
					     opacity_transform);

  CutPlaneDpy* pd=new CutPlaneDpy(Vector(0,1,0), Point(0,0,0));
  dpys.add(pd);
  //  (new Thread(pd, "Cutting plane display thread"))->detach();
  head = new CutPlane(head, pd);
  head->set_matl(hmat);
  g->add(head);
  objects_of_interest.add(head);


#if 0
  //ADD THE HEAD DATA SET
  ColorMap *hcmap = new ColorMap("/usr/sci/data/Geometry/volumes2/head",256);
  Material *hmat=new LambertianMaterial(Color(0.7,0.7,0.7));
  hmat->my_lights.add(holo_light1);
  hmat->my_lights.add(holo_light2);
  hmat->my_lights.add(holo_light3);

  Material *hcutmat = new CutMaterial(hmat, hcmap, cpdpy);
  hcutmat->my_lights.add(holo_light1);
  hcutmat->my_lights.add(holo_light2);
  hcutmat->my_lights.add(holo_light3);

  InstanceWrapperObject *hiw = new InstanceWrapperObject(head);

  Transform *htrans = new Transform();
  htrans->rotate(Vector(1,0,0), Vector(0,0,-1));
  htrans->pre_scale(Vector(1.11,1.11,1.11)); //scale to fit max
  htrans->pre_translate(Vector(-8, 8, 1.75));
  htrans->pre_translate(Vector(0,0,-0.352)); //place 1cm above table

  SpinningInstance *hinst = new SpinningInstance(hiw, htrans, Point(-8,8,1.56), Vector(0,0,1), 0.1);
  
  hinst->name_ = "Spinning Head";

  CutGroup *hcut = new CutGroup(cpdpy);
  hcut->add(hinst);
  hcut->name_ = "Cutting Plane";
#endif
}

extern "C"
Scene* make_scene(int /*argc*/, char* /*argv[]*/, int nworkers)
{

  Group *g = new Group();
  //  g->add(new Sphere(new Phong(Color(1,0.5,0.5), Color(1,1,1), 100),
  //		    Point(0,0,0), 1));
  //  add_head(g, nworkers);
  //  add_fire(g, nworkers);
  add_fire2(g, nworkers);
  
  
		    
  Color cdown(0.1, 0.1, 0.1);
  Color cup(0.1, 0.1, 0.1);

  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(0.3, 0.3, 0.3);

  Point Eye(-11.85, 8.05916, 1.30671);
  Point Lookat(-8.83055, 8.24346, 1.21209);
  Vector Up(0,0,1);
  double fov=45;
  Camera cam(Eye,Lookat,Up,fov);

  Scene *scene = new Scene(g, cam, bgcolor, cdown, cup, groundplane, 0.3);

  ///////////////////////////////////////////////////////////
  // Add the interesting stuff
  for(int i = 0; i < dpys.size(); i++) {
    scene->attach_display(dpys[i]);
    (new Thread(dpys[i], "volume_color display thread"))->detach();
  }
  for(int i = 0; i < objects_of_interest.size(); i++)
    scene->addObjectOfInterest(objects_of_interest[i], true);
  
  ///////////////////////////////////////////////////////////
  // Set up the scene parameters
  scene->select_shadow_mode( No_Shadows );
  scene->maxdepth = 8;
  Light *science_room_light0 = new Light(Point(-8, 8, 3.9), Color(.5,.5,.5), 0, .3);
  science_room_light0->name_ = "science room overhead";
  scene->add_light(science_room_light0);
//  Light *science_room_light1 = new Light(Point(-5, 11, 3), Color(.5,.5,.5), 0);
//  science_room_light1->name_ = "science room corner";
//  scene->add_light(science_room_light1);
  Light *science_room_light1 = new Light(Point(-5, 8, 3), Color(.5,.5,.5), 0, .3);
  science_room_light1->name_ = "science room corner1";
  scene->add_light(science_room_light1);
  Light *science_room_light2 = new Light(Point(-8, 5, 3), Color(.5,.5,.5), 0, .3);
  science_room_light2->name_ = "science room corner2";
  scene->add_light(science_room_light2);
  scene->animate=true;


  return scene;
}

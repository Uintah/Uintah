#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/HierarchicalGrid.h>
#include <Packages/rtrt/Core/Disc.h>
#include <Packages/rtrt/Core/Ring.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <Packages/rtrt/Core/Point4D.h>
#include <Packages/rtrt/Core/CrowMarble.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Mesh.h>
#include <Packages/rtrt/Core/ASEReader.h>
#include <Packages/rtrt/Core/ObjReader.h>
#include <Packages/rtrt/Core/Bezier.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/Checker.h>
#include <Packages/rtrt/Core/Speckle.h>
#include <Packages/rtrt/Core/Box.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Core/Math/MinMax.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/Parallelogram.h>
#include <Packages/rtrt/Core/Cylinder.h>


#include <Core/Thread/Thread.h>
#include <Packages/rtrt/Core/SelectableGroup.h>
#include <Packages/rtrt/Core/HVolumeBrick16.h>
#include <Packages/rtrt/Core/MIPHVB16.h>
#include <Packages/rtrt/Core/CutVolumeDpy.h>
#include <Packages/rtrt/Core/CutPlaneDpy.h>
#include <Packages/rtrt/Core/ColorMap.h>
#include <Packages/rtrt/Core/CutMaterial.h>
#include <Packages/rtrt/Core/CutGroup.h>
#include <Packages/rtrt/Core/Instance.h>
#include <Packages/rtrt/Core/InstanceWrapperObject.h>
#include <Packages/rtrt/Core/SpinningInstance.h>
#include <Packages/rtrt/Core/DynamicInstance.h>

using namespace rtrt;
using SCIRun::Thread;

#define MAXBUFSIZE 256
#define SCALE 950

extern "C"
Scene* make_scene(int argc, char* argv[], int nworkers)
{
  for(int i=1;i<argc;i++) {
    cerr << "Unknown option: " << argv[i] << '\n';
    cerr << "Valid options for scene: " << argv[0] << '\n';
    return 0;
  }

//  Point Eye(-5.85, 6.2, 2.0);
//  Point Lookat(-13.5, 13.5, 2.0);
//  Vector Up(0,0,1);
  Point Eye(-10.9055, -0.629515, 1.56536);
  Point Lookat(-8.07587, 15.7687, 1.56536);
  Vector Up(0, 0, 1);
  double fov=60;

  Camera cam(Eye,Lookat,Up,fov);

  Material* marble1=new CrowMarble(5.0,
				   Vector(2,1,0),
				   Color(0.5,0.6,0.6),
				   Color(0.4,0.55,0.52),
				   Color(0.35,0.45,0.42));
//  marble1->my_lights.add(new Light(Point(-7.5,7.5,3.9), Color(.7,.4,.4),0));
  
  Material* marble2=new CrowMarble(7.5,
				   Vector(-1,3,0),
				   Color(0.4,0.3,0.2),
				   Color(0.35,0.34,0.32),
				   Color(0.20,0.24,0.24));
  Material* marble=new Checker(marble1,
			       marble2,
			       Vector(3,0,0), Vector(0,3,0));
  Object* check_floor=new Rect(marble, Point(-8, 8, 0),
			       Vector(4, 0, 0), Vector(0, 4, 0));
  Group* north_wall=new Group();
  Group* west_wall=new Group();
  Group* south_wall=new Group();
  Group* east_wall=new Group();
  Group* table=new Group();
  Group* ceiling_floor=new Group();
  ceiling_floor->add(check_floor);

  Material* whittedimg = 
    new ImageMaterial(1, "/usr/sci/data/Geometry/textures/whitted",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  
  Object* pic1=
    new Parallelogram(whittedimg, Point(-7.35, 11.9, 2.5), 
		      Vector(0,0,-1), Vector(-1.3,0,0));

  Material* bumpimg = 
    new ImageMaterial(1, "/usr/sci/data/Geometry/textures/bump",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);

  Object* pic2=
    new Parallelogram(bumpimg, Point(-11.9, 7.35, 2.5), 
		      Vector(0, 0, -1), Vector(0, 1.3, 0));

  Material* white = new LambertianMaterial(Color(0.8,0.8,0.8));

  north_wall->add(new Rect(white, Point(-8, 12, 2), 
		       Vector(4, 0, 0), Vector(0, 0, 2)));

  west_wall->add(new Rect(white, Point(-12, 8, 2), 
		       Vector(0, 4, 0), Vector(0, 0, 2)));

//  south_wall->add(new Rect(white, Point(-8, 4, 2), 
//		       Vector(4, 0, 0), Vector(0, 0, 2)));

  // doorway cut out of South wall for W. tube: attaches to Graphic Museum scene

  south_wall->add(new Rect(white, Point(-11.5, 4, 2), 
		       Vector(0.5, 0, 0), Vector(0, 0, 2)));
  south_wall->add(new Rect(white, Point(-7.5, 4, 3), 
		       Vector(3.5, 0, 0), Vector(0, 0, 1)));
  south_wall->add(new Rect(white, Point(-6.5, 4, 1), 
		       Vector(2.5, 0, 0), Vector(0, 0, 1)));

//  east_wall->add(new Rect(white, Point(-4, 8, 2), 
//		       Vector(0, 4, 0), Vector(0, 0, 2)));

  // doorway cut out of East wall for N. tube: attaches to Living Room scene

  east_wall->add(new Rect(white, Point(-4, 11.5, 2), 
		       Vector(0, 0.5, 0), Vector(0, 0, 2)));
  east_wall->add(new Rect(white, Point(-4, 7.5, 3), 
		       Vector(0, 3.5, 0), Vector(0, 0, 1)));
  east_wall->add(new Rect(white, Point(-4, 6.5, 1), 
		       Vector(0, 2.5, 0), Vector(0, 0, 1)));

  // add the ceiling
//  ceiling_floor->add(new Rect(white, Point(-8, 8, 4),
//		       Vector(4, 0, 0), Vector(0, 4, 0)));

  west_wall->add(pic1);
  north_wall->add(pic2);

  Material *silver = new MetalMaterial(Color(0.7,0.73,0.8), 12);
  //Material *air_to_glass = new DielectricMaterial(1.5, 0.66, 0.04, 400.0, Color(.87, .80, .93), Color(1,1,1), false);
  
  // top of the table is at 32 inches
  double i2m = 1./36.;             // convert inches to meters
  Point center(-8, 8, 0);

  // table base
  table->add(new Cylinder(silver, center+Vector(0,0,0.01)*i2m,
			  center+Vector(0,0,24)*i2m, 12*i2m));
  // table top
  table->add(new Cylinder(silver, center+Vector(0,0,24)*i2m,
			  center+Vector(0,0,36)*i2m, 30*i2m));
  table->add(new Disc(silver, center+Vector(0,0,24)*i2m,
		      Vector(0,0,-1)*i2m, 30*i2m));
  table->add(new Disc(silver, center+Vector(0,0,36)*i2m,
		      Vector(0,0,1)*i2m, 30*i2m));

  // TODO: need a way to bevel the corners with a normal map

  Group *g = new Group();

//  g->add(new BV1(north_wall));
//  g->add(new BV1(west_wall));
//  g->add(new BV1(south_wall));
//  g->add(new BV1(east_wall));

  g->add(ceiling_floor);
  g->add(north_wall);
  g->add(west_wall);
  g->add(south_wall);
  g->add(east_wall);
  g->add(table);

  Transform room_trans;
  room_trans.pre_translate(center.vector());

  string pathname("/usr/sci/data/Geometry/models/science-room/");
  Array1<string> names;
  names.add(string("386dx"));
  names.add(string("3dglasses1"));
  names.add(string("3dglasses2"));
  names.add(string("abacus"));
  names.add(string("coffee-mug2"));
  names.add(string("coffee-mug3"));
  names.add(string("coffee-mug4"));
  names.add(string("coffee-mug5"));
  names.add(string("corbu-float1"));
  names.add(string("corbu-float2"));
  names.add(string("corbu-float3"));
  names.add(string("corbu-float4"));
  names.add(string("corbu-float5"));
  names.add(string("counterbase"));
  names.add(string("countertop"));
  names.add(string("end-table1"));
  names.add(string("end-table2"));
  names.add(string("end-table3"));
  names.add(string("end-table4"));
  names.add(string("end-table5"));
  names.add(string("futuristic-curio1"));
  names.add(string("futuristic-curio2"));
  names.add(string("hmd"));
  names.add(string("microscope"));
  names.add(string("molecule"));
  names.add(string("plant-01"));
  
  Array1<Material *> matls;
  for (int i=0; i<names.size(); i++) {
    cerr << "Reading: "<<names[i]<<"\n";
    string objname(pathname+names[i]+string(".obj"));
    string mtlname(pathname+names[i]+string(".mtl"));
    if (!readObjFile(objname, mtlname, room_trans, g))
      exit(0);
  }



  //PER MATERIAL LIGHTS FOR THE HOLOGRAMS
  Light *holo_light0 = new Light(Point(-10, 10, 3), Color(0.4,0.4,0.6), 0);
  Light *holo_light1 = new Light(Point(-10, 6, 3), Color(0.4,0.4,0.6), 0);
  Light *holo_light2 = new Light(Point(-8, 8, 2), Color(0.4,0.4,0.6), 0);
  Light *holo_light3 = new Light(Point(-6, 6, 3), Color(0.4,0.4,0.6), 0);
  holo_light0->name_ = "hololight0";
  holo_light1->name_ = "hololight1";
  holo_light2->name_ = "hololight2";
  holo_light3->name_ = "hololight3";

  //CUTTING PLANE ACROSS SEVERAL OBJECTS
  CutPlaneDpy* cpdpy=new CutPlaneDpy(Vector(.45,-.45,-.76), Point(-8,8,2));
  ColorMap *cmap = new ColorMap("/usr/sci/data/Geometry/volumes/vol_cmap");


  //ADD THE VISIBLE FEMALE DATASET
  Material* vmat=new LambertianMaterial(Color(0.7,0.7,0.7));
  vmat->my_lights.add(holo_light0);
  vmat->my_lights.add(holo_light1);
  vmat->my_lights.add(holo_light2);
  vmat->my_lights.add(holo_light3);

  Material *vcutmat = new CutMaterial(vmat, cmap, cpdpy);
  vcutmat->my_lights.add(holo_light0);
  vcutmat->my_lights.add(holo_light1);
  vcutmat->my_lights.add(holo_light2);
  vcutmat->my_lights.add(holo_light3);

  CutVolumeDpy* vcvdpy = new CutVolumeDpy(1224.5, cmap);
  
  HVolumeBrick16* slc0=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/usr/sci/data/Geometry/volumes/vfem16_0",
					  3, nworkers);
  
  HVolumeBrick16* slc1=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/usr/sci/data/Geometry/volumes/vfem16_1",
					  3, nworkers);
  
  HVolumeBrick16* slc2=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/usr/sci/data/Geometry/volumes/vfem16_2",
					  3, nworkers);

  HVolumeBrick16* slc3=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/usr/sci/data/Geometry/volumes/vfem16_3",
					  3, nworkers);
  
  HVolumeBrick16* slc4=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/usr/sci/data/Geometry/volumes/vfem16_4",
					  3, nworkers);
  
  HVolumeBrick16* slc5=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/usr/sci/data/Geometry/volumes/vfem16_5",
					  3, nworkers);

  HVolumeBrick16* slc6=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/usr/sci/data/Geometry/volumes/vfem16_6",
					  3, nworkers);
					  

  Group *vig = new Group();
  vig->add(slc0);
  vig->add(slc1);
  vig->add(slc2);
  vig->add(slc3);
  vig->add(slc4);
  vig->add(slc5);
  vig->add(slc6);
  InstanceWrapperObject *viw = new InstanceWrapperObject(vig);

  Transform *vtrans = new Transform();
  vtrans->pre_translate(Vector(8, -8, -2));
  vtrans->pre_rotate(3.14/2.0, Vector(0,1,0));
  vtrans->pre_translate(Vector(-8, 8, 2));

  SpinningInstance *vinst = new SpinningInstance(viw, vtrans, Point(-8,8,2), Vector(0,0,1), 0.5);
  vinst->name_ = "Spinning Visible Woman";

  CutGroup *vcut = new CutGroup(cpdpy);
  vcut->add(vinst);

  //ADD THE DAVE HEAD DATA SET
  Material* hmat=new LambertianMaterial(Color(0.7,0.7,0.7));
  hmat->my_lights.add(holo_light0);
  hmat->my_lights.add(holo_light1);
  hmat->my_lights.add(holo_light2);
  hmat->my_lights.add(holo_light3);

  Material *hcutmat = new CutMaterial(hmat, cmap, cpdpy);
  hcutmat->my_lights.add(holo_light0);
  hcutmat->my_lights.add(holo_light1);
  hcutmat->my_lights.add(holo_light2);
  hcutmat->my_lights.add(holo_light3);

  CutVolumeDpy* hcvdpy = new CutVolumeDpy(82.5, cmap);

  HVolumeBrick16* davehead=new HVolumeBrick16(hcutmat, hcvdpy,
					      "/usr/sci/data/Geometry/volumes/dave",
					      3, nworkers);
  InstanceWrapperObject *diw = new InstanceWrapperObject(davehead);

  Transform *dtrans = new Transform();
  dtrans->pre_translate(Vector(8, -8, -2)); 
  dtrans->rotate(Vector(1,0,0), Vector(0,0,1));
  dtrans->pre_translate(Vector(-8, 8, 2));

  SpinningInstance *dinst = new SpinningInstance(diw, dtrans, Point(-8,8,2), Vector(0,0,1), 0.1);
  dinst->name_ = "Spinning Head";

  CutGroup *hcut = new CutGroup(cpdpy);
  hcut->add(dinst);
  hcut->name_ = "Cutting Plane";

  
  SelectableGroup *sg = new SelectableGroup(10);
  sg->add(vcut);
  sg->add(hcut);
  sg->name_ = "VolVis Switching Items";
  g->add(sg);


  Color cdown(0.1, 0.1, 0.1);
  Color cup(0.1, 0.1, 0.1);

  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(0.3, 0.3, 0.3);

  
  Scene *scene = new Scene(new HierarchicalGrid(g, 8, 8, 8, 20, 20, 5),
			   cam, bgcolor, cdown, cup, groundplane, 0.3);

  scene->select_shadow_mode( Hard_Shadows );
  scene->maxdepth = 8;
  Light *science_room_light0 = new Light(Point(-8, 8, 3.9), Color(.5,.5,.5), 0);
  science_room_light0->name_ = "science room overhead";
  scene->add_light(science_room_light0);
  Light *science_room_light1 = new Light(Point(-11, 11, 3), Color(.5,.5,.5), 0);
  science_room_light1->name_ = "science room corner";
  scene->add_light(science_room_light1);
  scene->animate=true;

  scene->addObjectOfInterest( sg, true );
  scene->addObjectOfInterest( vinst, false );
  scene->addObjectOfInterest( dinst, false );
  scene->addObjectOfInterest( hcut, false );

  scene->add_per_matl_mood_light(holo_light0);
  scene->add_per_matl_mood_light(holo_light1);
  scene->add_per_matl_mood_light(holo_light2);
  scene->add_per_matl_mood_light(holo_light3);

  scene->attach_auxiliary_display(vcvdpy);
  scene->attach_auxiliary_display(hcvdpy);
  scene->attach_auxiliary_display(cpdpy);

  scene->attach_display(vcvdpy);
  (new Thread(vcvdpy, "VFEM Volume Dpy"))->detach();
  scene->attach_display(hcvdpy);
  (new Thread(hcvdpy, "HEAD Volume Dpy"))->detach();
  scene->attach_display(cpdpy);
  (new Thread(cpdpy, "CutPlane Dpy"))->detach();


  return scene;
}

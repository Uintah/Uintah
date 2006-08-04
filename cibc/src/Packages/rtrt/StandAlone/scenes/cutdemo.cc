#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Grid.h>
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

  /*
  //between doors
  Point Eye(-5.85, 6.2, 1.6);
  Point Lookat(-13.5, 13.5, 2.0);
  Vector Up(0,0,1);
  double fov=60;
  */

  /*
  //outside room
  Point Eye(-10.9055, -0.629515, 1.56536);
  Point Lookat(-8.07587, 15.7687, 1.56536);
  Vector Up(0, 0, 1);
  double fov=60;
  */

  //centered above room
  Point Eye(-8, 8, 11);
  Point Lookat(-8, 8, 0);
  Vector Up(0, 1, 0);
  double fov=40;

  Camera cam(Eye,Lookat,Up,fov);


  Material* marble1=new CrowMarble(5.0,
				   Vector(2,1,0),
				   Color(0.5,0.6,0.6),
				   Color(0.4,0.55,0.52),
				   Color(0.35,0.45,0.42));
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


  Material* white = new LambertianMaterial(Color(0.8,0.8,0.8));

  north_wall->add(new Rect(white, Point(-8, 12, 2), 
		       Vector(4, 0, 0), Vector(0, 0, 2)));

  west_wall->add(new Rect(white, Point(-12, 8, 2), 
		       Vector(0, 4, 0), Vector(0, 0, 2)));

  south_wall->add(new Rect(white, Point(-11.5, 4, 2), 
		       Vector(0.5, 0, 0), Vector(0, 0, 2)));
  south_wall->add(new Rect(white, Point(-7.5, 4, 3), 
		       Vector(3.5, 0, 0), Vector(0, 0, 1)));
  south_wall->add(new Rect(white, Point(-6.5, 4, 1), 
		       Vector(2.5, 0, 0), Vector(0, 0, 1)));

  east_wall->add(new Rect(white, Point(-4, 11.5, 2), 
		       Vector(0, 0.5, 0), Vector(0, 0, 2)));
  east_wall->add(new Rect(white, Point(-4, 7.5, 3), 
		       Vector(0, 3.5, 0), Vector(0, 0, 1)));
  east_wall->add(new Rect(white, Point(-4, 6.5, 1), 
		       Vector(0, 2.5, 0), Vector(0, 0, 1)));


  Material *silver = new MetalMaterial(Color(0.7,0.73,0.8), 12);
  Material *air_to_glass = new DielectricMaterial(1.5, 0.66, 0.04, 400.0, Color(.87, .80, .93), Color(1,1,1), false);
  
  // top of the table is at 32 inches
  double i2m = 1./36.;             // convert inches to meters
  Point center(-8, 8, 0);

#if 0
  // N/S horizontal bar to support glass
  table->add(new Box(silver, center+Vector(-1,-24,31.85)*i2m, 
		     center+Vector(1,24,32.15)*i2m));
  
  // E/W horizontal bar to support glass
  table->add(new Box(silver, center+Vector(-24,-1,31.85)*i2m,
		     center+Vector(24,1,32.15)*i2m));
		     
  
  // connecting circle for glass supports
  table->add(new Cylinder(silver, center+Vector(0,0,32.15)*i2m, 
			  center+Vector(0,0,31.85)*i2m, 3*i2m));
  table->add(new Disc(silver, center+Vector(0,0,32.15)*i2m, 
			  Vector(0,0,1), 3*i2m));
  table->add(new Disc(silver, center+Vector(0,0,31.85)*i2m, 
			  Vector(0,0,-1), 3*i2m));

  // glass
  table->add(new Cylinder(air_to_glass, center+Vector(0,0,32.151)*i2m,
			  center+Vector(0,0,32.451)*i2m, 23.75*i2m));
  table->add(new Disc(air_to_glass, center+Vector(0,0,32.451)*i2m,
		      Vector(0,0,1), 23.75*i2m));
  table->add(new Disc(air_to_glass, center+Vector(0,0,32.151)*i2m,
		      Vector(0,0,-1), 23.75*i2m));

  // rim
  table->add(new Cylinder(silver, center+Vector(0,0,32.151)*i2m,
			  center+Vector(0,0,32.451)*i2m, 23.80*i2m));
  table->add(new Cylinder(silver, center+Vector(0,0,32.151)*i2m,
			  center+Vector(0,0,32.451)*i2m, 24.80*i2m));
  table->add(new Ring(silver, center+Vector(0,0,32.151)*i2m, Vector(0,0,-1),
		      23.80*i2m, 1.*i2m));
  table->add(new Ring(silver, center+Vector(0,0,32.451)*i2m, Vector(0,0,1),
		      23.80*i2m, 1.*i2m));

  // N leg
  table->add(new Box(silver, center+Vector(22.8,-1,0)*i2m, 
		     center+Vector(24.8,1,31.672)*i2m));
  
  // S leg
  table->add(new Box(silver, center+Vector(-24.8,-1,0)*i2m, 
		     center+Vector(-22.8,1,31.672)*i2m));
  
  // E leg
  table->add(new Box(silver, center+Vector(-1,22.8,0)*i2m, 
		     center+Vector(1,24.8,31.672)*i2m));
  
  // W leg
  table->add(new Box(silver, center+Vector(-1,-24.8,0)*i2m, 
		     center+Vector(1,-22.8,31.672)*i2m));

#endif
  
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
  //g->add(table);

  Array1<Material *> matls;
  string env_map;

#ifdef CHAIRS
  Transform corbusier_trans;

  // first, get it centered at the origin (in x and y), and scale it
  corbusier_trans.pre_scale(Vector(0.007, 0.007, 0.007));
  corbusier_trans.pre_translate(Vector(0.5,0.2,0));

  // now rotate/translate it to the right angle/position
  for (int i=0; i<5; i++) {
    Transform t(corbusier_trans);
    double rad=(65+35*i)*(M_PI/180.);
    t.pre_rotate(rad, Vector(0,0,1));
    t.pre_translate(center.vector()+Vector(cos(rad),sin(rad),0)*2.9);
    if (!readASEFile("/usr/sci/projects/rtrt/geometry/lebebe.ASE", t, g, matls, env_map)) {
      exit(0);
    }
  }
#endif

#define DOVFEM
#define DODAVE
#define DOINST
#define DOSPIN
#define DOCUT

#ifdef DOVFEM
  Material* vmat=new LambertianMaterial(Color(0.5,0.5,0.5));
  VolumeDpy* vdpy = new VolumeDpy(1650);


  HVolumeBrick16* slc0=new HVolumeBrick16(vmat, vdpy,
					  "/opt/SCIRun/data/Geometry/volumes/vfem16_0",
					  3, nworkers);
  
  HVolumeBrick16* slc1=new HVolumeBrick16(vmat, vdpy,
					  "/opt/SCIRun/data/Geometry/volumes/vfem16_1",
					  3, nworkers);
  
  HVolumeBrick16* slc2=new HVolumeBrick16(vmat, vdpy,
					  "/opt/SCIRun/data/Geometry/volumes/vfem16_2",
					  3, nworkers);

  HVolumeBrick16* slc3=new HVolumeBrick16(vmat, vdpy,
					  "/opt/SCIRun/data/Geometry/volumes/vfem16_3",
					  3, nworkers);
  
  HVolumeBrick16* slc4=new HVolumeBrick16(vmat, vdpy,
					  "/opt/SCIRun/data/Geometry/volumes/vfem16_4",
					  3, nworkers);
  
  HVolumeBrick16* slc5=new HVolumeBrick16(vmat, vdpy,
					  "/opt/SCIRun/data/Geometry/volumes/vfem16_5",
					  3, nworkers);

  HVolumeBrick16* slc6=new HVolumeBrick16(vmat, vdpy,
					  "/opt/SCIRun/data/Geometry/volumes/vfem16_6",
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
  //vtrans->pre_rotate(3.14*(5.0/4.0), Vector(0,0,1));
  //vtrans->pre_rotate(.314, Vector(.7,.7,0));
  vtrans->pre_translate(Vector(-8, 8, 2));

#ifdef DOINST
#ifdef DOSPIN
  SpinningInstance *vinst = new SpinningInstance(viw, vtrans, Point(-8,8,2), Vector(0,0,1), 0.5);
#else
  Instance *vinst = new Instance(viw, vtrans);
#endif
#endif
#endif

#ifdef DODAVE
  Material* hmat=new LambertianMaterial(Color(0.5,0.5,0.5));
  CutPlaneDpy* cpdpy=new CutPlaneDpy(Vector(-.45,.45,-.76), Point(-8,8,2));
  ColorMap *cmap = new ColorMap("/opt/SCIRun/data/Geometry/volumes/vol_cmap");
  Material *cutmat = new CutMaterial(hmat, cmap, cpdpy);
  CutGroup *cut = new CutGroup(cpdpy);
  CutVolumeDpy* cvdpy = new CutVolumeDpy(82.5, cmap);

#ifdef DOCUT
  HVolumeBrick16* davehead=new HVolumeBrick16(cutmat, cvdpy,
					      "/opt/SCIRun/data/Geometry/volumes/dave",
					      3, nworkers);
#else
  HVolumeBrick16* davehead=new HVolumeBrick16(hmat, cvdpy,
					      "/opt/SCIRun/data/Geometry/volumes/dave",
					      3, nworkers);
#endif
  InstanceWrapperObject *diw = new InstanceWrapperObject(davehead);
  Transform *dtrans = new Transform();

  dtrans->pre_translate(Vector(8, -8, -2)); 
  //dtrans->rotate(Vector(0,1,0), Vector(0,-1,0));
  dtrans->rotate(Vector(1,0,0), Vector(0,0,1));
  //don't do the next one, it's good for static images but it will be rotated again and
  //that will bloat the bounding box uneccessarily
  //dtrans->rotate(Vector(0,1,0), Vector(-.70,.70,0)); 

  //for testing scale in Instance
  //dtrans->pre_scale(Vector(4, 4, 4)); 

  dtrans->pre_translate(Vector(-8, 8, 2));

#ifdef DOINST
#ifdef DOSPIN
  SpinningInstance *dinst = new SpinningInstance(diw, dtrans, Point(-8,8,2), Vector(0,0,1), 0.1);
#else
  Instance *dinst = new Instance(diw, dtrans);
#endif
#endif

#ifdef DOINST
  cut->add(dinst);
#else
  cut->add(davehead);
#endif

#endif

  SelectableGroup *sg = new SelectableGroup(10);
#ifdef DOVFEM
#ifdef DOINST
  sg->add(vinst);
#else
  sg->add(vig);
#endif
#endif

#ifdef DODAVE

#ifdef DOCUT
  sg->add(cut);
#else

#ifdef DOINST
  sg->add(dinst);
#else
  sg->add(davehead);  
#endif

#endif

#endif
  g->add(sg);



  Color cdown(0.1, 0.1, 0.1);
  Color cup(0.1, 0.1, 0.1);

  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(0.3, 0.3, 0.3);
  Scene *scene = new Scene(g, cam, bgcolor, cdown, cup, groundplane, 0.5);

  scene->maxdepth = 8;
  scene->add_light(new Light(Point(-10, 8, 3.9), Color(.8,.8,.8), 0));
  scene->animate=true;

  scene->addObjectOfInterest( sg, true );
#ifdef DOVFEM
  scene->attach_display(vdpy);
  (new Thread(vdpy, "Volume Dpy"))->detach();
#endif

#ifdef DODAVE
  scene->attach_display(cpdpy);
  scene->attach_display(cvdpy);
  (new Thread(cpdpy, "CutPlane Dpy"))->detach();
  (new Thread(cvdpy, "Cut Volume Dpy"))->detach();
#endif
  return scene;
}








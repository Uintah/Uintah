#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/UVSphere.h>
#include <Packages/rtrt/Core/Satellite.h>
#include <Packages/rtrt/Core/Parallelogram.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/UVCylinder.h>
#include <Packages/rtrt/Core/UVCylinderArc.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/TileImageMaterial.h>
#include <Packages/rtrt/Core/MapBlendMaterial.h>
#include <Packages/rtrt/Core/MultiMaterial.h>
#include <Packages/rtrt/Core/PBNMaterial.h>
#include <Packages/rtrt/Core/CrowMarble.h>
#include <Packages/rtrt/Core/Checker.h>
#include <Packages/rtrt/Core/Halo.h>
#include <Packages/rtrt/Core/LightMaterial.h>
#include <iostream>
#include <math.h>
#include <string.h>

using namespace rtrt;

#define CYLTEXSCALEX  .06
#define CYLTEXSCALEY  .3
#define DOORWIDTH     .05
#define DOORHEIGHT    2.5
#define ROOMHEIGHT    10
#define HEIGHTRATIO   (DOORHEIGHT/ROOMHEIGHT)
#define ROOMCENTER    9,9
#define ROOMRADIUS    4
#define WALLTHICKNESS .1

#define INSCILAB 0

extern "C"
Scene* make_scene(int /*argc*/, char* /*argv*/[], int /*nworkers*/)
{
  Camera cam( Point(0,0,0), Point( 0,-1,0 ), Vector(0,0,1), 45.0 );

  Material *white = new LambertianMaterial(Color(1,1,1));
  Material *black = new LambertianMaterial(Color(0,0,0));
  LightMaterial *sun = new LightMaterial(Color(1,1,1));

  TileImageMaterial *sunmap = 
    new TileImageMaterial(
#if INSCILAB
                          "/home/sci/moulding/images/sun_map.ppm",
#else
                          "/home/moulding/images/sunmap.ppm",
#endif
                          1,
                          Color(1,1,1), 0,0, false);

  TileImageMaterial *earthmap = 
    new TileImageMaterial(
#if INSCILAB
                          "/home/sci/moulding/images/earth.ppm",
#else
                          "/home/moulding/images/earth.ppm",
#endif
                          1,
                          Color(1,1,1), 0,0, false);

  TileImageMaterial *earthmap_specular = 
    new TileImageMaterial(
#if INSCILAB
                          "/home/sci/moulding/images/earth.ppm",
#else
                          "/home/moulding/images/earth.ppm",
#endif
                          1,
                          Color(1,1,1), 100,0, false);

  TileImageMaterial *earthcloudmap = 
    new TileImageMaterial(
#if INSCILAB
                          "/home/sci/moulding/images/earth.ppm",
#else
                          "/home/moulding/images/earth.ppm",
#endif
                          1,
                          Color(1,1,1), 0,0, false);

  TileImageMaterial *marsmap = 
    new TileImageMaterial(
#if INSCILAB
                          "/home/sci/moulding/images/mars.ppm",
#else
                          "/home/moulding/images/mars.ppm",
#endif
                          1,
                          Color(1,1,1), 0,0, false);

  TileImageMaterial *neptunemap = 
    new TileImageMaterial(
#if INSCILAB
                          "/home/sci/moulding/images/neptune.ppm",
#else
                          "/home/moulding/images/neptune.ppm",
#endif
                          1,
                          Color(1,1,1), 0,0, false);

  TileImageMaterial *jupitermap = 
    new TileImageMaterial(
#if INSCILAB
                          "/home/sci/moulding/images/jupiter.ppm",
#else
                          "/home/moulding/images/jupiter.ppm",
#endif
                          1,
                          Color(1,1,1), 0,0, false);

  TileImageMaterial *saturnmap = 
    new TileImageMaterial(
#if INSCILAB
                          "/home/sci/moulding/images/saturn.ppm",
#else
                          "/home/moulding/images/saturn.ppm",
#endif
                          1,
                          Color(1,1,1), 0,0, false);

  TileImageMaterial *lunamap = 
    new TileImageMaterial(
#if INSCILAB
                          "/home/sci/moulding/images/luna.ppm",
#else
                          "/home/moulding/images/luna.ppm",
#endif
                          1,
                          Color(1,1,1), 0,0, false);

  TileImageMaterial *mercurymap = 
    new TileImageMaterial(
#if INSCILAB
                          "/home/sci/moulding/images/mercury.ppm",
#else
                          "/home/moulding/images/mercury.ppm",
#endif
                          1,
                          Color(1,1,1), 0,0, false);

  TileImageMaterial *venusmap = 
    new TileImageMaterial(
#if INSCILAB
                          "/home/sci/moulding/images/venus.ppm",
#else
                          "/home/moulding/images/venus.ppm",
#endif
                          1,
                          Color(1,1,1), 0,0, false);

  TileImageMaterial *iomap = 
    new TileImageMaterial(
#if INSCILAB
                          "/home/sci/moulding/images/io.ppm",
#else
                          "/home/moulding/images/io.ppm",
#endif
                          1,
                          Color(1,1,1), 0,0, false);

  TileImageMaterial *stadium = 
    new TileImageMaterial(
#if INSCILAB
                          "/home/sci/moulding/images/StadiumAtNoon.ppm",
#else
                          "/home/moulding/images/StadiumAtNoon.ppm",
#endif
                          1,
                          Color(1,1,1), 1000,0, false);
  
  TileImageMaterial *bookcoverimg = 
    new TileImageMaterial(
#if INSCILAB
                      "/opt/SCIRun/data/Geometry/textures/i3d97.ppm",
#else
                      "/home/moulding/i3d97.ppm",
#endif
                      1,
                      Color(1,1,1), 4000, 0, true);

  TileImageMaterial *matl0 = new TileImageMaterial(
#if INSCILAB                                         
                                           "/home/sci/moulding/holo.ppm",
#else
                                           "/home/moulding/images/holowall.ppm",
#endif
                                           4,Color(.5,.5,.5),40,0,0,true);

  CrowMarble *crow = new CrowMarble(10,Vector(0,1,0),Color(.1,.1,.8),
                                    Color(.9,.8,.9),Color(.6,.6,.8));

  Halo *halo = new Halo(sun,2);

  //MultiMaterial *matl1 = new MultiMaterial();
#if INSCILAB
  MapBlendMaterial *matl1 = new MapBlendMaterial("/home/sci/moulding/images/scuff.ppm",matl0,stadium);
#else
  MapBlendMaterial *matl1 = new MapBlendMaterial("/home/moulding/images/scuff.ppm",matl0,stadium);
#endif
  //PBNMaterial *matl1 = new PBNMaterial("/home/sci/moulding/images/pbnmaterial.ppm");

  //matl1->insert(stadium,255);
  //matl1->insert(crow,0);
#if INSCILAB
  MapBlendMaterial *earthspec = new MapBlendMaterial("/home/sci/moulding/images/earthspec4k.ppm",earthmap_specular,earthmap);
  MapBlendMaterial *earthblend = new MapBlendMaterial("/home/sci/moulding/images/earthclouds.ppm",white,earthspec);
#else
  MapBlendMaterial *earthspec = new MapBlendMaterial("/home/moulding/images/earthspec4k.ppm",earthmap_specular,earthmap);
  MapBlendMaterial *earthblend = new MapBlendMaterial("/home/moulding/images/earthclouds.ppm",white,earthspec);
#endif

  UVCylinderArc* uvcylarc0 = new UVCylinderArc(matl1,Point(ROOMCENTER,0),
                                               Point(ROOMCENTER,DOORHEIGHT),
                                               ROOMRADIUS);
  UVCylinderArc* uvcylarc1 = new UVCylinderArc(matl1,Point(ROOMCENTER,0),
                                               Point(ROOMCENTER,DOORHEIGHT),
                                               ROOMRADIUS);
  UVCylinderArc* uvcylarc2 = new UVCylinderArc(white,Point(ROOMCENTER,0),
                                               Point(ROOMCENTER,DOORHEIGHT),
                                               ROOMRADIUS+WALLTHICKNESS);
  UVCylinderArc* uvcylarc3 = new UVCylinderArc(white,Point(ROOMCENTER,0),
                                               Point(ROOMCENTER,DOORHEIGHT),
                                               ROOMRADIUS+WALLTHICKNESS);
  UVCylinder* uvcyl0 = new UVCylinder(matl1,Point(ROOMCENTER,DOORHEIGHT),
                                      Point(ROOMCENTER,ROOMHEIGHT),ROOMRADIUS);
  UVCylinder* uvcyl1 = new UVCylinder(white,Point(ROOMCENTER,DOORHEIGHT),
                                      Point(ROOMCENTER,ROOMHEIGHT),
                                      ROOMRADIUS+WALLTHICKNESS);

  Rect* floor = new Rect(matl1,Point(ROOMCENTER,0),Vector(ROOMRADIUS,0,0),
                          Vector(0,ROOMRADIUS,0));
  UVSphere *obj1 = new UVSphere(matl0,Point(5,5,0),1);
  Point p1(ROOMCENTER,.2);
  Parallelogram *bookcover = new Parallelogram(bookcoverimg,p1,
                                               Vector(.774,0,0),
                                               Vector(0,1,0));

  Satellite *sol = new Satellite("Sol",sunmap,Point(0,0,0),.7);
  sol->set_orb_radius(0);
  sol->set_orb_speed(0);
  UVSphere *corona = new UVSphere(halo,Point(0,0,0),1);
  Satellite *earth = new Satellite("Earth",earthblend,Point(0,0,8),2,
                                   Vector(-.3,0,1),sol);
  UVSphere *mars  = new UVSphere(marsmap,Point(0,4,8),1);
  UVSphere *jupiter = new UVSphere(jupitermap,Point(0,50,6),20);
  UVSphere *neptune = new UVSphere(neptunemap,Point(20,4,8),5);
  UVSphere *saturn = new  UVSphere(saturnmap,Point(20,15,6),6);
  UVSphere *luna = new UVSphere(lunamap,Point(4,0,8),.66);
  UVSphere *mercury = new UVSphere(mercurymap,Point(-4,0,7),.66);
  UVSphere *venus = new UVSphere(venusmap,Point(-8,-4,6),2);
  UVSphere *io = new UVSphere(iomap,Point(-10,-20,8),3);

  
  uvcylarc0->set_arc((DOORWIDTH)*M_PI,(.5-DOORWIDTH)*M_PI);
  uvcylarc1->set_arc((.5+DOORWIDTH)*M_PI,(2-DOORWIDTH)*M_PI);
  uvcylarc2->set_arc((DOORWIDTH)*M_PI,(.5-DOORWIDTH)*M_PI);
  uvcylarc3->set_arc((.5+DOORWIDTH)*M_PI,(2-DOORWIDTH)*M_PI);

  Group * group = new Group();
  //group->add( uvcylarc0 );
  //group->add( uvcylarc1 );
  //group->add( uvcylarc2 );
  //group->add( uvcylarc3 );
  //group->add( uvcyl0 );
  //group->add( uvcyl1 );
  //group->add( floor );
  //group->add( obj1 );
  //group->add( bookcover );
  group->add( sol );
  group->add( corona );
  group->add( earth );
  group->add( mars );
  //group->add( jupiter );
  //group->add( neptune );
  //group->add( saturn );
  //group->add( luna );
  //group->add( venus );
  //group->add( mercury );
  //group->add( io );

  double ambient_scale=1.0;
  Color bgcolor(0.1, 0.2, 0.45);
  Color cdown(0.82, 0.62, 0.62);
  Color cup(0.1, 0.3, 0.8);

  rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 1) );
  Scene* scene=new Scene(group, cam,
                         bgcolor, cdown, cup, groundplane,
                         ambient_scale, Arc_Ambient);
  scene->addObjectOfInterest(earth,true);
  scene->add_light( new Light(Point(0,0,0), Color(1,1,1), 0.0) );

#if INSCILAB
  scene->set_background_ptr( new EnvironmentMapBackground("/home/sci/moulding/images/tycho8.ppm"));
#else
  scene->set_background_ptr( new EnvironmentMapBackground("/home/moulding/images/tycho8.ppm"));
#endif

/*
  scene->select_shadow_mode("hard");
*/
  return scene;
}




#include <Packages/rtrt/Core/MapBlendMaterial.h>
#include <Packages/rtrt/Core/HaloMaterial.h>
#include <Packages/rtrt/Core/LightMaterial.h>
#include <Packages/rtrt/Core/TileImageMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>

#include <Packages/rtrt/Core/Satellite.h>
#include <Packages/rtrt/Core/PortalParallelogram.h>
#include <Packages/rtrt/Core/UVCylinder.h>
#include <Packages/rtrt/Core/UVCylinderArc.h>
#include <Packages/rtrt/Core/Rect.h>

#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Light2.h>

#include <iostream>
#include <math.h>

using namespace rtrt;
using namespace std;

#define DOORWIDTH              .05
#define DOORHEIGHT             2.5
#define ROOMHEIGHT             30
#define HEIGHTRATIO            (DOORHEIGHT/ROOMHEIGHT)
#define ROOMCENTER             46,46
#define ROOMRADIUS             50
#define WALLTHICKNESS          .1
#define INSCILAB               0
#define SYSTEM_SIZE_SCALE      1.438848E-5/*1.438848E-6*/
#define SYSTEM_DISTANCE_SCALE  6.76E-9/*3.382080E-9*/
#define SYSTEM_TIME_SCALE1     .4
#define SYSTEM_TIME_SCALE2     .02
#define FLIP_IMAGES            true
#if 0
#define IMAGEDIR      "/home/moulding/images/"
#else
#define IMAGEDIR      "/home/sci/moulding/images/"
#endif

typedef struct {

  double    radius_;
  double    orb_radius_;
  double    rot_speed_;
  double    orb_speed_;
  double    eccentricity_;
  double    tilt_;
  double    incline_;
  unsigned  parent_;
  unsigned  moons_;
  Satellite *self_;
  char      name_[20];

} satellite_data;

// the data in this table was gathered from 
// the "views of the solar system" web site
// www.scienceviews.com
satellite_data table[] = {
    
  { 695000, 0, 25.38, 0, 0, 0, 0, 
    0, 0, 0, "sunmap" },
  
  { 6378, 1.496E8, .99727, 365.26, .0167, 23.45, 0,
    0, 0, 0, "earth" },
  
  { 1737.4, 384400*50, 27.32166, 27.32166, .0549, 1.5424, 5.1454,
    1, 0, 0, "luna" },
  
  { 2439.7, 5.791E7, 58.65, 87.97, .2056, 0, 7.004, 
    0, 0, 0, "mercury" },
  
  { 6052, 1.082E8, -243.01, 224.7, .0068, 177.36, 3.394, 
    0, 0, 0, "venus" },
  
  { 3397, 2.2794E8, 1.026, 686.98, .0934, 25.19, 1.85,
    0, 2, 0, "mars" },
  
  { 71492, 7.7833E8, .4135, 4332.71, .0483, 3.13, 1.308,
    0, 26, 0, "jupiter" },
  
  { 60268, 1.4294E9, 2.394, 10759.5, .0560, 25.33, 2.488, 
    0, 34, 0, "saturn" },
  
  { 25559, 2.87099E9, -1.3687, 30685, .0461, 97.86, .774,
    0, 21, 0, "uranus" },
  
  { 24746, 4.5043E9, 1.52, 60190, .0097, 28.31, 1.774,
    0, 8, 0, "neptune" },
  
  { 1137*6, 5.91352E9, -6.3872, 90779, .2482, 122.52, 17.148,
    0, 1, 0, "pluto" },

  { 1815, 421600*400, 1.769, 1.769, .004, 0, .04,
    6, 0, 0, "io" },

  { 1569, 670900*400, 3.55, 3.55, .009, 0, .47,
    6, 0, 0, "europa" },

  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
};
    
extern "C"
Scene *make_scene(int /*argc*/, char* /*argv*/[], int /*nworkers*/) 
{
  Group *solar_system = new Group();

  Camera cam( Point(20,20,ROOMHEIGHT*.75), 
              Point(ROOMCENTER, ROOMHEIGHT/2.), 
              Vector(0,0,1), 60.0 );

  // create a scene
  double ambient_scale=0.1;
  Color bgcolor(0.1, 0.2, 0.45);
  Color cdown(0.82, 0.62, 0.62);
  Color cup(0.1, 0.3, 0.8);
  rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 1) );
  Scene* scene=new Scene(solar_system, cam,
                         bgcolor, cdown, cup, groundplane,
                         ambient_scale, Constant_Ambient);
  scene->select_shadow_mode(No_Shadows);
  EnvironmentMapBackground *starfield = 
    new EnvironmentMapBackground(IMAGEDIR"tycho8.ppm");
  scene->set_background_ptr( starfield );

  // materials
  Material *white = new LambertianMaterial(Color(1,1,1));

  string solppm(IMAGEDIR); solppm+=table[0].name_; solppm+=".ppm";
  TileImageMaterial *sol_m = 
    new TileImageMaterial(solppm, 1, Color(1,1,1), 0, 0, 
                          FLIP_IMAGES);

  string earthppm(IMAGEDIR); earthppm+=table[1].name_; earthppm+=".ppm";
  MapBlendMaterial *earth_spec = 
    new MapBlendMaterial(IMAGEDIR"earthspec4k.ppm", 
                         new TileImageMaterial(earthppm, 1, 
                                               Color(1,1,1), 100, 0, 
                                               FLIP_IMAGES),
                         new TileImageMaterial(earthppm, 1, 
                                               Color(1,1,1), 0, 0, 
                                               FLIP_IMAGES),
                         true);

  MapBlendMaterial *earth_m = 
    new MapBlendMaterial(IMAGEDIR"earthclouds.ppm", white, earth_spec);
              
  TileImageMaterial *matl0 = 
    new TileImageMaterial(IMAGEDIR"holowall.ppm",
                          4,Color(.5,.5,.5),0,0,0,true);
  matl0->SetScale(9,.9);

  TileImageMaterial *matl1 = 
    new TileImageMaterial(IMAGEDIR"holowall.ppm",
                          4,Color(.5,.5,.5),0,0,0,true);
  matl1->SetScale(3,3);

  // objects
  double radius;
  double orb_radius;

  UVCylinderArc* uvcylarc0 = new UVCylinderArc(matl0,Point(ROOMCENTER,0),
                                               Point(ROOMCENTER,DOORHEIGHT),
                                               ROOMRADIUS);
  UVCylinderArc* uvcylarc1 = new UVCylinderArc(matl0,Point(ROOMCENTER,0),
                                               Point(ROOMCENTER,DOORHEIGHT),
                                               ROOMRADIUS);
  UVCylinderArc* uvcylarc2 = new UVCylinderArc(white,Point(ROOMCENTER,0),
                                               Point(ROOMCENTER,DOORHEIGHT),
                                               ROOMRADIUS+WALLTHICKNESS);
  UVCylinderArc* uvcylarc3 = new UVCylinderArc(white,Point(ROOMCENTER,0),
                                               Point(ROOMCENTER,DOORHEIGHT),
                                               ROOMRADIUS+WALLTHICKNESS);
  UVCylinder* uvcyl0 = new UVCylinder(matl0,Point(ROOMCENTER,DOORHEIGHT),
                                      Point(ROOMCENTER,ROOMHEIGHT),ROOMRADIUS);
  UVCylinder* uvcyl1 = new UVCylinder(white,Point(ROOMCENTER,DOORHEIGHT),
                                      Point(ROOMCENTER,ROOMHEIGHT),
                                      ROOMRADIUS+WALLTHICKNESS);

  uvcylarc0->set_arc((DOORWIDTH)*M_PI,(.5-DOORWIDTH)*M_PI);
  uvcylarc1->set_arc((.5+DOORWIDTH)*M_PI,(2-DOORWIDTH)*M_PI);
  uvcylarc2->set_arc((DOORWIDTH)*M_PI,(.5-DOORWIDTH)*M_PI);
  uvcylarc3->set_arc((.5+DOORWIDTH)*M_PI,(2-DOORWIDTH)*M_PI);

  Rect* floor = new Rect(matl1,Point(ROOMCENTER,0),Vector(ROOMRADIUS,0,0),
                         Vector(0,ROOMRADIUS,0));
  Rect* ceiling = new Rect(matl1,Point(ROOMCENTER,ROOMHEIGHT),
                           Vector(ROOMRADIUS,0,0),
                           Vector(0,ROOMRADIUS,0));

  solar_system->add( uvcylarc0 );
  solar_system->add( uvcylarc1 );
  solar_system->add( uvcylarc2 );
  solar_system->add( uvcylarc3 );
  solar_system->add( uvcyl0 );
  solar_system->add( uvcyl1 );
  solar_system->add( floor );
  solar_system->add( ceiling );


  // test portals
#if 1
  PortalParallelogram *portal1 = 
      new PortalParallelogram(Point(-3,-4,16),
                              Vector(6,0,0),
                              Vector(0,8,0));

  PortalParallelogram *portal2 = 
      new PortalParallelogram(Point(4,-4,16),
                              Vector(0,0,-6),
                              Vector(0,8,0));

  PortalParallelogram::attach(portal1,portal2);
  solar_system->add(portal1);
  solar_system->add(portal2);
#endif

  // build the sun but don't add it to the scene 
  // (represented later as a light in the scene)
  Satellite *sol = new Satellite(table[0].name_, sol_m, Point(ROOMCENTER,
                                                              ROOMHEIGHT/2.), 
                                 .01);

  radius = table[1].radius_*SYSTEM_SIZE_SCALE;
  orb_radius = table[1].orb_radius_*SYSTEM_DISTANCE_SCALE;
  cerr << "radius = " << radius << endl;
  cerr << "orb radius = " << orb_radius << endl;
  Satellite *earth = new Satellite(table[1].name_, earth_m, 
                                   Point(orb_radius,0,0), radius);

  sol->set_orb_radius(0);
  sol->set_orb_speed(0);
  sol->set_rev_speed(1./table[0].rot_speed_*SYSTEM_TIME_SCALE1);
  table[0].self_ = sol;
  cerr << table[0].name_ << " = " << sol << endl;
  earth->set_orb_speed(1./table[1].orb_speed_*SYSTEM_TIME_SCALE2);
  earth->set_rev_speed(1./table[1].rot_speed_*SYSTEM_TIME_SCALE1);
  earth->set_up(Vector(cos(table[1].tilt_),0,sin(table[1].tilt_)));
  earth->set_parent(sol);
  table[1].self_ = earth;
  cerr << table[1].name_ << " = " << earth << endl;

  solar_system->add( earth );

  scene->addObjectOfInterest(earth,true);

  // build the other satellites
  for (unsigned loop=2; table[loop].radius_!=0; ++loop) {

    string satppm(IMAGEDIR); satppm+=table[loop].name_; satppm+=".ppm";
    Material *newmat = 
        new TileImageMaterial(satppm, 1, Color(1,1,1), 0, 0, FLIP_IMAGES);

    radius = table[loop].radius_*SYSTEM_SIZE_SCALE;
    orb_radius = table[loop].orb_radius_*SYSTEM_DISTANCE_SCALE;

    cerr << "radius = " << radius << endl;
    cerr << "orb radius = " << orb_radius << endl;
    
    Satellite *newsat = new Satellite(table[loop].name_,
                                      newmat, Point(orb_radius,0,0), 
                                      radius, 0);
    table[loop].self_ = newsat;
    
    newsat->set_parent(table[table[loop].parent_].self_);
    cerr << "satellite " << newsat->get_name() << " parent = " 
         << table[table[loop].parent_].self_ << endl;
    newsat->set_rev_speed(1./table[loop].rot_speed_*SYSTEM_TIME_SCALE1);
    newsat->set_orb_speed(1./table[loop].orb_speed_*SYSTEM_TIME_SCALE2);
    newsat->set_up(Vector(sin(table[loop].tilt_),0,cos(table[loop].tilt_)));

    solar_system->add( newsat );

    scene->addObjectOfInterest( newsat, true );
  }

  // add the light (the sun, as mentioned above)
  Light2 *light = new Light2(sol_m, Color(1,1,1), Point(ROOMCENTER,
                                                        ROOMHEIGHT/2.), .2);
  scene->add_light( light );
  scene->addObjectOfInterest(light->getSphere(),true);

  return scene;
}















#include <Packages/rtrt/Core/MapBlendMaterial.h>
#include <Packages/rtrt/Core/HaloMaterial.h>
#include <Packages/rtrt/Core/LightMaterial.h>
#include <Packages/rtrt/Core/TileImageMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>

#include <Packages/rtrt/Core/Satellite.h>
#include <Packages/rtrt/Core/Sphere.h>

#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Light.h>

#include <iostream>
#include <math.h>

using namespace rtrt;
using namespace std;

#define CYLTEXSCALEX           .06
#define CYLTEXSCALEY           .3
#define DOORWIDTH              .05
#define DOORHEIGHT             2.5
#define ROOMHEIGHT             10
#define HEIGHTRATIO            (DOORHEIGHT/ROOMHEIGHT)
#define ROOMCENTER             9,9
#define ROOMRADIUS             4
#define WALLTHICKNESS          .1
#define INSCILAB               0
#define SYSTEM_SIZE_SCALE      1.438848E-4/*1.438848E-6*/
#define SYSTEM_DISTANCE_SCALE  6.76E-8/*3.382080E-9*/
#define SYSTEM_TIME_SCALE1     1
#define SYSTEM_TIME_SCALE2     .5
#if 1
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
  char      tmap_filename_[20];

} satellite_data;

satellite_data table[] = {
    
    // solar satellites
  { 695000, 0, 25.38, 0, 0, 0, 0, 
    0, 0, 0, "sunmap.ppm" },
  
  { 6378, 1.496E8, .99727, 365.26, .0167, 23.45, 0,
    0, 0, 0, "earth.ppm" },
  
  { 2439.7, 5.791E7, 58.65, 87.97, .2056, 0, 7.004, 
    0, 0, 0, "mercury.ppm" },
  
  { 6052, 1.082E8, -243.01, 224.7, .0068, 177.36, 3.394, 
    0, 0, 0, "venus.ppm" },
  
  { 3397, 2.2794E8, 1.026, 686.98, .0934, 25.19, 1.85,
    0, 2, 0, "mars.ppm" },
  
  { 71492, 7.7833E8, .4135, 4332.71, .0483, 3.13, 1.308,
    0, 26, 0, "jupiter.ppm" },
  
  { 60268, 1.4294E9, 2.394, 10759.5, .0560, 25.33, 2.488, 
    0, 34, 0, "saturn.ppm" },
  
  { 25559, 2.87099E9, -1.3687, 30685, .0461, 97.86, .774,
    0, 21, 0, "uranus.ppm" },
  
  { 24746, 4.5043E9, 1.52, 60190, .0097, 28.31, 1.774,
    0, 8, 0, "neptune.ppm" },
  
  { 1137, 5.91352E9, -6.3872, 90779, .2482, 122.52, 17.148,
    0, 1, 0, "pluto.ppm" },
#if 0
  // planetary satellites
  { 1737.4, 384400, 27.32166, 27.32166, .0549, 1.5424, 5.1454,
    1, 0, 0, "luna.ppm" },
  
  { 1815, 421600, 1.769, 1.769, .004, 0, .04,
    5, 0, 0, "io.ppm" },

  { 1569, 670900, 3.55, 3.55, .009, 0, .47,
    5, 0, 0, "europa.ppm" },
#endif

  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
};
    
extern "C"
Scene *make_scene(int /*argc*/, char* /*argv*/[], int /*nworkers*/) 
{
  Group *solar_system = new Group();

  Camera cam( Point(20,20,20), Point( 0,0,0 ), Vector(0,0,1), 45.0 );

  // creat a scene
  double ambient_scale=0.5;
  Color bgcolor(0.1, 0.2, 0.45);
  Color cdown(0.82, 0.62, 0.62);
  Color cup(0.1, 0.3, 0.8);
  rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 1) );
  Scene* scene=new Scene(solar_system, cam,
                         bgcolor, cdown, cup, groundplane,
                         ambient_scale, Constant_Ambient);
  scene->add_light( new Light(Point(0,0,0), Color(1,1,1), 0.0) );
  scene->set_background_ptr( new EnvironmentMapBackground(IMAGEDIR"tycho8.ppm"));
  scene->select_shadow_mode(No_Shadows);

  Material *white = new LambertianMaterial(Color(1,1,1));

  TileImageMaterial *sol_m = 
    new TileImageMaterial(IMAGEDIR"sunmap.ppm", 1, Color(1,1,1), 0, 0, false);

  HaloMaterial *corona_m = new HaloMaterial(new LightMaterial(Color(1,1,1)),4);

  MapBlendMaterial *earth_spec = 
    new MapBlendMaterial(IMAGEDIR"earthspec4k.ppm", 
                         new TileImageMaterial(IMAGEDIR"earth.ppm", 1, 
                                               Color(1,1,1), 100, 0, false),
                         new TileImageMaterial(IMAGEDIR"earth.ppm", 1, 
                                               Color(1,1,1), 0, 0, false));

  MapBlendMaterial *earth_m = 
    new MapBlendMaterial(IMAGEDIR"earthclouds.ppm", white, earth_spec);
                                            
  double radius = 1;
  double orb_radius = 0;

  // build sol and the earth first
  cerr << "radius = " << radius << endl;
  cerr << "orb radius = " << orb_radius << endl;  
  Satellite *sol = new Satellite("Sol", sol_m, Point(0,0,0), radius*.7);
  Sphere *corona = new Sphere(corona_m, Point(0,0,0), radius);

  radius = table[1].radius_*SYSTEM_SIZE_SCALE;
  orb_radius = table[1].orb_radius_*SYSTEM_DISTANCE_SCALE;
  cerr << "radius = " << radius << endl;
  cerr << "orb radius = " << orb_radius << endl;
  Satellite *earth = new Satellite("Earth", earth_m, Point(orb_radius,0,0),
                                   radius);

  sol->set_orb_radius(0);
  sol->set_orb_speed(0);
  sol->set_rev_speed(1./table[0].rot_speed_*SYSTEM_TIME_SCALE1);
  table[0].self_ = sol;
  earth->set_orb_speed(1./table[1].orb_speed_*SYSTEM_TIME_SCALE2);
  earth->set_rev_speed(1./table[1].rot_speed_*SYSTEM_TIME_SCALE1);
  earth->set_up(Vector(sin(table[1].tilt_),0,cos(table[1].tilt_)));
  table[1].self_ = earth;

  //solar_system->add( sol );
  solar_system->add( corona );
  solar_system->add( earth );

  //scene->addObjectOfInterest(sol,true);
  scene->addObjectOfInterest(earth,true);

  // build the other satellites
  for (unsigned loop=2; loop<5/*table[loop].radius_!=0*/; ++loop) {

    Material *newmat = 
        new TileImageMaterial(string(IMAGEDIR)+table[loop].tmap_filename_,
                              1, Color(1,1,1), 0, 0, false);

    radius = table[loop].radius_*SYSTEM_SIZE_SCALE;
    orb_radius = table[loop].orb_radius_*SYSTEM_DISTANCE_SCALE;

    cerr << "radius = " << radius << endl;
    cerr << "orb radius = " << orb_radius << endl;
    
    Satellite *newsat = new Satellite(string("Planet ")+(char)loop,
                                      newmat, Point(orb_radius,0,0), 
                                      radius, 0);
    table[loop].self_ = newsat;
    
    newsat->set_parent(table[table[loop].parent_].self_);
    newsat->set_rev_speed(1./table[loop].rot_speed_*SYSTEM_TIME_SCALE1);
    newsat->set_orb_speed(1./table[loop].orb_speed_*SYSTEM_TIME_SCALE2);
    newsat->set_up(Vector(sin(table[loop].tilt_),0,cos(table[loop].tilt_)));

    solar_system->add( newsat );

    scene->addObjectOfInterest( newsat, true );
  }

  return scene;
}















#include <Packages/rtrt/Core/MapBlendMaterial.h>
#include <Packages/rtrt/Core/HaloMaterial.h>
#include <Packages/rtrt/Core/LightMaterial.h>
#include <Packages/rtrt/Core/TileImageMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/EMBMaterial.h>
#include <Packages/rtrt/Core/MultiMaterial.h>

#include <Packages/rtrt/Core/Satellite.h>
#include <Packages/rtrt/Core/PortalParallelogram.h>
#include <Packages/rtrt/Core/Disc.h>

#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Light2.h>

#include <iostream>
#include <math.h>

using namespace rtrt;
using namespace std;

#define DOORWIDTH              2
#define DOORHEIGHT             3
#define DOOROFFSET             5
#define ROOMHEIGHT             30
#define HEIGHTRATIO            (DOORHEIGHT/ROOMHEIGHT)
#define ROOMRADIUS             50
#define ROOMOFFSET             4
#define PORTALOFFSET           .001
#define ROOMCENTER             (ROOMRADIUS/2.+ROOMOFFSET),(ROOMRADIUS/2.+ROOMOFFSET)
#define WALLTHICKNESS          .1
#define INSCILAB               0
#define SYSTEM_SIZE_SCALE      1.438848E-5 /*1.438848E-6*/
#define SYSTEM_DISTANCE_SCALE  6.76E-9 /*3.382080E-9*/
#define SYSTEM_TIME_SCALE1     .4
#define SYSTEM_TIME_SCALE2     .01
#define FLIP_IMAGES            true
#define ANIMATE                true
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
    0, 0, 0, "sol" },
  
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

  //
  // create a scene
  //

  double ambient_scale=0.1;
  Color bgcolor(0.1, 0.2, 0.45);
  Color cdown(0.82, 0.62, 0.62);
  Color cup(0.1, 0.3, 0.8);
  rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 1) );
  Scene* scene=new Scene(solar_system, cam,
                         bgcolor, cdown, cup, groundplane,
                         ambient_scale, Constant_Ambient);
  scene->select_shadow_mode(No_Shadows);
  LinearBackground *background= 
    new LinearBackground(Color(.8,.8,.8),Color(.2,.3,.8),Vector(0,0,-1));
  scene->set_background_ptr( background );

  //
  // materials
  //

  Material *white = new LambertianMaterial(Color(1,1,1));

  EMBMaterial *starfield = new EMBMaterial(IMAGEDIR"tycho8.ppm");

  string solppm(IMAGEDIR); solppm+=table[0].name_; solppm+=".ppm";
  TileImageMaterial *sol_m = 
    new TileImageMaterial(solppm, 1, Color(1,1,1), 0, 0, 0,
                          FLIP_IMAGES);

  string earthppm(IMAGEDIR); earthppm+=table[1].name_; earthppm+=".ppm";
  MapBlendMaterial *earth_spec = 
    new MapBlendMaterial(IMAGEDIR"earthspec4k.ppm", 
                         new TileImageMaterial(earthppm, 1, 
                                               Color(1,1,1), 100, 0, 0,
                                               FLIP_IMAGES),
                         new TileImageMaterial(earthppm, 1, 
                                               Color(1,1,1), 0, 0, 0,
                                               FLIP_IMAGES),
                         FLIP_IMAGES);

  MapBlendMaterial *earth_m = 
    new MapBlendMaterial(IMAGEDIR"earthclouds.ppm", white, earth_spec,
                         FLIP_IMAGES);
              
  TileImageMaterial *matl0 = 
    new TileImageMaterial(IMAGEDIR"holowall.ppm",
                          4,Color(.5,.5,.5),0,0,0,FLIP_IMAGES);
  matl0->SetScale(6,6);

  TileImageMaterial *matl1 = 
    new TileImageMaterial(IMAGEDIR"holowall.ppm",
                          4,Color(.5,.5,.5),0,0,0,FLIP_IMAGES);
  matl1->SetScale(6,6*(ROOMHEIGHT/(double)(ROOMRADIUS*2)));

  MultiMaterial *holo0 = new MultiMaterial();
  holo0->insert(matl0,.8);
  holo0->insert(starfield,1);
  MultiMaterial *holo1 = new MultiMaterial();
  holo1->insert(matl1,.8);
  holo1->insert(starfield,1);

  //
  // objects
  //

  double radius;
  double orb_radius;


  // galaxy room

  Parallelogram *floor = new Parallelogram(holo0, 
                                           Point(ROOMOFFSET,ROOMOFFSET,0),
                                           Vector(ROOMRADIUS*2,0,0),
                                           Vector(0,ROOMRADIUS*2,0));

  Parallelogram *ceiling = new Parallelogram(holo0, 
                                             Point(ROOMOFFSET,
                                                   ROOMOFFSET,
                                                   ROOMHEIGHT),
                                             Vector(ROOMRADIUS*2,0,0),
                                             Vector(0,ROOMRADIUS*2,0));

  Parallelogram *wall0 = new Parallelogram(holo1, 
                                           Point(ROOMOFFSET,ROOMOFFSET,0),
                                           Vector(ROOMRADIUS*2,0,0),
                                           Vector(0,0,ROOMHEIGHT));
  Parallelogram *wall1 = new Parallelogram(holo1, \
                                           Point(ROOMRADIUS*2+ROOMOFFSET,
                                                 ROOMRADIUS*2+ROOMOFFSET,0),
                                           Vector(-ROOMRADIUS*2,0,0),
                                           Vector(0,0,ROOMHEIGHT));
  Parallelogram *wall2 = new Parallelogram(holo1, 
                                           Point(ROOMRADIUS*2+ROOMOFFSET,
                                                 ROOMOFFSET,0),
                                           Vector(0,ROOMRADIUS*2,0),
                                           Vector(0,0,ROOMHEIGHT));
  Parallelogram *wall3 = new Parallelogram(holo1, 
                                           Point(ROOMOFFSET,
                                                 ROOMRADIUS*2+ROOMOFFSET,0),
                                           Vector(0,-ROOMRADIUS*2,0),
                                           Vector(0,0,ROOMHEIGHT));

  solar_system->add( ceiling );
  solar_system->add( floor );
  solar_system->add( wall0 );
  solar_system->add( wall1 );
  solar_system->add( wall2 );
  solar_system->add( wall3 );

  // doors
  
  PortalParallelogram *door0a = 
    new PortalParallelogram(Point(ROOMOFFSET+DOOROFFSET,
                                  ROOMOFFSET-PORTALOFFSET,0),
                            Vector(DOORWIDTH,0,0),
                            Vector(0,0,DOORHEIGHT));

  PortalParallelogram *door0b = 
    new PortalParallelogram(Point(ROOMOFFSET+DOOROFFSET,
                                  ROOMOFFSET+PORTALOFFSET,0),
                            Vector(DOORWIDTH,0,0),
                            Vector(0,0,DOORHEIGHT));

  PortalParallelogram *door1a = 
    new PortalParallelogram(Point(ROOMOFFSET-PORTALOFFSET,
                                  ROOMOFFSET+DOOROFFSET+DOORWIDTH,0),
                            Vector(0,-DOORWIDTH,0),
                            Vector(0,0,DOORHEIGHT));

  PortalParallelogram *door1b = 
    new PortalParallelogram(Point(ROOMOFFSET+PORTALOFFSET,
                                  ROOMOFFSET+DOOROFFSET+DOORWIDTH,0),
                            Vector(0,-DOORWIDTH,0),
                            Vector(0,0,DOORHEIGHT));

  solar_system->add(door0a);
  solar_system->add(door0b);
  solar_system->add(door1a);
  solar_system->add(door1b);
  PortalParallelogram::attach(door0a,door0b);
  PortalParallelogram::attach(door1a,door1b);


  // build the sun but don't add it to the scene 
  // (represented later as a light in the scene)
  Satellite *sol = new Satellite(table[0].name_, white, 
                                 Point(ROOMCENTER, ROOMHEIGHT/2.), 
                                 .01, 0);
  sol->set_orb_radius(0);
  sol->set_orb_speed(0);
  sol->set_rev_speed(1./table[0].rot_speed_*SYSTEM_TIME_SCALE1);
  table[0].self_ = sol;
  cerr << sol->name_ << " = " << sol << endl;

  // build the earth (it has special texturing needs)
  radius = table[1].radius_*SYSTEM_SIZE_SCALE;
  orb_radius = table[1].orb_radius_*SYSTEM_DISTANCE_SCALE;
  cerr << "earth radius = " << radius << endl;
  cerr << "earth orb radius = " << orb_radius << endl;
  Satellite *earth = new Satellite(table[1].name_, earth_m, 
                                   Point(0,0,0), radius, orb_radius,
                                   Vector(cos(table[1].tilt_),0,
                                          sin(table[1].tilt_)), sol);
  earth->set_orb_speed(1./table[1].orb_speed_*SYSTEM_TIME_SCALE2);
  earth->set_rev_speed(1./table[1].rot_speed_*SYSTEM_TIME_SCALE1);
  table[1].self_ = earth;
  cerr << earth->name_ << " = " << earth << endl;
  solar_system->add( earth );
  scene->addObjectOfInterest(earth,ANIMATE);

  // build the other satellites
  for (unsigned loop=2; table[loop].radius_!=0; ++loop) {

    string satppm(IMAGEDIR); satppm+=table[loop].name_; satppm+=".ppm";
    Material *newmat = 
        new TileImageMaterial(satppm, 1, Color(1,1,1), 0, 0, 0, FLIP_IMAGES);

    radius = table[loop].radius_*SYSTEM_SIZE_SCALE;
    orb_radius = table[loop].orb_radius_*SYSTEM_DISTANCE_SCALE;

    Satellite *newsat = new Satellite(table[loop].name_,
                                      newmat, Point(0,0,0), 
                                      radius, orb_radius, 
                                      Vector(sin(table[loop].tilt_),0,
                                             cos(table[loop].tilt_)),
                                      table[table[loop].parent_].self_);
    table[loop].self_ = newsat;
    cerr << newsat->get_name() << " radius = " << radius << endl;
    cerr << newsat->get_name() << " orb radius = " << orb_radius << endl;
    cerr << "satellite " << newsat->get_name() << " parent = " 
         << newsat->get_parent() << endl;
    newsat->set_rev_speed(1./table[loop].rot_speed_*SYSTEM_TIME_SCALE1);
    newsat->set_orb_speed(1./table[loop].orb_speed_*SYSTEM_TIME_SCALE2);
    solar_system->add( newsat );
    scene->addObjectOfInterest( newsat, ANIMATE );
  }

  // add the light (the sun, as mentioned above)
  Light2 *light = new Light2(sol_m, Color(1,.9,.8), 
                             Point(ROOMCENTER, ROOMHEIGHT/2.), .2,4);
  scene->add_light( light );
  scene->addObjectOfInterest(light->getSphere(),ANIMATE);

  return scene;
}














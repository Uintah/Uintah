// need this before including anything else
// set _USING_GRID2_ to: 1 for Grid2, 0 for regular Grid
#define _USING_GRID_  0
#define _USING_GRID2_ 0

#include <Packages/rtrt/Core/MapBlendMaterial.h>
#include <Packages/rtrt/Core/HaloMaterial.h>
#include <Packages/rtrt/Core/LightMaterial.h>
#include <Packages/rtrt/Core/TileImageMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/EMBMaterial.h>
#include <Packages/rtrt/Core/MultiMaterial.h>
#include <Packages/rtrt/Core/InvisibleMaterial.h>

#include <Packages/rtrt/Core/Satellite.h>
#include <Packages/rtrt/Core/PortalParallelogram.h>
#include <Packages/rtrt/Core/Parallelogram2.h>
#include <Packages/rtrt/Core/RingSatellite.h>
#include <Packages/rtrt/Core/TexturedTri2.h>

#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Light2.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Grid2.h>
#include <Packages/rtrt/Core/BV1.h>

#include <iostream>
#include <math.h>

using namespace rtrt;
using namespace std;

#define RENDERWALLS            1
#define RENDERPLANETS          1
#define DOORWIDTH              1.5
#define DOORHEIGHT             2.3
#define DOOROFFSET             5.25
#define ROOMSCALE              .5
#define ROOMFLOOR              0
#define ROOMHEIGHT             ((30*ROOMSCALE)+ROOMFLOOR)
#define ROOMRADIUS             (50*ROOMSCALE)
#define ROOMOFFSETX            4
#define ROOMOFFSETY            4
#define PORTALOFFSET           .002
#define ROOMCENTER             (ROOMRADIUS+ROOMOFFSETX),\
                               (ROOMRADIUS+ROOMOFFSETY)
#define WALLTHICKNESS          .001
#define SYSTEM_SIZE_SCALE      (1.438848E-5*ROOMSCALE) /*1.438848E-6*/
#define SYSTEM_DISTANCE_SCALE  (6.76E-9*ROOMSCALE) /*3.382080E-9*/
#define SYSTEM_TIME_SCALE1     .4
#define SYSTEM_TIME_SCALE2     .001
#define FLIP_IMAGES            true
#define ANIMATE                true
#if 0
#define IMAGEDIR      "/home/moulding/images/"
#else
#define IMAGEDIR      "/opt/SCIRun/data/Geometry/textures/holo-room/"
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

#define DEG2RAD(x) (x*M_PI/180.)

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
  
  { 6052, 1.082E8, 243.01, 224.7, .0068, 177.36, 3.394, 
    0, 0, 0, "venus" },
  
  { 3397, 2.2794E8, 1.026, 686.98, .0934, 25.19, 1.85,
    0, 2, 0, "mars" },
  
  { 71492, 7.7833E8, .4135, 4332.71, .0483, 3.13, 1.308,
    0, 26, 0, "jupiter" },
  
  { 60268, 1.4294E9, 2.394, 10759.5, .0560, 25.33, 2.488, 
    0, 34, 0, "saturn" },
  
  { 25559, 2.87099E9, 1.3687, 30685, .0461, 97.86, .774,
    0, 21, 0, "uranus" },
  
  { 24746, 4.5043E9, 1.52, 60190, .0097, 28.31, 1.774,
    0, 8, 0, "neptune" },
  
  { 1137*6, 5.91352E9, 6.3872, 90779, .2482, 122.52, 17.148,
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
  Group *galaxy_room = new Group();
  Group *solar_system = new Group();
#if _USING_GRID_
#if _USING_GRID2_
  Grid2 *solar_grid = new Grid2(solar_system,8);
#else
  Grid *solar_grid = new Grid(solar_system,8);
#endif
  galaxy_room->add( solar_grid );
#else
  BV1 *solar_grid = new BV1(solar_system);
  galaxy_room->add( solar_grid );
#endif
  BV1 *room_grid = new BV1(galaxy_room);

  Camera cam( Point(10,0,1.8), 
              Point(9,ROOMRADIUS,1.8), 
              Vector(0,0,1), 60.0 );

  //
  // create a scene
  //

  double ambient_scale=0.1;
  Color bgcolor(0.1, 0.2, 0.45);
  Color cdown(0.82, 0.62, 0.62);
  Color cup(0.1, 0.3, 0.8);
  rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 1) );
  Scene* scene=new Scene(room_grid, cam,
                         bgcolor, cdown, cup, groundplane,
                         ambient_scale, Constant_Ambient);
  scene->select_shadow_mode(No_Shadows);
  LinearBackground *background = 
    new LinearBackground(Color(.8,.5,.3),Color(.05,.1,.4),Vector(0,0,-1));
  scene->set_background_ptr( background );

  //
  // materials
  //

  Material *white = new LambertianMaterial(Color(1,1,1));

  EMBMaterial *starfield = new EMBMaterial(IMAGEDIR"tycho8.ppm");

  MapBlendMaterial *rings_m = 
    new MapBlendMaterial(IMAGEDIR"rings.ppm",
                         new InvisibleMaterial(),
                         white);

  string solppm(IMAGEDIR); solppm+=table[0].name_; solppm+=".ppm";
  TileImageMaterial *sol_m = 
    new TileImageMaterial(solppm, 1, Color(1,1,1), 0, 0, 0,
                          FLIP_IMAGES);

  string earthppm(IMAGEDIR); earthppm+=table[1].name_; earthppm+=".ppm";

  TileImageMaterial *espec0 = 
    new TileImageMaterial(earthppm, 1, 
                          Color(1,1,1), 20, 0, 0,
                          FLIP_IMAGES);
  TileImageMaterial *espec1 = 
    new TileImageMaterial(earthppm, 1, 
                          Color(1,1,1), 0, 0, 0,
                          FLIP_IMAGES);

  MapBlendMaterial *earth_spec = 
    new MapBlendMaterial(IMAGEDIR"earthspec4k.ppm", 
                         espec0, espec1,
                         FLIP_IMAGES);

  MapBlendMaterial *earth_m = 
    new MapBlendMaterial(IMAGEDIR"earthclouds.ppm", white, earth_spec,
                         FLIP_IMAGES);
              
  TileImageMaterial *matl0 = 
    new TileImageMaterial(IMAGEDIR"holowall3.ppm",
                          4,Color(.5,.5,.5),0,0,0,FLIP_IMAGES);
  matl0->SetScale(3,3);

  TileImageMaterial *matl1 = 
    new TileImageMaterial(IMAGEDIR"holowall3.ppm",
                          4,Color(.5,.5,.5),0,0,0,FLIP_IMAGES);
  matl1->SetScale(3,3*(ROOMHEIGHT/(double)(ROOMRADIUS*2)));

  MultiMaterial *holo0 = new MultiMaterial();
  holo0->insert(matl0,1);
  holo0->insert(starfield,1);
  MultiMaterial *holo1 = new MultiMaterial();
  holo1->insert(matl1,1);
  holo1->insert(starfield,1);

  // add the light (the sun, as mentioned above)
  Light2 *light = new Light2(sol_m, Color(1,.9,.8), 
                             Point(ROOMCENTER, ROOMHEIGHT/2.), .2*ROOMSCALE,4);
  //Light *light = new Light(Color(1,.9,.8), 
  //                         Point(ROOMCENTER, ROOMHEIGHT/2.), .2*ROOMSCALE);
  scene->add_perm_per_matl_light( light );
  scene->addObjectOfInterest(light->getSphere(),ANIMATE, false);

  //
  // objects
  //

  double radius;
  double orb_radius;

  // galaxy room

#if RENDERWALLS
  Parallelogram2 *floor = new Parallelogram2(holo0, 
                                             Point(ROOMOFFSETX,
                                                   ROOMOFFSETY,
                                                   ROOMFLOOR),
                                             Vector(ROOMRADIUS*2,0,0),
                                             Vector(0,ROOMRADIUS*2,0));

  Parallelogram *ceiling = new Parallelogram(holo0, 
                                             Point(ROOMOFFSETX,
                                                   ROOMOFFSETY,
                                                   ROOMHEIGHT),
                                             Vector(ROOMRADIUS*2,0,0),
                                             Vector(0,ROOMRADIUS*2,0));

  // south
  Parallelogram2 *wall0 = new Parallelogram2(holo1, 
                                             Point(ROOMOFFSETX,ROOMOFFSETY,
                                                   ROOMFLOOR),
                                            Vector(ROOMRADIUS*2,0,0),
                                             Vector(0,0,ROOMHEIGHT));
  // north
  Parallelogram *wall1 = new Parallelogram(holo1, \
                                           Point(ROOMRADIUS*2+ROOMOFFSETX,
                                                 ROOMRADIUS*2+ROOMOFFSETY,
                                                 ROOMFLOOR),
                                           Vector(-ROOMRADIUS*2,0,0),
                                           Vector(0,0,ROOMHEIGHT));
  // east
  Parallelogram *wall2 = new Parallelogram(holo1, 
                                           Point(ROOMRADIUS*2+ROOMOFFSETX,
                                                 ROOMOFFSETY,ROOMFLOOR),
                                           Vector(0,ROOMRADIUS*2,0),
                                           Vector(0,0,ROOMHEIGHT));
  // west
  Parallelogram *wall3 = new Parallelogram(holo1, 
                                           Point(ROOMOFFSETX,
                                                 ROOMRADIUS*2+ROOMOFFSETY,
                                                 ROOMFLOOR),
                                           Vector(0,-ROOMRADIUS*2,0),
                                           Vector(0,0,ROOMHEIGHT));

  // south
  Parallelogram *outwall0 = new Parallelogram(white, 
                                              Point(ROOMOFFSETX-
                                                    WALLTHICKNESS,
                                                    ROOMOFFSETY-
                                                    WALLTHICKNESS,
                                                    ROOMFLOOR),
                                              Vector((ROOMRADIUS*2)+
                                                     (WALLTHICKNESS*2),0,0),
                                              Vector(0,0,ROOMHEIGHT+
                                                     (WALLTHICKNESS*2)));
  // north
  Parallelogram *outwall1 = new Parallelogram(white,
                                              Point(ROOMRADIUS*2+ROOMOFFSETX+
                                                    WALLTHICKNESS,
                                                    ROOMRADIUS*2+ROOMOFFSETY+
                                                    WALLTHICKNESS,
                                                    ROOMFLOOR),
                                              Vector(-ROOMRADIUS*2-
                                                     WALLTHICKNESS*2,0,0),
                                              Vector(0,0,ROOMHEIGHT+
                                                     WALLTHICKNESS*2));
  // east
  Parallelogram *outwall2 = new Parallelogram(white, 
                                              Point(ROOMRADIUS*2+ROOMOFFSETX+
                                                    WALLTHICKNESS,
                                                    ROOMOFFSETY-WALLTHICKNESS,
                                                    ROOMFLOOR),
                                              Vector(0,ROOMRADIUS*2+
                                                     WALLTHICKNESS*2,0),
                                              Vector(0,0,ROOMHEIGHT+
                                                     WALLTHICKNESS*2));
  // west
  Parallelogram *outwall3 = new Parallelogram(white, 
                                              Point(ROOMOFFSETX-
                                                    WALLTHICKNESS,
                                                    ROOMRADIUS*2+ROOMOFFSETY+
                                                    WALLTHICKNESS,
                                                    ROOMFLOOR),
                                              Vector(0,-ROOMRADIUS*2-
                                                     (WALLTHICKNESS*2),0),
                                              Vector(0,0,ROOMHEIGHT+
                                                     (WALLTHICKNESS*2)));
  
#if _USING_GRID2_
  // extend the bbox for the grid2 to the size of the room
  solar_system->add( floor );
  solar_system->add( ceiling );
#else
  galaxy_room->add( floor );
  galaxy_room->add( ceiling );
#endif
  galaxy_room->add( wall0 );
  galaxy_room->add( wall1 );
  galaxy_room->add( wall2 );
  galaxy_room->add( wall3 );
  galaxy_room->add( outwall0 );
  galaxy_room->add( outwall1 );
  galaxy_room->add( outwall2 );
  galaxy_room->add( outwall3 );

  // to animate the holo room on/off
  scene->addObjectOfInterest( floor, true, false );
  scene->addObjectOfInterest( wall0, true, false );

  // doors
  
  PortalParallelogram *door0a = 
    new PortalParallelogram(Point(ROOMOFFSETX+DOOROFFSET,
                                  ROOMOFFSETY-PORTALOFFSET,ROOMFLOOR),
                            Vector(DOORWIDTH,0,0),
                            Vector(0,0,DOORHEIGHT));

  PortalParallelogram *door0b = 
    new PortalParallelogram(Point(ROOMOFFSETX+DOOROFFSET,
                                  ROOMOFFSETY+PORTALOFFSET,ROOMFLOOR),
                            Vector(DOORWIDTH,0,0),
                            Vector(0,0,DOORHEIGHT));

  PortalParallelogram *door1a = 
    new PortalParallelogram(Point(ROOMOFFSETX-PORTALOFFSET,
                                  ROOMOFFSETY+DOOROFFSET+DOORWIDTH,ROOMFLOOR),
                            Vector(0,-DOORWIDTH,0),
                            Vector(0,0,DOORHEIGHT));

  PortalParallelogram *door1b = 
    new PortalParallelogram(Point(ROOMOFFSETX+PORTALOFFSET,
                                  ROOMOFFSETY+DOOROFFSET+DOORWIDTH,ROOMFLOOR),
                            Vector(0,-DOORWIDTH,0),
                            Vector(0,0,DOORHEIGHT));

  galaxy_room->add( door0a );
  galaxy_room->add( door0b );
  galaxy_room->add( door1a );
  galaxy_room->add( door1b );
  PortalParallelogram::attach(door0a,door0b);
  PortalParallelogram::attach(door1a,door1b);
#endif

#if RENDERPLANETS
  // build the sun but don't add it to the scene 
  // (represented later by the light in the scene)
  Satellite *sol = new Satellite(table[0].name_, white, 
                                 Point(ROOMCENTER, ROOMHEIGHT/2.), 
                                 .01, 0);
  sol->set_orb_radius(0);
  sol->set_orb_speed(0);
  sol->set_rev_speed(1./table[0].rot_speed_*SYSTEM_TIME_SCALE1);
  table[0].self_ = sol;
  cerr << sol->get_name() << " = " << sol << endl;

  // build the earth first (it has special texturing needs)
  radius = table[1].radius_*SYSTEM_SIZE_SCALE;
  orb_radius = table[1].orb_radius_*SYSTEM_DISTANCE_SCALE;
  cerr << "earth radius = " << radius << endl;
  cerr << "earth orb radius = " << orb_radius << endl;
  Vector up(sin(DEG2RAD(table[1].tilt_)), 0, cos(DEG2RAD(table[1].tilt_)));
  up.normalize();
  Satellite *earth = new Satellite(table[1].name_, earth_m, 
                                   Point(0,0,0), radius, orb_radius,
                                   up, sol);
  earth->set_orb_speed(1./table[1].orb_speed_*SYSTEM_TIME_SCALE2);
  earth->set_rev_speed(1./table[1].rot_speed_*SYSTEM_TIME_SCALE1);
  table[1].self_ = earth;
  cerr << earth->get_name() << " = " << earth << endl;
  solar_system->add( earth );
  
  // these two lines needed for animation
#if _USING_GRID_
#if _USING_GRID2_
  earth->set_anim_grid(solar_grid);
#endif
  scene->addObjectOfInterest( earth->get_name(), earth, ANIMATE, true);
#else
  cerr << "adding animate object " << earth->get_name() << endl;
  scene->addObjectOfInterest( earth, ANIMATE, false );
#endif

  // build the other satellites
  for (unsigned loop=2; table[loop].radius_!=0; ++loop) {

    string satppm(IMAGEDIR); satppm+=table[loop].name_; satppm+=".ppm";
    Material *newmat = 
        new TileImageMaterial(satppm, 1, Color(1,1,1), 0, 0, 0, FLIP_IMAGES);

    newmat->my_lights.add(light);
    radius = table[loop].radius_*SYSTEM_SIZE_SCALE;
    orb_radius = table[loop].orb_radius_*SYSTEM_DISTANCE_SCALE;

    up = Vector(sin(DEG2RAD(table[1].tilt_)), 0, cos(DEG2RAD(table[1].tilt_)));
    up.normalize();
    Satellite *newsat = new Satellite(table[loop].name_,
                                      newmat, Point(0,0,0), 
                                      radius, orb_radius, 
                                      up,
                                      table[table[loop].parent_].self_);
    table[loop].self_ = newsat;
    cerr << newsat->get_name() << " radius = " << radius << endl;
    cerr << newsat->get_name() << " orb radius = " << orb_radius << endl;
    cerr << "satellite " << newsat->get_name() << " parent = " 
         << newsat->get_parent() << endl;
    newsat->set_rev_speed(1./table[loop].rot_speed_*SYSTEM_TIME_SCALE1);
    newsat->set_orb_speed(1./table[loop].orb_speed_*SYSTEM_TIME_SCALE2);
    solar_system->add( newsat );
#if _USING_GRID_
#if _USING_GRID2_
    newsat->set_anim_grid(solar_grid);
#endif
    scene->addObjectOfInterest( newsat->get_name(), newsat, ANIMATE, true );
#else
    cerr << "adding animate object " << earth->get_name() << endl;
    scene->addObjectOfInterest( newsat, ANIMATE, false );
#endif

    if (newsat->get_name() == "saturn") {
      cerr << "adding rings!!!! " << radius << endl;
      up = Vector(-sin(DEG2RAD(table[loop].tilt_))+.25, 0, 
                  cos(DEG2RAD(table[loop].tilt_)));
      RingSatellite *rings = 
        new RingSatellite("rings",rings_m,
                          newsat->get_center(),
                          up,
                          74400*SYSTEM_SIZE_SCALE,
                          65754*SYSTEM_SIZE_SCALE,
                          newsat);

      solar_system->add( rings );
#if _USING_GRID_
#if _USING_GRID2_
      rings->set_anim_grid(solar_grid);
#endif
      scene->addObjectOfInterest( "rings", rings, ANIMATE, true );
#else
      scene->addObjectOfInterest( rings, ANIMATE, false );
#endif
    }
  }
#endif

  // set material lighting

  white->my_lights.add(light);
  //starfield->my_lights.add(light);
  rings_m->my_lights.add(light);
  sol_m->my_lights.add(light);
  espec0->my_lights.add(light);
  espec1->my_lights.add(light);
  earth_spec->my_lights.add(light);
  earth_m->my_lights.add(light);
  matl0->my_lights.add(light);
  matl1->my_lights.add(light);
  holo0->my_lights.add(light);
  holo1->my_lights.add(light);

  return scene;
}














#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/TimeVaryingInstance.h>
#include <Packages/rtrt/Core/InstanceWrapperObject.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/AirBubble.h>
#include <Packages/rtrt/Core/Grid2.h>
#include <Packages/rtrt/Core/Box.h>
#include <Packages/rtrt/Core/Ring.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Heightfield.h>
#include <Packages/rtrt/Core/BrickArray2.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/VideoMap.h>
#include <Packages/rtrt/Core/CrowMarble.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/TimeVaryingCheapCaustics.h>
#include <Packages/rtrt/Core/SeaLambertian.h>
#include <Packages/rtrt/Core/Checker.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/Cylinder.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/ASEReader.h>
#include <Packages/rtrt/Core/ObjReader.h>
#include <Core/Math/MinMax.h>

using namespace rtrt;

extern "C"
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
  for(int i=1;i<argc;i++) {
    cerr << "Unknown option: " << argv[i] << '\n';
    cerr << "Valid options for scene: " << argv[0] << '\n';
    return 0;
  }

  Point Eye(0, 5, 2.0);
  Point Lookat(0, 0, 2.0);
  Vector Up(0,0,1);
  //Point Eye (0, 0, 10);
  //Point Lookat(0, 0, 0);
  //Vector Up (0, 1, 0);
  double fov=60;

  Camera cam(Eye,Lookat,Up,fov);

  Group *all_tubes = new Group;
  Group *south_tube = new Group;
  Group *north_tube = new Group;
  Group *west_tube = new Group;
  Group *east_tube = new Group;
  Group *ruins = new Group;
  Group *col1 = new Group;
  Group *col2 = new Group;
  Group *col3 = new Group;
  Group *col4 = new Group;
  Group *col5 = new Group;
  Group *col6 = new Group;
  Group *col7 = new Group;
  Group *patch1 = new Group;
  Group *patch2 = new Group;
  Group *patch3 = new Group;
  Group *patch4 = new Group;
  Group *gazebo = new Group;
  Group *erect = new Group;
  Group *parth = new Group;
  Group *panth = new Group;
  Group *temple = new Group;
  Group *temple2 = new Group;
  Group *bubbles = new Group;
  Group *rock1 = new Group;
  Group *rock2 = new Group;
  Group *iceberg = new Group;
  Group *iceberg2 = new Group;
  Group *craters = new Group;
  Group *rock_tower = new Group;

  TimeVaryingCheapCaustics* tvcc= new TimeVaryingCheapCaustics("caustics/caust%d.pgm", 32,
	                                                        Point(0,0,6), Vector(1,0,0), Vector(0,1,0),
							        Color(0.5,0.5,0.5), 0.1, .3);// last should be .6
  
  Material* glass_to_air = new DielectricMaterial(1.0, 1.5, 0.04, 400.0, Color(.80, .93 , .87), Color(1,1,1), false);
  Material* water_to_glass = new DielectricMaterial(1.0, 1.2, 0.04, 400.0, Color(.80, .84 , .93), Color(1,1,1), false);

  Material* air_bubble = new DielectricMaterial(1.0, 1.1, 0.004, 400.0, Color(1, 1, 1), Color(1.01,1.01,1.01), false);

  Material* white = new LambertianMaterial(Color(0.8,0.8,0.8));
  Material* red = new LambertianMaterial(Color(1,0,0));
  Material* tan = new SeaLambertianMaterial(Color(0.6,0.6,0.2), tvcc);
  //Material* tan = new LambertianMaterial(Color(0.6,0.6,0.2));
  Material* seawhite = new SeaLambertianMaterial(Color(0.3,0.3,0.3), tvcc);
  Material* black = new PhongMaterial(Color(0.05,0.05,0.05), 1.0);
  Material* metal = new MetalMaterial(Color(0.1,0.1,0.1));
  Material* marble1 = new CrowMarble(4.5, Vector(.3, .3, 0), Color(.9,.9,.9), Color(.8, .8, .8), Color(.7, .7, .7)); 
  Material* marble2 = new CrowMarble(4.5, Vector(-.3, -.3, 0), Color(.05, .05, .05), Color(.075, .075, .075), Color(.1, .1, .1)); 
  Material* ruin_marble1 = new CrowMarble(7.5, Vector(.3, .3, 0), Color(.9,.9,.9), Color(.7, .7, .6), Color(.5, .5, .5)); 
  Material* ruin_marble2 = new CrowMarble(7.5, Vector(.3, 0, .2), Color(.9,.9,.9), Color(.7, .7, .6), Color(.5, .5, .5)); 
  Material* checker = new Checker(marble1, marble2, Vector(1.2, 0, 0), Vector(0, 1.2, 0));  
  Material* ruin_checker1 = new Checker(ruin_marble2, ruin_marble1, Vector(2, 0, 0), Vector(0, 2, 0));  
  Material* ruin_checker2 = new Checker(ruin_marble1, ruin_marble2, Vector(2, 0, 0), Vector(0, 2, 0));  
  
  
  /**********************************************************************/
  // south tube
  
  // glass tube
  //Object *south_tube_inner = new Cylinder(glass_to_air, Point(-4, -6, 1), Point(4, -6, 1), 2);
  Object *south_tube_outer = new Cylinder(water_to_glass, Point(-4, -6, 1), Point(7.5, -6, 1), 2.05);
  //south_tube->add(south_tube_inner);
  south_tube->add(south_tube_outer);
  // floor
  south_tube->add(new Rect(checker, Point(1.75, -6, 0), Vector(5.75, 0, 0), Vector(0, 1.5, 0)));
  // north curb
  south_tube->add(new Rect(white, Point(1.75, -4.25, .25), Vector(5.75, 0, 0), Vector(0, .15, 0)));
  south_tube->add(new Rect(white, Point(1.75, -4.475, .1), Vector(5.75, 0, 0), Vector(0, -.025, -.1)));
  south_tube->add(new Cylinder(white, Point(-4, -4.4, .2), Point(7.5, -4.4, .2), .05));
  // south curb
  south_tube->add(new Rect(white, Point(1.75, -7.75, .25), Vector(5.75, 0, 0), Vector(0, .15, 0)));
  south_tube->add(new Rect(white, Point(1.75, -7.525, .1), Vector(5.75, 0, 0), Vector(0, .025, -.1)));
  south_tube->add(new Cylinder(white, Point(-4, -7.6, .2), Point(7.5, -7.6, .2), .05));
  // seals
  // west seal
  south_tube->add(new Ring(black, Point(-4, -6, 1), Vector(1, 0, 0), 1.9, .3));
  south_tube->add(new Ring(black, Point(-3.9, -6, 1), Vector(1, 0, 0), 1.9, .3));
  south_tube->add(new Cylinder(black, Point(-3.9, -6, 1), Point(-4, -6, 1), 1.9));
  south_tube->add(new Cylinder(black, Point(-3.9, -6, 1), Point(-4, -6, 1), 2.2));
  // east seal
  south_tube->add(new Ring(black, Point(7.5, -6, 1), Vector(1, 0, 0), 1.9, .3));
  south_tube->add(new Ring(black, Point(7.4, -6, 1), Vector(1, 0, 0), 1.9, .3));
  south_tube->add(new Cylinder(black, Point(7.4, -6, 1), Point(7.5, -6, 1), 1.9));
  south_tube->add(new Cylinder(black, Point(7.4, -6, 1), Point(7.5, -6, 1), 2.2));  
  
  all_tubes->add(south_tube);


  /**********************************************************************/
  // north tube

  // glass tube
  //Object *north_tube_inner = new Cylinder(glass_to_air, Point(-4, 10, 1), Point(4, 10, 1), 2);
  Object *north_tube_outer = new Cylinder(water_to_glass, Point(-4, 10, 1), Point(4, 10, 1), 2.05);
  //north_tube->add(north_tube_inner);
  north_tube->add(north_tube_outer);
  // floor
  north_tube->add(new Rect(checker, Point(0, 10, 0), Vector(4, 0, 0), Vector(0, 1.5 ,0)));
  // south curb
  north_tube->add(new Rect(white, Point(0, 8.25, .25), Vector(4, 0, 0), Vector(0, .15, 0)));
  north_tube->add(new Rect(white, Point(0, 8.475, .1), Vector(4, 0, 0), Vector(0, -.025, -.1)));
  north_tube->add(new Cylinder(white, Point(-4, 8.4, .2), Point(4, 8.4, .2), .05));
  // north curb
  north_tube->add(new Rect(white, Point(0, 11.75, .25), Vector(4, 0, 0), Vector(0, .15, 0)));
  north_tube->add(new Rect(white, Point(0, 11.525, .1), Vector(4, 0, 0), Vector(0, .025, -.1)));
  north_tube->add(new Cylinder(white, Point(-4, 11.6, .2), Point(4, 11.6, .2), .05));
  // seals
  // west seal
  south_tube->add(new Ring(black, Point(-4, 10, 1), Vector(1, 0, 0), 1.9, .3));
  south_tube->add(new Ring(black, Point(-3.9, 10, 1), Vector(1, 0, 0), 1.9, .3));
  south_tube->add(new Cylinder(black, Point(-3.9, 10, 1), Point(-4, 10, 1), 1.9));
  south_tube->add(new Cylinder(black, Point(-3.9, 10, 1), Point(-4, 10, 1), 2.2));
  // east seal
  south_tube->add(new Ring(black, Point(4, 10, 1), Vector(1, 0, 0), 1.9, .3));
  south_tube->add(new Ring(black, Point(3.9, 10, 1), Vector(1, 0, 0), 1.9, .3));
  south_tube->add(new Cylinder(black, Point(3.9, 10, 1), Point(4, 10, 1), 1.9));
  south_tube->add(new Cylinder(black, Point(3.9, 10, 1), Point(4, 10, 1), 2.2)); 
    
  all_tubes->add(north_tube);

  /**********************************************************************/
  // west tube

  // glass tube
  //Object *west_tube_inner = new Cylinder(glass_to_air, Point(-10, -4, 1), Point(-10, 4, 1), 2);
  Object *west_tube_outer = new Cylinder(water_to_glass, Point(-10, -4, 1), Point(-10, 4, 1), 2.05);
  //west_tube->add(west_tube_inner);
  west_tube->add(west_tube_outer);

  // floor
  west_tube->add(new Rect(checker, Point(-10, 0, 0), Vector(1.5, 0, 0), Vector(0, 4, 0)));
  // east curb
  west_tube->add(new Rect(white, Point(-8.25, 0, .25), Vector(0, 4, 0), Vector(.15, 0, 0)));
  west_tube->add(new Rect(white, Point(-8.475, 0, .1), Vector(0, 4, 0), Vector(-.025, 0, -.1)));
  west_tube->add(new Cylinder(white, Point(-8.4, -4, .2), Point(-8.4, 4, .2), .05));
  // west curb
  west_tube->add(new Rect(white, Point(-11.75, 0, .25), Vector(0, 4, 0), Vector(.15, 0, 0)));
  west_tube->add(new Rect(white, Point(-11.525, 0, .1), Vector(0, 4, 0), Vector(.025,0, -.1)));
  west_tube->add(new Cylinder(white, Point(-11.6, -4, .2), Point(-11.6, 4, .2), .05));
  // seals
  // west seal
  south_tube->add(new Ring(black, Point(-10, -4, 1), Vector(0, 1, 0), 1.9, .3));
  south_tube->add(new Ring(black, Point(-10, -3.9, 1), Vector(0, 1, 0), 1.9, .3));
  south_tube->add(new Cylinder(black, Point(-10, -3.9, 1), Point(-10, -4, 1), 1.9));
  south_tube->add(new Cylinder(black, Point(-10, -3.9, 1), Point(-10, -4, 1), 2.2));
  // east seal
  south_tube->add(new Ring(black, Point(-10, 4, 1), Vector(0, 1, 0), 1.9, .3));
  south_tube->add(new Ring(black, Point(-10, 3.9, 1), Vector(0, 1, 0), 1.9, .3));
  south_tube->add(new Cylinder(black, Point(-10, 3.9, 1), Point(-10, 4, 1), 1.9));
  south_tube->add(new Cylinder(black, Point(-10, 3.9, 1), Point(-10, 4, 1), 2.2)); 
 
  all_tubes->add(west_tube);

  /**********************************************************************/
  // east tube
  
  Point center_point1(10, .25, 0);
  Vector center_vec1(10, .25, 0);
  Point south_point1(10, -3.5, 1);
  Point north_point1(10, 4, 1);
  Vector north_vec1(10, 4, 1);
  Vector south_vec1(10, -3.5, 1);
  //Object *east_tube_inner = new Cylinder(glass_to_air, Point(10, -4, 1), Point(10, 4, 1), 2);
  Object *east_tube_outer = new Cylinder(water_to_glass, south_point1, north_point1, 2.05);
  //east_tube->add(east_tube_inner);
  east_tube->add(east_tube_outer);
  // floor
  east_tube->add(new Rect(checker, center_point1, Vector(1.5, 0, 0), Vector(0, 3.75, 0)));
  // west curb
  east_tube->add(new Rect(white, Point(8.25, .25, .25), Vector(0, 3.75, 0), Vector(-.15, 0, 0)));
  east_tube->add(new Rect(white, Point(8.475, .25, .1), Vector(0, 3.75, 0), Vector(.025, 0, -.1)));
  east_tube->add(new Cylinder(white, Point(8.4, -3.5, .2), Point(8.4, 4, .2), .05));
  // east curb
  east_tube->add(new Rect(white, Point(11.75, .25, .25), Vector(0, 3.75, 0), Vector(-.15, 0, 0)));
  east_tube->add(new Rect(white, Point(11.525, .25, .1), Vector(0, 3.75, 0), Vector(-.025,0, -.1)));
  east_tube->add(new Cylinder(white, Point(11.6, -3.5, .2), Point(11.6, 4, .2), .05));
  // seals
  // south seal
  south_tube->add(new Ring(black, Point(10, -3.5, 1), Vector(0, 1, 0), 1.9, .3));
  south_tube->add(new Ring(black, Point(10, -3.4, 1), Vector(0, 1, 0), 1.9, .3));
  south_tube->add(new Cylinder(black, Point(10, -3.4, 1), Point(10, -3.5, 1), 1.9));
  south_tube->add(new Cylinder(black, Point(10, -3.4, 1), Point(10, -3.5, 1), 2.2));
  // north seal
  south_tube->add(new Ring(black, Point(10, 4, 1), Vector(0, 1, 0), 1.9, .3));
  south_tube->add(new Ring(black, Point(10, 3.9, 1), Vector(0, 1, 0), 1.9, .3));
  south_tube->add(new Cylinder(black, Point(10, 3.9, 1), Point(10, 4, 1), 1.9));
  south_tube->add(new Cylinder(black, Point(10, 3.9, 1), Point(10, 4, 1), 2.2));
 
  all_tubes->add(east_tube);

  /*********************************************************************/
  // ruins

  // base steps
  Object* step0 = new Box(seawhite, Point(-16.5, -16.5, -1.5), Point(16.5, 16.5, -1.3)); 
  Object* step1 = new Box(seawhite, Point(-16, -16, -1.3), Point(16, 16, -1.1)); 
  Object* step2 = new Box(seawhite, Point(-15.5, -15.5, -1.1), Point(15.5, 15.5, -.9)); 
//  Object* step3 = new Box(seawhite, Point(-7, -4, -.9), Point(7, 7, -.7)); 
//  Object* step4 = new Box(seawhite, Point(-6.5, -3.5, -.7), Point(6.5, 6.5, -.5)); 
//  Object* step5 = new Box(seawhite, Point(-6, -3, -.5), Point(6, 6, -.3)); 
  ruins->add(step0);
  ruins->add(step1);
  ruins->add(step2);
//  ruins->add(step3);
//  ruins->add(step4);
//  ruins->add(step5);

  ruins->add(new Box(seawhite, Point(-6.5, -3.5, -.7), Point(-4.5, 6.5, -.5)));
  ruins->add(new Box(seawhite, Point(-7, -4, -.9), Point(-4, 7, -.7)));
  ruins->add(new Box(seawhite, Point(6.5, -3.5, -.7), Point(4.5, 6.5, -.5)));
  ruins->add(new Box(seawhite, Point(7, -4, -.9), Point(4, 7, -.7)));
 
//  ruins->add(new Ring(seawhite, Point(0 , 0, -.7), Vector(0, 0, 1), 3.5, .5));
//  ruins->add(new Cylinder(seawhite, Point(0, 0, -.9), Point(0, 0, -.7), 3.5));
  ruins->add(new Ring(seawhite, Point(0 , 0, -.7), Vector(0, 0, 1), 4, .5));
  ruins->add(new Cylinder(seawhite, Point(0, 0, -.9), Point(0, 0, -.7), 4));
  ruins->add(new Ring(seawhite, Point(0 , 0, -.5), Vector(0, 0, 1), 4.5, 2));
  ruins->add(new Cylinder(seawhite, Point(0, 0, -.7), Point(0, 0, -.5), 4.5));
  ruins->add(new Cylinder(seawhite, Point(0, 0, -.9), Point(0, 0, -.5), 6.5));

  // pedestals for columns
  ruins->add(new Box(seawhite, Point(-5.5, -13, -.8), Point(5.5, -10, -.2)));

  Array1<Material *> matls;
  string env_map;
  Transform t, t1, t2, t3;
  
/*
  // video textures test/////////////////////////////////////////////////////////////////////////
  Material* vid = new VideoMap("/usr/sci/data/Geometry/models/videotex/gatcha%d.ppm", 550, 20, 
	                       Color(.7, .7, .7), 50, .3); 
  all_tubes->add(new Rect(vid, Point(0, 4, 4), Vector(2, 0, 0), Vector(0, 0, -2)));  
*/  
  /**********************************************************************/
  // gazebo ////////////////////////////////////////
/*  t3.load_identity();
  t3.pre_scale(Vector(.0005, .0005, .0005));
  t3.pre_translate(Vector(-3.9, -2.6, -.80));
  if (!readObjFile("/usr/sci/data/Geometry/models/read_in_models/gaz.obj",
                   "/usr/sci/data/Geometry/models/read_in_models/gaz.mtl",
                   t3, gazebo))
    exit(-1);
*/
  /**********************************************************************/
  // fishies!!

  //Group *seahorse1 = new Group;
  Group *shark1 = new Group;
  Group *tiger = new Group;
 /* 
  t3.load_identity();
  t3.pre_scale(Vector(1.2, 1.2, 1.2));
  t3.pre_translate(Vector(-1, -3, 1.5));
  if (!readObjFile("/usr/sci/data/Geometry/models/read_in_models/fish/fish5/fish5.obj",
                   "/usr/sci/data/Geometry/models/read_in_models/fish/fish5/fish5.mtl",
                   t3, shark1))
      exit(-1);
   Grid* grid = new Grid(shark1, 30); 
   TimeVaryingInstance* TVI1 = new TimeVaryingInstance(new InstanceWrapperObject(grid));   
   Grid2* dynamicGrid1 = new Grid2(TVI1, 20);
   TVI1->set_anim_grid(dynamicGrid1);
   all_tubes->add(dynamicGrid1);
*/
  
  t3.load_identity();
  t3.pre_scale(Vector(1.2, 1.2, 1.2));
  t3.pre_translate(Vector(0, -3, 1.5));
  t3.pre_rotate(M_PI * .1, Vector(0, 0, 1));
  if (!readObjFile("/usr/sci/data/Geometry/models/read_in_models/fish/fish5/fish5.obj",
                   "/usr/sci/data/Geometry/models/read_in_models/fish/fish5/fish5.mtl",
                   t3, shark1))
      exit(-1);
/*  
   Sphere* extend1 = new Sphere(white, Point(-6, -6, 1), .0001);
   Sphere* extend2 = new Sphere(white, Point(6, 6, 2.5), .0001);
   
   Grid* grid = new Grid(shark1, 100); 
   TimeVaryingInstance* TVI1 = new TimeVaryingInstance(new InstanceWrapperObject(grid));   
   Group* extend_group = new Group();
   extend_group->add(TVI1);
   extend_group->add(extend1);
   extend_group->add(extend2);
   Grid2* dynamicGrid1 = new Grid2(extend_group, 20);
   TVI1->set_anim_grid(dynamicGrid1);
   all_tubes->add(dynamicGrid1);
*/    
   Grid* grid = new Grid(shark1, 30);
    
   FishInstance1* TVI1 
      = new FishInstance1(new InstanceWrapperObject(grid), .25, .25, .3, .5, .05, -3, 0);
   FishInstance1* TVI2
      = new FishInstance1(new InstanceWrapperObject(grid), .20, .2, .25, .6, .05, 0, -.5);   
   FishInstance1* TVI3
      = new FishInstance1(new InstanceWrapperObject(grid), .25, .28, .275, .7, .05, 1.5, .7);
   FishInstance1* TVI4
      = new FishInstance1(new InstanceWrapperObject(grid), .15, .31, .31, .4, .05, 4.5, .2); 
   FishInstance1* TVI5
      = new FishInstance1(new InstanceWrapperObject(grid), .2, .26, .325, .8, .05, -8.5, -.25);   
   FishInstance1* TVI6
      = new FishInstance1(new InstanceWrapperObject(grid), .35, .38, .3, .65, .05, -6.5, .2);   
   FishInstance1* TVI7
      = new FishInstance1(new InstanceWrapperObject(grid), .15, .2, .26, .55, .05, -2, .6);   
   
  t3.load_identity();
  t3.pre_scale(Vector(1.3, 1.3, 1.3));
  //t3.pre_translate(Vector(2, 7, 1));
  t3.pre_translate(Vector(0, 3, 1));
  if (!readObjFile("/usr/sci/data/Geometry/models/read_in_models/fish/fish8/SiameseTiger.obj",
                   "/usr/sci/data/Geometry/models/read_in_models/fish/fish8/SiameseTiger.mtl",
                   t3, tiger))
     exit(-1);
   
   Grid* grid2 = new Grid(tiger, 30);
   FishInstance2* TVI8
      = new FishInstance2(new InstanceWrapperObject(grid2), .35, .25, .3, .4, .05, 5, .28);
   FishInstance2* TVI9
      = new FishInstance2(new InstanceWrapperObject(grid2), .25, .2, .35, .6, .05, 0, 0);
   FishInstance2* TVI10
      = new FishInstance2(new InstanceWrapperObject(grid2), .3, .15, .25, .5, .05, 3, -.2);

   all_tubes->add(TVI1);
   all_tubes->add(TVI2);
   all_tubes->add(TVI3);
   all_tubes->add(TVI4);
   all_tubes->add(TVI5);
   all_tubes->add(TVI6);
   all_tubes->add(TVI7);

   all_tubes->add(TVI8);
   all_tubes->add(TVI9);
   all_tubes->add(TVI10);
 
   all_tubes->add(ruins);  
  
   Group* col8 = new Group;
   Group* col9 = new Group;
   Group* col10 = new Group;
  
  /*********************************************************************/
  // bubbles

  bubbles->add(new AirBubble(air_bubble, Point(-.25, 6.88, 0) , .05  , 10, .5, .8));
  bubbles->add(new AirBubble(air_bubble, Point(.2, 6.8, -1)   , .09, 10, .7, .7));
  bubbles->add(new AirBubble(air_bubble, Point(-.1, 7.12, -2)  , .1  , 11, .75, .7));
  bubbles->add(new AirBubble(air_bubble, Point(1.125, 7.25, 3), .125 , 12, .8, .5));
  bubbles->add(new AirBubble(air_bubble, Point(-.15, 7.0, 0)   , .05  , 11, .5, .8));
  bubbles->add(new AirBubble(air_bubble, Point(-1.1, 7.3, -1)  , .06 , 10, .55, .8));
  bubbles->add(new AirBubble(air_bubble, Point(1.1, 7.1, -1)  , .03 , 10, .55, .9));
  bubbles->add(new AirBubble(air_bubble, Point(-1.8, 7.3, -1.5)  , .03 , 10, .55, .9));
  bubbles->add(new AirBubble(air_bubble, Point(-2.1, 6.9, -2)  , .03 , 10, .55, .9));
  bubbles->add(new AirBubble(air_bubble, Point(-4.1, 6.8, -2.5)  , .03 , 10, .55, .9));
  all_tubes->add(bubbles);

   
  /*********************************************************************/
  // ocean floor
  
  // add a plane for the ocean floor base
  all_tubes->add(new Heightfield<BrickArray2<float>, Array2<HMCell<float> > >(tan, "ocean_floor", 3, 8));
  all_tubes->add(east_tube);

  all_tubes->add(new Rect(tan, Point(-100, 0, -1.5), Vector(50, 0, 0), Vector(0, 150, 0)));  
  all_tubes->add(new Rect(tan, Point(100, 0, -1.5), Vector(50, 0, 0), Vector(0, 150, 0)));  
  all_tubes->add(new Rect(tan, Point(0, 100, -1.5), Vector(50, 0, 0), Vector(0, 50, 0)));  
  all_tubes->add(new Rect(tan, Point(0, -100, -1.5), Vector(50, 0, 0), Vector(0, 50, 0)));  

  Color cdown(0.1, 0.1, 0.1);
  Color cup(0.7, 0.7, 0.7);

  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(0.1, 0.1, 0.4);
  Scene *scene = new Scene(all_tubes, cam, bgcolor, cdown, cup, groundplane, 0.5);
  scene->addObjectOfInterest(bubbles, true); 
  scene->addObjectOfInterest(TVI1, true);
  scene->addObjectOfInterest(TVI2, true);
  scene->addObjectOfInterest(TVI3, true);
  scene->addObjectOfInterest(TVI4, true);
  scene->addObjectOfInterest(TVI5, true);
  scene->addObjectOfInterest(TVI6, true);
  scene->addObjectOfInterest(TVI7, true);

  scene->addObjectOfInterest(TVI8, true);
  scene->addObjectOfInterest(TVI9, true);
  scene->addObjectOfInterest(TVI10, true);
  
  scene->select_shadow_mode( No_Shadows );
  scene->maxdepth = 8;
  //scene->add_light(new Light(Point(20, 20, 90), Color(.3,.3,.3), 0));
  //scene->add_light(new Light(Point(0, -10, 2.9), Color(.8,.8,.8), 0));
  //scene->add_light(new Light(Point(0, 10, 2.9), Color(.8,.8,.8), 0));
  //scene->add_light(new Light(Point(-10, 0, 2.9), Color(.8,.8,.8), 0));
  //scene->add_light(new Light(Point(10, 0, 2.9), Color(.8,.8,.8), 0));
  scene->animate=true;
  return scene;
}

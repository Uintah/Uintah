#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/TimeVaryingInstance.h>
#include <Packages/rtrt/Core/InstanceWrapperObject.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/HierarchicalGrid.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/AirBubble.h>
#include <Packages/rtrt/Core/Box.h>
#include <Packages/rtrt/Core/Ring.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Heightfield.h>
#include <Packages/rtrt/Core/BrickArray2.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/InvisibleMaterial.h>
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

  Point Eye(-8.85, -10, 2.0);
  Point Lookat(5.85, -10, 2.0);
  Vector Up(0,0,1);
  //Point Eye (0, 0, 10);
  //Point Lookat(0, 0, 0);
  //Vector Up (0, 1, 0);
  double fov=60;

  Camera cam(Eye,Lookat,Up,fov);

  Group *all_tubes = new Group;
  Group* erect_group = new Group;  
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
  Group *col8 = new Group;
  Group *col9 = new Group;
  Group *col10 = new Group;
  Group *patch1 = new Group;
  Group *patch2 = new Group;
  Group *patch3 = new Group;
  Group *patch4 = new Group;
  Group *patch5 = new Group;
  Group *patch6 = new Group;
  Group *gazebo = new Group;
  Group *erect = new Group;
  Group *parth = new Group;
  Group *panth = new Group;
  Group *temple = new Group;
  Group *temple2 = new Group;
  Group *bubbles = new Group;
  Group *rock1 = new Group;
  Group *rock2 = new Group;
  Group *rock3 = new Group;
  Group *rock4 = new Group;
  Group *iceberg = new Group;
  Group *iceberg2 = new Group;
  Group *iceberg3 = new Group;
  Group *craters = new Group;
  Group *rock_tower = new Group;

  TimeVaryingCheapCaustics* tvcc= new TimeVaryingCheapCaustics("/opt/SCIRun/data/Geometry/textures/caustics/caust%d.pgm", 32,
	                                                        Point(0,0,6), Vector(1,0,0), Vector(0,1,0),
							        Color(0.5,0.5,0.5), 0.1, .3);// last should be .6
  
  Material* water_to_glass = new DielectricMaterial(1.0, 1.2, 0.0, 400.0, Color(.97, .98, 1), Color(1, 1, 1), false, 1);
  Material* air_bubble     = new DielectricMaterial(1.0, 1.1, 0.004, 400.0, Color(1, 1, 1), Color(1.01,1.01,1.01), false);
//  Material* water_to_glass   = new PhongMaterial(Color(.5, .5, .5), .2, 0.8, 20, true); 
//  Material* water_to_glass   = new InvisibleMaterial(); 
  Material* white = new LambertianMaterial(Color(0.8,0.8,0.8));
  Material* red = new LambertianMaterial(Color(1,0,0));
  SeaLambertianMaterial* tan = new SeaLambertianMaterial(Color(0.6,0.6,0.2), tvcc);
  SeaLambertianMaterial* seawhite = new SeaLambertianMaterial(Color(0.3,0.3,0.3), tvcc);
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
  south_tube->add(new Ring(black, Point(-3.8, -6, 1), Vector(1, 0, 0), 1.9, .3));
  south_tube->add(new Ring(black, Point(-3.7, -6, 1), Vector(1, 0, 0), 1.9, .3));
  south_tube->add(new Cylinder(black, Point(-3.7, -6, 1), Point(-3.8, -6, 1), 1.9));
  south_tube->add(new Cylinder(black, Point(-3.7, -6, 1), Point(-3.8, -6, 1), 2.2));
  // east seal
  south_tube->add(new Ring(black, Point(7.5, -6, 1), Vector(1, 0, 0), 1.9, .3));
  south_tube->add(new Ring(black, Point(7.4, -6, 1), Vector(1, 0, 0), 1.9, .3));
  south_tube->add(new Cylinder(black, Point(7.4, -6, 1), Point(7.5, -6, 1), 1.9));
  south_tube->add(new Cylinder(black, Point(7.4, -6, 1), Point(7.5, -6, 1), 2.2));

  Object * bv = new BV1(south_tube);
  bv->set_name("south_tube");
  all_tubes->add(bv);
  

  /**********************************************************************/
  // north tube

  //  Object *north_tube_inner = new Cylinder(glass_to_air, Point(-4, 10, 1), Point(4, 10, 1), 2);
  Object *north_tube_outer = new Cylinder(water_to_glass, Point(-4, 10, 1), Point(4, 10, 1), 2.05);
  //  north_tube->add(north_tube_inner);
  north_tube->add(north_tube_outer);
  // floor
  north_tube->add(new Rect(checker, Point(0, 10, 0), Vector(4, 0, 0), Vector(0, 1.5 ,0)));
  // south curb
  north_tube->add(new Rect(white, Point(0, 8.25, .25), Vector(4, 0, 0), Vector(0, .15, 0)));
  north_tube->add(new Rect(white, Point(0, 8.475, .1), Vector(4, 0, 0), Vector(0, .025, -.1)));
  north_tube->add(new Cylinder(white, Point(-4, 8.4, .2), Point(4, 8.4, .2), .05));
  // north curb
  north_tube->add(new Rect(white, Point(0, 11.75, .25), Vector(4, 0, 0), Vector(0, .15, 0)));
  north_tube->add(new Rect(white, Point(0, 11.525, .1), Vector(4, 0, 0), Vector(0, -.025, -.1)));
  north_tube->add(new Cylinder(white, Point(-4, 11.6, .2), Point(4, 11.6, .2), .05));
  // seals
  // west seal
  north_tube->add(new Ring(black, Point(-3.8, 10, 1), Vector(1, 0, 0), 1.9, .3));
  north_tube->add(new Ring(black, Point(-3.7, 10, 1), Vector(1, 0, 0), 1.9, .3));
  north_tube->add(new Cylinder(black, Point(-3.7, 10, 1), Point(-3.8, 10, 1), 1.9));
  north_tube->add(new Cylinder(black, Point(-3.7, 10, 1), Point(-3.8, 10, 1), 2.2));
  // east seal
  north_tube->add(new Ring(black, Point(4, 10, 1), Vector(1, 0, 0), 1.9, .3));
  north_tube->add(new Ring(black, Point(3.9, 10, 1), Vector(1, 0, 0), 1.9, .3));
  north_tube->add(new Cylinder(black, Point(3.9, 10, 1), Point(4, 10, 1), 1.9));
  north_tube->add(new Cylinder(black, Point(3.9, 10, 1), Point(4, 10, 1), 2.2)); 
  
  bv = new BV1(north_tube);
  bv->set_name("north_tube");
  all_tubes->add(bv);
   
  /**********************************************************************/
  // west tube

  // glass tube
  //  Object *west_tube_inner = new Cylinder(glass_to_air, Point(-10, -4, 1), Point(-10, 4, 1), 2);
  Object *west_tube_outer = new Cylinder(water_to_glass, Point(-10, -4, 1), Point(-10, 4, 1), 2.05);
  //  west_tube->add(west_tube_inner);
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
  west_tube->add(new Ring(black, Point(-10, -3.8, 1), Vector(0, 1, 0), 1.9, .3));
  west_tube->add(new Ring(black, Point(-10, -3.7, 1), Vector(0, 1, 0), 1.9, .3));
  west_tube->add(new Cylinder(black, Point(-10, -3.7, 1), Point(-10, -3.8, 1), 1.9));
  west_tube->add(new Cylinder(black, Point(-10, -3.7, 1), Point(-10, -3.8, 1), 2.2));
  // east seal
  west_tube->add(new Ring(black, Point(-10, 3.8, 1), Vector(0, 1, 0), 1.9, .3));
  west_tube->add(new Ring(black, Point(-10, 3.7, 1), Vector(0, 1, 0), 1.9, .3));
  west_tube->add(new Cylinder(black, Point(-10, 3.7, 1), Point(-10, 3.8, 1), 1.9));
  west_tube->add(new Cylinder(black, Point(-10, 3.7, 1), Point(-10, 3.8, 1), 2.2)); 
 
  bv = new BV1(west_tube);
  bv->set_name("west_tube");

  all_tubes->add(bv);

      
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
  east_tube->add(new Ring(black, Point(10, -3.5, 1), Vector(0, 1, 0), 1.9, .3));
  east_tube->add(new Ring(black, Point(10, -3.4, 1), Vector(0, 1, 0), 1.9, .3));
  east_tube->add(new Cylinder(black, Point(10, -3.4, 1), Point(10, -3.5, 1), 1.9));
  east_tube->add(new Cylinder(black, Point(10, -3.4, 1), Point(10, -3.5, 1), 2.2));
  // north seal
  east_tube->add(new Ring(black, Point(10, 4, 1), Vector(0, 1, 0), 1.9, .3));
  east_tube->add(new Ring(black, Point(10, 3.9, 1), Vector(0, 1, 0), 1.9, .3));
  east_tube->add(new Cylinder(black, Point(10, 3.9, 1), Point(10, 4, 1), 1.9));
  east_tube->add(new Cylinder(black, Point(10, 3.9, 1), Point(10, 4, 1), 2.2));

  bv = new BV1(east_tube);
  bv->set_name("east_tube");
  all_tubes->add(bv);

    
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
  ruins->add(new Box(seawhite, Point(-4.5, -13, -.8), Point(6.5, -10, -.2)));


  ruins->add(new Box(seawhite, Point(7.3,-15.5,-1.5), Point(19.5,-3.3,0)));
  ruins->add(new Box(seawhite, Point(-23.3,-25.5,-1.5), Point(-3.5,-3.5,0)));
  
  Array1<Material *> matls;
  string env_map;
  Transform t, t1, t2, t3;
  
  /**********************************************************************/ 
  // read in columns 
  t.pre_scale(Vector(.02, .02, .02));
  t.pre_translate(Vector(-2.5, -10.81, .67));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/column01/COLUMN.obj",
	           "/opt/SCIRun/data/Geometry/models/column01/COLUMN.mtl",
	           t,col1))
        exit(1);
  
  t.pre_translate(Vector(8.0 / 3.0, 0, 0));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/column01/COLUMN.obj",
                   "/opt/SCIRun/data/Geometry/models/column01/COLUMN.mtl",
                   t,col2))
        exit(1);
  
  
  t.pre_translate(Vector(8.0 / 3.0, 0, 0));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/column01/COLUMN.obj",
                   "/opt/SCIRun/data/Geometry/models/column01/COLUMN.mtl",
                   t,col3))
        exit(1);

  t.pre_translate(Vector(8.0 / 3.0, 0, 0));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/column01/COLUMN.obj",
                   "/opt/SCIRun/data/Geometry/models/column01/COLUMN.mtl",
                   t,col4))
        exit(1);
  
  t.load_identity();
  t.pre_scale(Vector(1, 1, 2));
  t.pre_translate(Vector(4, 16, 1));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/marble_tex/pedastal-01.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/marble_tex/pedastal-01.mtl",
                   t,col5))
        exit(1);

  t.pre_translate(Vector(-8, 0, 0));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/marble_tex/pedastal-01.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/marble_tex/pedastal-01.mtl",
                   t,col6))
     exit(1);
  
  // columns by west tube
  t.load_identity();
  t.pre_scale(Vector(.02, .02, .02));
  t.pre_rotate(-1.3, Vector(0, 1, 0));
  t.pre_rotate(-.4, Vector(0, 0, 1));
  t.pre_translate(Vector(-16, 5, -1.2));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/column01/COLUMN.obj",
                   "/opt/SCIRun/data/Geometry/models/column01/COLUMN.mtl",
                   t,col7))
        exit(1);

  // columns by west tube
  t.load_identity();
  t.pre_scale(Vector(.02, .02, .02));
  t.pre_translate(Vector(-16, -2, -.8));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/column01/COLUMN.obj",
                   "/opt/SCIRun/data/Geometry/models/column01/COLUMN.mtl",
                   t,col8))
        exit(1);
  // columns by west tube
  t.load_identity();
  t.pre_scale(Vector(.02, .02, .02));
  t.pre_rotate(.1, Vector(0, 1, 0));
  t.pre_rotate(.2, Vector(1, 0, 0));
  t.pre_translate(Vector(-19, -2, -.8));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/column01/COLUMN.obj",
                   "/opt/SCIRun/data/Geometry/models/column01/COLUMN.mtl",
                   t,col9))
        exit(1);
  // columns by west tube
  t.load_identity(); 
  t.pre_scale(Vector(.02, .02, .02));
  t.pre_translate(Vector(-19, 5, -.8));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/column01/COLUMN.obj",
                   "/opt/SCIRun/data/Geometry/models/column01/COLUMN.mtl",
                   t,col10))
        exit(1);
  
  
  /*********************************************************************/
  // read in rocks
  
  t1.pre_rotate(.5, Vector(0, 1, 0));
  t1.pre_scale(Vector(.005, .005, .005)); 
  t1.pre_translate(Vector(-7, -4.5, -3.5));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/rocks/stein002.obj",
	           "/opt/SCIRun/data/Geometry/models/read_in_models/rocks/stein002.mtl",
		    t1,rock3))
        exit(1);

  t1.load_identity();
  t1.pre_rotate(.6, Vector(0, 0, 1));
  t1.pre_scale(Vector(.005, .005, .005));
  t1.pre_translate(Vector(4.2, 5.5, -3.5));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/rocks/stein002.obj", 
                   "/opt/SCIRun/data/Geometry/models/read_in_models/rocks/stein002.mtl", 
                    t1,rock4))
        exit(1);

  t1.load_identity();
  t1.pre_rotate(.6, Vector(0, 0, 1));
  t1.pre_scale(Vector(.05, .05, .05));
  t1.pre_translate(Vector(14.2, -3.5, -43.5));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/rocks/stein003.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/rocks/stein003.mtl",
                   t1,rock2))
  
        exit(1);

  t1.load_identity();
  t1.pre_rotate(.6, Vector(0, 0, 1));
  t1.pre_scale(Vector(.05, .05, .05));
  t1.pre_translate(Vector(-33.2, 19.5, -43.5));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/rocks/stein003.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/rocks/stein003.mtl",
                   t1,rock1))
  
        exit(1);

  t1.load_identity();
  //t1.pre_rotate(.6, Vector(0, 0, 1));
  t1.pre_scale(Vector(.08, .05, .05));
  t1.pre_translate(Vector(33.2, 36.5, -26.5));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/rocks/stein002.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/rocks/stein002.mtl",
                   t1,temple))

        exit(1);

  t1.load_identity();
  t1.pre_rotate(M_PI, Vector(0, 0, 1));
  t1.pre_scale(Vector(.07, .05, .05));
  t1.pre_translate(Vector(-20.2, 80.5, -26.5));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/rocks/stein002.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/rocks/stein002.mtl",
                   t1,temple))

        exit(1);
 
  t1.load_identity();
  t1.pre_rotate(M_PI, Vector(0, 0, 1));
  t1.pre_scale(Vector(.12, .05, .05));
  t1.pre_translate(Vector(0, 80.5, -16.5));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/rocks/stein002.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/rocks/stein002.mtl",
                   t1,temple))

        exit(1);

  /**********************************************************************/
  // this is the clump near southwest corner
  t2.pre_scale(Vector(.03, .03, .03));
  t2.pre_translate(Vector(-6.5, -3, .5));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aqua11.obj", 
	           "/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aqua11.mtl",
		   t2, patch1))
     exit(-1);
  
  t2.pre_translate(Vector(-.75, .5, 0));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aqua11.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aqua11-2.mtl",
                   t2, patch1))
     exit(-1);  

  /**********************************************************************/
  // plant patch by north tube
  t3.pre_scale(Vector(.08, .02, .02));
  t3.pre_translate(Vector(-1.5, 7.3, 0));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aqua11.obj",
	           "/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aqua11-3.mtl",
		   t3, patch2))
     exit(-1);
  
  t3.load_identity();
  t3.pre_scale(Vector(.04, .03, .03));
  t3.pre_translate(Vector(1, 7.1, .3));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aqua11.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aqua11.mtl",
                   t3, patch2))
     exit(-1);
  t3.load_identity();
  t3.pre_scale(Vector(.03, .02, .025));
  t3.pre_translate(Vector(4, 6.8, .31));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aqua11.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aqua11-2.mtl",
                   t3, patch2))
    exit(-1);
  
  t3.load_identity();
  t3.pre_scale(Vector(.14, .06, .03));
  t3.pre_translate(Vector(0, 13.6, .5));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aqua11.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aqua11.mtl",
                   t3, patch4))	

    exit(-1);   
  
  /**********************************************************************/
  // plant patch by east tube ////////////////////
  t3.load_identity();
  t3.pre_scale(Vector(2, 2, 2));
  t3.pre_translate(Vector(7.2, 3.7, -.1));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aq13.obj",
	           "/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aq13.mtl",
		   t3, patch3)) 
    exit(-1); 

  t3.load_identity();
  t3.pre_rotate(.4, Vector(0, 0, 1));
  t3.pre_scale(Vector(1.8, 2, 2.63));
  t3.pre_translate(Vector(7.25, 2.8, -.1));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aq13.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aq13-2.mtl",
	           t3, patch3))      
    exit(-1);
 
  t3.load_identity();
  t3.pre_rotate(2.78, Vector(0, 0, 1));
  t3.pre_scale(Vector(1.5, 1.5, 1.5));
  t3.pre_translate(Vector(7.285, 2.4, -.1));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aq13.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aq13-3.mtl",
                   t3, patch3))               
    exit(-1);

  t3.load_identity();
  t3.pre_rotate(3.78, Vector(0, 0, 1));
  t3.pre_scale(Vector(1.5, 1.5, 1.5));
  t3.pre_translate(Vector(7.285, -1.4, -.1));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aq13.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aq13-3.mtl",
                   t3, patch3))
    exit(-1);

  t3.load_identity();
  t3.pre_rotate(3.78, Vector(0, 0, 1));
  t3.pre_scale(Vector(1.8, 1.9, 1.5));
  t3.pre_translate(Vector(7.1, -1.9, -.1));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aq13.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aq13-2.mtl",
                   t3, patch3))
    exit(-1);

  /**********************************************************************/
  // plants by parthenon
  t3.load_identity();
  t3.pre_scale(Vector(.16, .06, .06));
  //t3.pre_rotate(-.8, Vector(0, 0, 1));
  t3.pre_translate(Vector(1, -15.6, 1.3));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aqua11.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/models_rtrt/aqua11.mtl",
                   t3, patch5))
    exit(-1);
  
  /**********************************************************************/
  // gazebo ////////////////////////////////////////
  t3.load_identity();
  t3.pre_scale(Vector(.0005, .0005, .0005));
  t3.pre_translate(Vector(-3.9, -2.6, -.80));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/gaz.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/gaz.mtl",
                   t3, gazebo))
    exit(-1);

  /**********************************************************************/
  // erectheion ////////////////////////////////////////
  t3.load_identity(); 
  t3.pre_scale(Vector(.37, .37, .37));
  t3.pre_translate(Vector(60, 0, -2));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/keith/erectheion/Erectheion.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/keith/erectheion/Erectheion.mtl",
                   t3, erect))
    exit(-1); 

  /**********************************************************************/
  // pantheon 
/*
  t3.load_identity();
  t3.pre_rotate(1.5, Vector(0, 0, 1));
  t3.pre_scale(Vector(.07, .07, .07));
  t3.pre_translate(Vector(-70, 0, -1));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/keith/pantheon/pantheon1.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/keith/pantheon/pantheon1.mtl",
                   t3, panth))
    exit(-1);
*/
  /**********************************************************************/
  // temple 
  t3.load_identity();
  t3.pre_scale(Vector(.002, .002, .002));
  t3.pre_translate(Vector(0, 60, 5));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/keith/architecture/stone_temple.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/keith/architecture/stone_temple.mtl",
                   t3, temple))
    exit(-1);

  /**********************************************************************/
/*  // parth 
  t3.load_identity();
  t3.pre_rotate(M_PI + 1.5, Vector(0, 0, 1));
  t3.pre_scale(Vector(.08, .04, .08));
  t3.pre_translate(Vector(0, -50, -4));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/keith/parthenon/The_Parthenon.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/keith/parthenon/The_Parthenon.mtl",
                   t3, parth))
    exit(-1);
*/
  /**********************************************************************/
  // temple2
  t3.load_identity();
  t3.pre_rotate(M_PI / 2.0 , Vector(1, 0, 0));
  t3.pre_scale(Vector(4.5, 4.5, 4.5));
  t3.pre_translate(Vector(4, -30, 0));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/models1/temple/Greek_temple.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/models1/temple/Greek_temple.mtl",
                   t3, temple2))
    exit(-1);
 
  /**********************************************************************/
  // rock tower
  t3.load_identity();
  t3.pre_scale(Vector(.03, .03, .03));
  t3.pre_translate(Vector(-40, 0 , -3));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/models1/telescope_island/TelescopeIsland.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/models1/telescope_island/TelescopeIsland.mtl",
                   t3, rock_tower))
    exit(-1);
 
  /**********************************************************************/
/*  // iceberg1
  t3.load_identity();
  t3.pre_scale(Vector(.3, .3, .3));
  t3.pre_translate(Vector(-60, 30, 0));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/models1/icebergs/ICEBERG1.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/models1/icebergs/ICEBERG1.mtl",
                   t3, iceberg))
    exit(-1);
*/
  /**********************************************************************/
  // iceberg2
  t3.load_identity();
  t3.pre_scale(Vector(1.4, .5, .9));
  t3.pre_translate(Vector(0, -70, 0));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/models1/icebergs/ICEBERG1.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/models1/icebergs/ICEBERG1.mtl",
                   t3, iceberg2))
    exit(-1);

  /**********************************************************************/
/*  // iceberg3
  t3.load_identity();
  t3.pre_scale(Vector(.3, .3, .3));
  t3.pre_translate(Vector(50, 32, 0));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/models1/icebergs/ICEBERG2.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/models1/icebergs/ICEBERG2.mtl",
                   t3, iceberg3))
    exit(-1);
*/

 
  /**********************************************************************/
  // fishies!!

  Group *shell1 = new Group;
  Group *shell2 = new Group;
  Group *shell3 = new Group;
  Group *krabbe = new Group;
  Group *school1 = new Group;
  Group *school2 = new Group;
  Group *school3 = new Group;
  Group *school4 = new Group;
  Group *tiger = new Group;
  Group *pot1 = new Group;
  
  t3.load_identity();
  t3.pre_rotate(-.25 * M_PI, Vector(0, 0, 1));
  t3.pre_rotate(.1 * M_PI, Vector(-1, 0, 0));
  t3.pre_scale(Vector(.001, .001, .001));
  t3.pre_translate(Vector(-6.8, -.9, -.4));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/fish/krabbe/krabbe.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/fish/krabbe/krabbe.mtl",
                   t3, krabbe))
      exit(-1); 
  
  t3.load_identity();
  t3.pre_scale(Vector(1.2, 1.2, 1.2));
  t3.pre_translate(Vector(0, -3, 1.5));
  t3.pre_rotate(M_PI * .1, Vector(0, 0, 1));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/fish/fish5/fish5.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/fish/fish5/fish5.mtl",
                   t3, school1))
      exit(-1);

  t3.load_identity();
  t3.pre_scale(Vector(1.3, 1.3, 1.3));
  //t3.pre_translate(Vector(2, 7, 1));
  t3.pre_translate(Vector(0, 3, 1));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/fish/fish8/SiameseTiger.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/fish/fish8/SiameseTiger.mtl",
                   t3, tiger))
     exit(-1);

 
  t3.load_identity();
  t3.pre_rotate(-.6 , Vector(1, 0, 0));
  t3.pre_scale(Vector(.04, .04, .04));
  t3.pre_translate(Vector(7, -2, .1));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/fish/schnecken1/shell1.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/fish/schnecken1/shell1.mtl",
                   t3, shell1))
      exit(-1);

  t3.pre_translate(Vector(-23.5, -2.5, 0));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/fish/schnecken1/shell1.obj",
	           "/opt/SCIRun/data/Geometry/models/read_in_models/fish/schnecken1/shell1.mtl",
	           t3, shell2))
      exit(-1);

  t.load_identity();
  t.pre_scale(Vector(.15, .15, .15));
  //t.pre_rotate(-1.3, Vector(0, 1, 0));
  t.pre_rotate(-1.1, Vector(1, 0, 0));
  t.pre_translate(Vector(14, -.5, -.4));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/fish/schnecken2/shell2.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/fish/schnecken2/shell2.mtl",
                   t,shell3))
        exit(1);

  t.load_identity();
  t.pre_scale(Vector(.03, .03, .03));
  t.pre_translate(Vector(0, 0, -8));
  t.pre_rotate(.352, Vector(1, 0, 0));
  t.pre_rotate(.2, Vector(0, 1, 0));
  t.pre_translate(Vector(-14, 1.8, -2.6));
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/oceanpots_obj/pot3.obj",
                   "/opt/SCIRun/data/Geometry/models/oceanpots_obj/pot3.mtl",
                   t,pot1))
        exit(1);

   Grid* grid = new Grid(school1, 30);

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
  if (!readObjFile("/opt/SCIRun/data/Geometry/models/read_in_models/fish/fish8/SiameseTiger.obj",
                   "/opt/SCIRun/data/Geometry/models/read_in_models/fish/fish8/SiameseTiger.mtl",
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

  Object* temp;
/*
  temp = new Grid (shell1, 15);
  temp->set_name("shell1");
  all_tubes->add(temp);

  temp = new Grid (shell2, 15);
  temp->set_name("shell2");
  all_tubes->add(temp);
  
  temp = new Grid (shell3, 15);
  temp->set_name("shell3");
  all_tubes->add(temp);
 
  temp = new Grid (pot1, 15);
  temp->set_name("pot1");
  all_tubes->add(temp);
  
  temp = new Grid (krabbe, 15);
  temp->set_name("krabbe");
  all_tubes->add(temp);  
*/
  temp = new Grid (col1, 15);
  temp->set_name("col1");
  all_tubes->add(temp);
  
  temp = new Grid (col2, 15);
  temp->set_name("col2");
  all_tubes->add(temp);
  
  temp = new Grid (col3, 15);
  temp->set_name("col3");
  all_tubes->add(temp);
  
  temp = new Grid (col4, 15);
  temp->set_name("col4");
  all_tubes->add(temp);
  
  temp = new Grid (col7, 15);
  temp->set_name("col7");
  all_tubes->add(temp);
  
  temp = new Grid (col8, 15);
  temp->set_name("col8");
  all_tubes->add(temp);
  
  temp = new Grid (col9, 15);
  temp->set_name("col9");
  all_tubes->add(temp);
  
  temp = new Grid (col10, 15);
  temp->set_name("col10");
  all_tubes->add(temp);
/*  
  //temp = new Grid (col5, 35);
  temp = (new HierarchicalGrid (col5, 6, 6, 6, 10, 10, 4));
  temp->set_name("col5");
  all_tubes->add(temp);
  //all_tubes->add(new HierarchicalGrid (col5, 6, 4, 4, 10, 10, 4));

  //temp = new Grid (col6, 35);
  temp = (new HierarchicalGrid (col6, 6, 6, 6, 10, 10, 4));
  temp->set_name("col6");
  all_tubes->add(temp);  

  temp = new HierarchicalGrid (patch1, 6, 6, 6, 10, 10, 4);
  temp->set_name("patch1");
  all_tubes->add(temp);  
  // all_tubes->add(new Grid (patch1, 25));
  
  temp = new HierarchicalGrid (patch2, 6, 6, 6, 10, 10, 4); 
  temp->set_name("patch2");
  all_tubes->add(temp);
  // all_tubes->add(new Grid (patch2, 30));

  temp = new HierarchicalGrid (patch3, 6, 6, 6, 10, 10, 4); 
  temp->set_name("patch3");
  all_tubes->add(temp);  
  //  all_tubes->add(new Grid (patch3, 20));
  // all_tubes->add(new Grid (patch3, 20));

  temp = new HierarchicalGrid (patch4, 6, 6, 6, 10, 10, 4); 
  temp->set_name("patch4");
  all_tubes->add(temp);
  // all_tubes->add(new BV1  (patch4));
  // all_tubes->add(new HierarchicalGrid (patch4, 6, 6, 6, 10, 10, 4));
  // all_tubes->add(new Grid (patch4, 20));

  temp = new HierarchicalGrid (patch5, 6, 6, 6, 10, 10, 4);
  temp->set_name("patch5");
  all_tubes->add(temp);  
  // all_tubes->add(new BV1  (patch5));
  // all_tubes->add(new Grid (patch5, 20));
*/

  temp = new HierarchicalGrid (gazebo, 8, 8, 6, 10, 10, 4);
  temp->set_name("gazebo");
  all_tubes->add(temp);  
  //  all_tubes->add(new Grid (gazebo, 15));
/*
  temp = new HierarchicalGrid (erect, 6, 6, 6, 10, 10, 4);
  temp->set_name("erect");
  all_tubes->add(temp);  
  // all_tubes->add(new Grid (erect, 10));

  temp = new HierarchicalGrid(temple, 6, 6, 6, 10, 10, 4);
  temp->set_name("temple");
  all_tubes->add(temp);  
  // all_tubes->add(new Grid(temple, 10));

  temp = new HierarchicalGrid(temple2, 6, 6, 6, 10, 10, 4);
  temp->set_name("temple2");
  all_tubes->add(temp);  
  // all_tubes->add(new Grid(temple2, 20));

  temp = new HierarchicalGrid (rock1, 5, 5, 5, 10, 10, 4);
  temp->set_name("rock1");
  all_tubes->add(temp); 
  // all_tubes->add(new Grid (rock1, 40));

  //temp = new Grid (rock2, 80);
  temp = (new HierarchicalGrid (rock2, 5, 5, 5, 10, 10, 4));
  temp->set_name("rock2");
  all_tubes->add(temp);  
  // all_tubes->add(new HierarchicalGrid (rock2, 20, 6, 6, 10, 10, 4));

  //temp = new Grid (rock3, 80);
  temp = (new HierarchicalGrid (rock3, 5, 5, 5, 10, 10, 4));
  temp->set_name("rock3");
  all_tubes->add(temp);  
  // all_tubes->add(new HierarchicalGrid (rock3, 20, 6, 6, 10, 10, 4));

  //temp = new Grid (rock4, 80);
  temp = (new HierarchicalGrid (rock4, 5, 5, 5, 10, 10, 4));
  temp->set_name("rock4");
  all_tubes->add(temp);  
  //  all_tubes->add(new HierarchicalGrid (rock4, 20, 6, 6, 10, 10, 4));

  temp = new Grid (iceberg2, 40);
  temp->set_name("iceberg2");
  all_tubes->add(temp);  

  temp = new HierarchicalGrid(rock_tower, 6, 6, 6, 10, 10, 4);
  temp->set_name("rock_tower");
  all_tubes->add(temp);  
  // all_tubes->add(new Grid(rock_tower, 10));
 */ 
//  temp = new BV1(erect_group);
//  temp->set_name("erect_group");
//  all_tubes->add(temp);

  temp = new BV1(ruins);
  temp->set_name("ruins");
  all_tubes->add(temp);  
  
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
  all_tubes->add(new Heightfield<BrickArray2<float>, Array2<HMCell<float> > >(tan, "/opt/SCIRun/data/Geometry/models/ocean_floor", 3, 8));
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
  scene->addObjectOfInterest(tan, true);
  scene->addObjectOfInterest(seawhite, true);

  
  bubbles->set_name("bubbles");

  scene->select_shadow_mode( Hard_Shadows );
  scene->maxdepth = 8;
  //scene->add_light(new Light(Point(20, 20, 90), Color(.3,.3,.3), 0));
  scene->animate=true;
  
  return scene;
}

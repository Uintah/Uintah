#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
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

  Point Eye(-5.85, -6.2, 2.0);
  Point Lookat(-13.5, -13.5, 2.0);
  Vector Up(0,0,1);
  double fov=60;

  Camera cam(Eye,Lookat,Up,fov);

  Group *all_tubes = new Group;
  Group *south_tube = new Group;
  Group *north_tube = new Group;
  Group *west_tube = new Group;
  Group *east_tube = new Group;

  Material* glass_to_air = new DielectricMaterial(1.0, 1.5, 0.04, 400.0, Color(.80, .93 , .87), Color(1,1,1), false);
  Material* water_to_glass = new DielectricMaterial(1.5, 1.3, 0.04, 400.0, Color(.80, .84 , .93), Color(1,1,1), false);
  Material* white = new LambertianMaterial(Color(0.8,0.8,0.8));

  Object *south_tube_inner = new Cylinder(glass_to_air, Point(-4, -10, 1), 
				     Point(4, -10, 1), 2);
  Object *south_tube_outer = new Cylinder(water_to_glass, Point(-4, -10, 1), 
				     Point(4, -10, 1), 2.05);
  south_tube->add(south_tube_inner);
  south_tube->add(south_tube_outer);
  south_tube->add(new Rect(white, Point(0, -10, 0), 
		      Vector(4, 0, 0), Vector(0, 2, 0)));

  all_tubes->add(south_tube);

  Object *north_tube_inner = new Cylinder(glass_to_air, Point(-4, 10, 1), 
				     Point(4, 10, 1), 2);
  Object *north_tube_outer = new Cylinder(water_to_glass, Point(-4, 10, 1), 
				     Point(4, 10, 1), 2.05);
  north_tube->add(north_tube_inner);
  north_tube->add(north_tube_outer);
  north_tube->add(new Rect(white, Point(0, 10, 0), 
		      Vector(4, 0, 0), Vector(0, 2, 0)));
  
  all_tubes->add(north_tube);

  Object *west_tube_inner = new Cylinder(glass_to_air, Point(-10, -4, 1), 
				     Point(-10, 4, 1), 2);
  Object *west_tube_outer = new Cylinder(water_to_glass, Point(-10, -4, 1), 
				     Point(-10, 4, 1), 2.05);
  west_tube->add(west_tube_inner);
  west_tube->add(west_tube_outer);
  west_tube->add(new Rect(white, Point(-10, 0, 0), 
		      Vector(2, 0, 0), Vector(0, 4, 0)));
  
  all_tubes->add(west_tube);

  Object *east_tube_inner = new Cylinder(glass_to_air, Point(10, -4, 1), 
				     Point(10, 4, 1), 2);
  Object *east_tube_outer = new Cylinder(water_to_glass, Point(10, -4, 1), 
				     Point(10, 4, 1), 2.05);
  east_tube->add(east_tube_inner);
  east_tube->add(east_tube_outer);
  east_tube->add(new Rect(white, Point(10, 0, 0), 
		      Vector(2, 0, 0), Vector(0, 4, 0)));
  
  all_tubes->add(east_tube);

  Color cdown(0.1, 0.1, 0.7);
  Color cup(0.5, 0.5, 0.0);

  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(0.1, 0.1, 0.4);
  Scene *scene = new Scene(all_tubes, cam, bgcolor, cdown, cup, groundplane, 0.5);
  scene->ambient_hack = false;

  scene->select_shadow_mode("hard");
  scene->maxdepth = 8;
  scene->add_light(new Light(Point(0, -10, 2.9), Color(.8,.8,.8), 0));
  scene->add_light(new Light(Point(0, 10, 2.9), Color(.8,.8,.8), 0));
  scene->add_light(new Light(Point(-10, 0, 2.9), Color(.8,.8,.8), 0));
  scene->add_light(new Light(Point(10, 0, 2.9), Color(.8,.8,.8), 0));
  scene->animate=false;
  return scene;
}

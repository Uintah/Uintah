#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Disc.h>
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

using namespace rtrt;

#define MAXBUFSIZE 256
#define SCALE 950

extern "C"
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
  for(int i=1;i<argc;i++) {
    cerr << "Unknown option: " << argv[i] << '\n';
    cerr << "Valid options for scene: " << argv[0] << '\n';
    return 0;
  }

  Point Eye(5.85, -6.2, 2.0);
  Point Lookat(13.5, -13.5, 2.0);
  Vector Up(0,0,1);
  double fov=60;

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
  Object* check_floor=new Rect(marble, Point(8, -8, 0),
			       Vector(4, 0, 0), Vector(0, 4, 0));
  Group* south_wall=new Group();
  Group* east_wall=new Group();
  Group* north_wall=new Group();
  Group* west_wall=new Group();
  Group* ceiling_floor=new Group();
  ceiling_floor->add(check_floor);

  Material* whittedimg = 
    new ImageMaterial(1, "/usr/sci/projects/rtrt/textures/whitted",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      Color(0,0,0), 1, Color(0,0,0), 0);
  
  Object* pic1=
    new Parallelogram(whittedimg, Point(8.65, -11.9, 2.5), 
		      Vector(0,0,-1), Vector(-1.3,0,0));

  Material* bumpimg = 
    new ImageMaterial(1, "/usr/sci/projects/rtrt/textures/bump",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      Color(0,0,0), 1, Color(0,0,0), 0);

  Object* pic2=
    new Parallelogram(bumpimg, Point(11.9, -8.65, 2.5), 
		      Vector(0, 0, -1), Vector(0, 1.3, 0));

  Material* white = new LambertianMaterial(Color(0.8,0.8,0.8));

  south_wall->add(new Rect(white, Point(8, -12, 2), 
		       Vector(4, 0, 0), Vector(0, 0, 2)));

  east_wall->add(new Rect(white, Point(12, -8, 2), 
		       Vector(0, 4, 0), Vector(0, 0, 2)));

//  north_wall->add(new Rect(white, Point(8, -4, 2), 
//		       Vector(4, 0, 0), Vector(0, 0, 2)));

  // doorway cut out of North wall for E. tube: attaches to Living Room scene

  north_wall->add(new Rect(white, Point(11.5, -4, 2), 
		       Vector(0.5, 0, 0), Vector(0, 0, 2)));
  north_wall->add(new Rect(white, Point(7.5, -4, 3), 
		       Vector(3.5, 0, 0), Vector(0, 0, 1)));
  north_wall->add(new Rect(white, Point(6.5, -4, 1), 
		       Vector(2.5, 0, 0), Vector(0, 0, 1)));

//  west_wall->add(new Rect(white, Point(4, -8, 2), 
//		       Vector(0, 4, 0), Vector(0, 0, 2)));

  // doorway cut out of West wall for S. tube: attaches to Graphics Museum scene

  west_wall->add(new Rect(white, Point(4, -11.5, 2), 
		       Vector(0, 0.5, 0), Vector(0, 0, 2)));
  west_wall->add(new Rect(white, Point(4, -7.5, 3), 
		       Vector(0, 3.5, 0), Vector(0, 0, 1)));
  west_wall->add(new Rect(white, Point(4, -6.5, 1), 
		       Vector(0, 2.5, 0), Vector(0, 0, 1)));

  ceiling_floor->add(new Rect(white, Point(8, -8, 4),
		       Vector(4, 0, 0), Vector(0, 4, 0)));

  Group *g = new Group();
  east_wall->add(pic1);
  south_wall->add(pic2);

//  g->add(new BV1(south_wall));
//  g->add(new BV1(east_wall));
//  g->add(new BV1(north_wall));
//  g->add(new BV1(north_wall));
//  g->add(new BV1(west_wall));

  g->add(ceiling_floor);
  g->add(south_wall);
  g->add(east_wall);
  g->add(north_wall);
  g->add(west_wall);
  
  Color cdown(0.1, 0.1, 0.7);
  Color cup(0.5, 0.5, 0.0);

  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(0.3, 0.3, 0.3);
  Scene *scene = new Scene(g, cam, bgcolor, cdown, cup, groundplane, 0.5);
  scene->ambient_hack = false;

  scene->select_shadow_mode("hard");
  scene->maxdepth = 8;
  scene->add_light(new Light(Point(8, -8, 3.9), Color(.8,.8,.8), 0));
  scene->animate=false;
  return scene;
}

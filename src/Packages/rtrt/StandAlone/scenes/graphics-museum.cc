/* look from above:

rtrt -np 8 -eye -18.9261 -22.7011 52.5255 -lookat -7.20746 -8.61347 -16.643 -up 0.490986 -0.866164 -0.0932288 -fov 40 -scene scenes/multi-scene 2 -scene scenes/graphics-museum -scene scenes/seaworld-tubes

look from hallway:
rtrt -np 8 -eye -5.85 -6.2 2 -lookat -8.16796 -16.517 2 -up 0 0 1 -fov 60 -scene scenes/graphics-museum 

looking at David:
rtrt -np 8 -eye -18.5048 -25.9155 1.39435 -lookat -14.7188 -16.1192 0.164304 -up 0 0 1 -fov 60  -scene scenes/graphics-museum 
*/

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
#include <Packages/rtrt/Core/Cylinder.h>

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

  Point Eye(-5.85, -6.2, 2.0);
  Point Lookat(-13.5, -13.5, 2.0);
  Vector Up(0,0,1);
  double fov=60;

  Camera cam(Eye,Lookat,Up,fov);

  Material* flat_white = new LambertianMaterial(Color(.8,.8,.8));
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
  Material* marble3=new CrowMarble(1.0, 
				   Vector(2,1,0),
				   Color(0.5,0.6,0.6),
				   Color(0.4,0.55,0.52),
				   Color(0.35,0.45,0.42)
				   );
  Material* marble4=new CrowMarble(1.5, 
				   Vector(-1,3,0),
//  				   Color(0.4,0.3,0.2),
				   Color(0,0,0),
				   Color(0.35,0.34,0.32),
				   Color(0.20,0.24,0.24)
				   );
  Material* marble=new Checker(marble1,
			       marble2,
			       Vector(3,0,0), Vector(0,3,0));
  Object* check_floor=new Rect(marble, Point(-12, -16, 0),
			       Vector(8, 0, 0), Vector(0, 12, 0));
  Group* south_wall=new Group();
  Group* west_wall=new Group();
  Group* north_wall=new Group();
  Group* east_wall=new Group();
  Group* ceiling_floor=new Group();
  Group* partitions=new Group();
  Group *baseg = new Group();

  ceiling_floor->add(check_floor);

  /*
  Material* whittedimg = 
    new ImageMaterial(1, "/usr/sci/projects/rtrt/textures/whitted",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      Color(0,0,0), 1, Color(0,0,0), 0);
  
  Object* pic1=
    new Parallelogram(whittedimg, Point(-7.35, -11.9, 2.5), 
		      Vector(0,0,-1), Vector(-1.3,0,0));

  Material* bumpimg = 
    new ImageMaterial(1, "/usr/sci/projects/rtrt/textures/bump",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      Color(0,0,0), 1, Color(0,0,0), 0);

  Object* pic2=
    new Parallelogram(bumpimg, Point(-11.9, -8.65, 2.5), 
		      Vector(0, 0, -1), Vector(0, 1.3, 0));
  */

  Material* white = new LambertianMaterial(Color(0.8,0.8,0.8));

  south_wall->add(new Rect(white, Point(-12, -28, 4), 
		       Vector(8, 0, 0), Vector(0, 0, 4)));

  west_wall->add(new Rect(white, Point(-20, -16, 4), 
		       Vector(0, 12, 0), Vector(0, 0, 4)));

  //  north_wall->add(new Rect(white, Point(-12, -4, 4), 
  //		       Vector(8, 0, 0), Vector(0, 0, 4)));
  // doorway cut out of North wall for W. tube: attaches to Hologram scene

  north_wall->add(new Rect(white, Point(-15.5, -4, 4), 
		       Vector(4.5, 0, 0), Vector(0, 0, 4)));
  north_wall->add(new Rect(white, Point(-7.5, -4, 5), 
		       Vector(3.5, 0, 0), Vector(0, 0, 3)));
  north_wall->add(new Rect(white, Point(-6.5, -4, 1), 
		       Vector(2.5, 0, 0), Vector(0, 0, 1)));

  //  east_wall->add(new Rect(white, Point(-4, -16, 4), 
  //			  Vector(0, 12, 0), Vector(0, 0, 4)));

  // doorway cut out of East wall for S. tube: attaches to Sphere Room scene

  east_wall->add(new Rect(white, Point(-4, -17.5, 4), 
		       Vector(0, 10.5, 0), Vector(0, 0, 4)));
  east_wall->add(new Rect(white, Point(-4, -6, 5), 
		       Vector(0, 1, 0), Vector(0, 0, 3)));
  east_wall->add(new Rect(white, Point(-4, -4.5, 4), 
		       Vector(0, 0.5, 0), Vector(0, 0, 4)));

  /*
  ceiling_floor->add(new Rect(white, Point(-12, -16, 8),
		       Vector(8, 0, 0), Vector(0, 12, 0)));
  */

  partitions->add(new Box(white, Point(-8-.1,-24,0),
			  Point(-8+.1,-4,5)));

  partitions->add(new Box(white, Point(-16,-24-.1,0),
			  Point(-8,-24+.1,5)));

  partitions->add(new Box(white, Point(-20,-16-.1,0),
			  Point(-12,-16+.1,5)));

  // david pedestal
  baseg->add(new Cylinder(flat_white,Point(-14,-20,0),Point(-14,-20,1),2.7/2.));
  baseg->add(new Disc(flat_white,Point(-14,-20,1),Vector(0,0,1),2.7/2.));

  // history hall pedestals
  baseg->add(new Box(flat_white,Point(-5.375,-9.25,0),Point(-4.625,-8.5,1.4)));
  baseg->add(new Box(flat_white,Point(-7.375,-11.25,0),Point(-6.625,-10.5,1.4)));
  baseg->add(new Box(flat_white,Point(-5.375,-13.25,0),Point(-4.625,-12.5,1.4)));
  baseg->add(new Box(flat_white,Point(-7.375,-15.25,0),Point(-6.625,-14.5,1.4)));
  baseg->add(new Box(flat_white,Point(-5.375,-17.25,0),Point(-4.625,-16.5,1.4)));
  baseg->add(new Box(flat_white,Point(-7.375,-19.25,0),Point(-6.625,-18.5,1.4)));
  baseg->add(new Box(flat_white,Point(-5.375,-21.25,0),Point(-4.625,-20.5,1.4)));
  baseg->add(new Box(flat_white,Point(-7.375,-23.25,0),Point(-6.625,-22.5,1.4)));
  baseg->add(new Box(flat_white,Point(-5.375,-25.25,0),Point(-4.625,-24.5,1.4)));
  baseg->add(new Box(flat_white,Point(-7.375,-27.25,0),Point(-6.625,-26.5,1.4)));

  baseg->add(new Box(flat_white,Point(-9.37,-25.25,0),Point(-8.625,-24.5,1.4)));
  baseg->add(new Box(flat_white,Point(-11.375,-27.25,0),Point(-10.625,-26.5,1.4)));
  baseg->add(new Box(flat_white,Point(-13.375,-25.25,0),Point(-12.625,-24.5,1.4)));
  baseg->add(new Box(flat_white,Point(-15.375,-27.25,0),Point(-14.625,-26.5,1.4)));



  Group *g = new Group();
  /*
  west_wall->add(pic1);
  south_wall->add(pic2);
  */

//  g->add(new BV1(south_wall));
//  g->add(new BV1(west_wall));
//  g->add(new BV1(north_wall));
//  g->add(new BV1(north_wall));
//  g->add(new BV1(east_wall));

  g->add(ceiling_floor);
  g->add(south_wall);
  g->add(west_wall);
  g->add(north_wall);
  g->add(east_wall);
  g->add(partitions);
  g->add(baseg);
  
  Color cdown(0.1, 0.1, 0.7);
  Color cup(0.5, 0.5, 0.0);

  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(0.1, 0.1, 0.6);
  Scene *scene = new Scene(g, cam, bgcolor, cdown, cup, groundplane, 0.5);
  scene->ambient_hack = false;

  scene->select_shadow_mode("hard");
  scene->maxdepth = 8;
  scene->add_light(new Light(Point(-6, -16, 7.9), Color(.8,.8,.8), 0));
  scene->add_light(new Light(Point(-12, -26, 7.9), Color(.8,.8,.8), 0));
  scene->animate=false;
  return scene;
}

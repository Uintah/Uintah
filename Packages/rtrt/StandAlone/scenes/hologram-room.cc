#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Disc.h>
#include <Packages/rtrt/Core/Ring.h>
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
#include <Packages/rtrt/Core/ASEReader.h>
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

  Point Eye(-7.59842, 7.38245, 2.6612);
  Point Lookat(-11.5014, 12.3235, -5.83386);
  Vector Up(-0.501733, 0.627447, 0.595461);
//  Point Eye(-5.85, 6.2, 2.0);
//  Point Lookat(-13.5, 13.5, 2.0);
//  Vector Up(0,0,1);
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
  Object* check_floor=new Rect(marble, Point(-8, 8, 0),
			       Vector(4, 0, 0), Vector(0, 4, 0));
  Group* north_wall=new Group();
  Group* west_wall=new Group();
  Group* south_wall=new Group();
  Group* east_wall=new Group();
  Group* table=new Group();
  Group* ceiling_floor=new Group();
  ceiling_floor->add(check_floor);

  Material* whittedimg = 
    new ImageMaterial(1, "/usr/sci/projects/rtrt/textures/whitted",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      Color(0,0,0), 1, Color(0,0,0), 0);
  
  Object* pic1=
    new Parallelogram(whittedimg, Point(-7.35, 11.9, 2.5), 
		      Vector(0,0,-1), Vector(-1.3,0,0));

  Material* bumpimg = 
    new ImageMaterial(1, "/usr/sci/projects/rtrt/textures/bump",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      Color(0,0,0), 1, Color(0,0,0), 0);

  Object* pic2=
    new Parallelogram(bumpimg, Point(-11.9, 7.35, 2.5), 
		      Vector(0, 0, -1), Vector(0, 1.3, 0));

  Material* white = new LambertianMaterial(Color(0.8,0.8,0.8));

  north_wall->add(new Rect(white, Point(-8, 12, 2), 
		       Vector(4, 0, 0), Vector(0, 0, 2)));

  west_wall->add(new Rect(white, Point(-12, 8, 2), 
		       Vector(0, 4, 0), Vector(0, 0, 2)));

//  south_wall->add(new Rect(white, Point(-8, 4, 2), 
//		       Vector(4, 0, 0), Vector(0, 0, 2)));

  // doorway cut out of South wall for W. tube: attaches to Graphic Museum scene

  south_wall->add(new Rect(white, Point(-11.5, 4, 2), 
		       Vector(0.5, 0, 0), Vector(0, 0, 2)));
  south_wall->add(new Rect(white, Point(-7.5, 4, 3), 
		       Vector(3.5, 0, 0), Vector(0, 0, 1)));
  south_wall->add(new Rect(white, Point(-6.5, 4, 1), 
		       Vector(2.5, 0, 0), Vector(0, 0, 1)));

//  east_wall->add(new Rect(white, Point(-4, 8, 2), 
//		       Vector(0, 4, 0), Vector(0, 0, 2)));

  // doorway cut out of East wall for N. tube: attaches to Living Room scene

  east_wall->add(new Rect(white, Point(-4, 11.5, 2), 
		       Vector(0, 0.5, 0), Vector(0, 0, 2)));
  east_wall->add(new Rect(white, Point(-4, 7.5, 3), 
		       Vector(0, 3.5, 0), Vector(0, 0, 1)));
  east_wall->add(new Rect(white, Point(-4, 6.5, 1), 
		       Vector(0, 2.5, 0), Vector(0, 0, 1)));

  // add the ceiling
//  ceiling_floor->add(new Rect(white, Point(-8, 8, 4),
//		       Vector(4, 0, 0), Vector(0, 4, 0)));

  west_wall->add(pic1);
  north_wall->add(pic2);

  Material *silver = new MetalMaterial(Color(0.7,0.73,0.8), 12);
  Material *air_to_glass = new DielectricMaterial(1.5, 0.66, 0.04, 400.0, Color(.87, .80, .93), Color(1,1,1), false);
  
  // top of the table is at 32 inches
  double i2m = 1./36.;             // convert inches to meters
  Point center(-8, 8, 0);

#if 0
  // N/S horizontal bar to support glass
  table->add(new Box(silver, center+Vector(-1,-24,31.85)*i2m, 
		     center+Vector(1,24,32.15)*i2m));
  
  // E/W horizontal bar to support glass
  table->add(new Box(silver, center+Vector(-24,-1,31.85)*i2m,
		     center+Vector(24,1,32.15)*i2m));
		     
  
  // connecting circle for glass supports
  table->add(new Cylinder(silver, center+Vector(0,0,32.15)*i2m, 
			  center+Vector(0,0,31.85)*i2m, 3*i2m));
  table->add(new Disc(silver, center+Vector(0,0,32.15)*i2m, 
			  Vector(0,0,1), 3*i2m));
  table->add(new Disc(silver, center+Vector(0,0,31.85)*i2m, 
			  Vector(0,0,-1), 3*i2m));

  // glass
  table->add(new Cylinder(air_to_glass, center+Vector(0,0,32.151)*i2m,
			  center+Vector(0,0,32.451)*i2m, 23.75*i2m));
  table->add(new Disc(air_to_glass, center+Vector(0,0,32.451)*i2m,
		      Vector(0,0,1), 23.75*i2m));
  table->add(new Disc(air_to_glass, center+Vector(0,0,32.151)*i2m,
		      Vector(0,0,-1), 23.75*i2m));

  // rim
  table->add(new Cylinder(silver, center+Vector(0,0,32.151)*i2m,
			  center+Vector(0,0,32.451)*i2m, 23.80*i2m));
  table->add(new Cylinder(silver, center+Vector(0,0,32.151)*i2m,
			  center+Vector(0,0,32.451)*i2m, 24.80*i2m));
  table->add(new Ring(silver, center+Vector(0,0,32.151)*i2m, Vector(0,0,-1),
		      23.80*i2m, 1.*i2m));
  table->add(new Ring(silver, center+Vector(0,0,32.451)*i2m, Vector(0,0,1),
		      23.80*i2m, 1.*i2m));

  // N leg
  table->add(new Box(silver, center+Vector(22.8,-1,0)*i2m, 
		     center+Vector(24.8,1,31.672)*i2m));
  
  // S leg
  table->add(new Box(silver, center+Vector(-24.8,-1,0)*i2m, 
		     center+Vector(-22.8,1,31.672)*i2m));
  
  // E leg
  table->add(new Box(silver, center+Vector(-1,22.8,0)*i2m, 
		     center+Vector(1,24.8,31.672)*i2m));
  
  // W leg
  table->add(new Box(silver, center+Vector(-1,-24.8,0)*i2m, 
		     center+Vector(1,-22.8,31.672)*i2m));

#endif
  
  // table base
  table->add(new Cylinder(silver, center+Vector(0,0,0.01)*i2m,
			  center+Vector(0,0,24)*i2m, 12*i2m));
  // table top
  table->add(new Cylinder(silver, center+Vector(0,0,24)*i2m,
			  center+Vector(0,0,36)*i2m, 30*i2m));
  table->add(new Disc(silver, center+Vector(0,0,24)*i2m,
		      Vector(0,0,-1)*i2m, 30*i2m));
  table->add(new Disc(silver, center+Vector(0,0,36)*i2m,
		      Vector(0,0,1)*i2m, 30*i2m));

  // TODO: need a way to bevel the corners with a normal map

  Group *g = new Group();

//  g->add(new BV1(north_wall));
//  g->add(new BV1(west_wall));
//  g->add(new BV1(south_wall));
//  g->add(new BV1(east_wall));

  g->add(ceiling_floor);
  g->add(north_wall);
  g->add(west_wall);
  g->add(south_wall);
  g->add(east_wall);
  g->add(table);

  Array1<Material *> matls;
  string env_map;
  Transform corbusier_trans;

  // first, get it centered at the origin (in x and y), and scale it
  corbusier_trans.pre_scale(Vector(0.007, 0.007, 0.007));
  corbusier_trans.pre_translate(Vector(0.5,0.2,0));

  // now rotate/translate it to the right angle/position
  for (int i=0; i<5; i++) {
    Transform t(corbusier_trans);
    double rad=(65+35*i)*(M_PI/180.);
    t.pre_rotate(rad, Vector(0,0,1));
    t.pre_translate(center.vector()+Vector(cos(rad),sin(rad),0)*2.9);
    if (!readASEFile("/usr/sci/projects/rtrt/geometry/lebebe.ASE", t, g, matls, env_map)) {
      exit(0);
    }
  }

  Color cdown(0.1, 0.1, 0.1);
  Color cup(0.1, 0.1, 0.1);

  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(0.3, 0.3, 0.3);
  Scene *scene = new Scene(g, cam, bgcolor, cdown, cup, groundplane, 0.5);
  scene->ambient_hack = false;

  scene->select_shadow_mode("none");
  scene->maxdepth = 8;
  scene->add_light(new Light(Point(-8, 8, 3.9), Color(.8,.8,.8), 0));
  scene->animate=false;
  return scene;
}

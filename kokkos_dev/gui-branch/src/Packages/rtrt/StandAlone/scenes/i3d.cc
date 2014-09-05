#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Scene.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <Packages/rtrt/Core/Point.h>
#include <Packages/rtrt/Core/Vector.h>
#include <Packages/rtrt/Core/Parallelogram.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Light.h>

using namespace rtrt;

extern "C"
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
  for(int i=1;i<argc;i++) {
    cerr << "Unknown option: " << argv[i] << '\n';
    cerr << "Valid options for scene: " << argv[0] << '\n';
    return 0;
  }
  
  Point Eye(0,0,-5);
  Point Lookat(0,0,0);
  Vector Up(0,1,0);
  double fov=60;
 

  double bgscale=0.5;
  Color groundcolor(.82, .62, .62);
  Color averagelight(1,1,.8);
  double ambient_scale=.5;
  
  Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);
  Material* bookcoverimg = new ImageMaterial("i3d97.smaller.gamma",
					     ImageMaterial::Clamp,
                                             ImageMaterial::Clamp,
					     Color(0,0,0), 1,
                                             Color(0,0,0), 0);


  Point p(-1,-1,0);
  Vector v1(-.774,0,0);
  Vector v2(0,-1,0);
  Parallelogram *r = new Parallelogram(bookcoverimg,p,v2,v1);
  Group *g = new Group();

  g->add(r);

  Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 10) );
  Camera cam(Eye,Lookat,Up,fov);
  Scene* scene=new Scene(g, cam,
			 bgcolor, groundcolor*averagelight, bgcolor, groundplane,
			 ambient_scale);
  scene->add_light(new Light(Point(5,-3,3), Color(1,1,.8)*2, 0));
  scene->ambient_hack = true;
  
  scene->shadow_mode=1;
  return scene;
}
  

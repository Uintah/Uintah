#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Scene.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Parallelogram.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/MIPMaterial.h>
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
//    Material* bookcoverimg = new ImageMaterial(1,
//  					     "/usr/sci/data/rtrt/Geometry/textures/i3d97.smaller.gamma",
//  					     ImageMaterial::Clamp,
//                                               ImageMaterial::Clamp,
//  					     1,
//                                               Color(0,0,0), 0);

  Material* bookcoverimg = 
    new MIPMaterial("/opt/SCIRun/data/Geometry/textures/i3d97.smaller.gamma.ppm",
                    .7, Color(.1,.1,.1),30,0,1);
  Material* bookcoverimg1 = 
    new ImageMaterial("/opt/SCIRun/data/Geometry/textures/i3d97.smaller.gamma.ppm",
                      ImageMaterial::Clamp,
                      ImageMaterial::Clamp,
                      .7,Color(.1,.1,.1), 30,
                      0,0);


  Point p(-1,-1,0);
  Point p1(-1,-2,0);
  Vector v1(-.774,0,0);
  Vector v2(0,-1,0);
  Parallelogram *r = new Parallelogram(bookcoverimg,p,v2,v1);
  Parallelogram *r1 = new Parallelogram(bookcoverimg1,p1,v2,v1);
  Group *g = new Group();

  g->add(r);
  g->add(r1);

  Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 10) );
  Camera cam(Eye,Lookat,Up,fov);
  Scene* scene=new Scene(g, cam,
			 bgcolor, groundcolor*averagelight, bgcolor, 
			 groundplane, ambient_scale, Arc_Ambient);
  scene->add_light(new Light(Point(5,-3,3), Color(1,1,.8)*2, 0));
  
  scene->select_shadow_mode( Hard_Shadows );
  return scene;
}
  

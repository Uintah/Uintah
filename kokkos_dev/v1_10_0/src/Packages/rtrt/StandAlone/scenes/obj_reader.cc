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
#include <Packages/rtrt/Core/ObjReader.h>
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

extern "C"
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
  Point Eye(-10.9055, -0.629515, 1.56536);
  Point Lookat(-8.07587, 15.7687, 1.56536);
  Vector Up(0, 0, 1);
  double fov=60;

  Camera cam(Eye,Lookat,Up,fov);

  if( argc != 2 )
    {
      cout << "Usage: " << argv[0] << " objfile (without extension)\n";
      exit(1);
    }

  string objname = argv[1];

  Group     * g = new Group();

  Transform   t;

  if (!readObjFile( objname + ".obj", objname + ".mtl", t, g)) {
    cout << "Error reading file\n";
    exit(0);
  }

  Color cdown(0.1, 0.1, 0.1);
  Color cup(0.1, 0.1, 0.1);

  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(0.3, 0.3, 0.3);
  double ambient_scale=1.0;

  Scene *scene = new Scene(g, cam, bgcolor, cdown, cup, groundplane, 
			   ambient_scale, Arc_Ambient);
  BBox b;
  g->compute_bounds(b, 0.001);
  scene->select_shadow_mode( No_Shadows );
  scene->maxdepth = 4;
  Light *l = new Light(b.max()+b.diagonal(),Color(1.0,1.0,1.0), 0);
  l->name_ = "Light0";
  scene->add_light(l);
  scene->animate=false;
  return scene;
}

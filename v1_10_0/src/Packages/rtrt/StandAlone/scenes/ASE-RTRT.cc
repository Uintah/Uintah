#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Array3.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/TexturedTri.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Plane.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/ASEReader.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/CycleMaterial.h>
#include <Packages/rtrt/Core/InvisibleMaterial.h>
#include <Packages/rtrt/Core/Rect.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>

using namespace SCIRun;
using namespace rtrt;
using namespace std;

extern "C" Scene *make_scene(int argc, char** argv, int)
{
  if (argc < 2) {
    cerr << endl << "usage: rtrt ... -scene ASE-RTRT <ase filename>" << endl;
    return 0;
  }

  Array1<Material*> ase_matls;
  string env_map;

  Transform t;
  t.load_identity();
  Group *all = new Group;
  if (!readASEFile(argv[1], t, all, ase_matls, env_map)) return 0;

  Camera cam(Point(1,0,0), Point(0,0,0),
             Vector(0,0,1), 40);
  
  Color groundcolor(.7,.6,.5);
  double ambient_scale=.3;
  
  Color bgcolor(.2,.2,.4);
  
  rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, 1, 0) );
  Scene* scene=new Scene(all, cam, bgcolor, 
			 Color(1,0,0), Color(0,0,01),
			 groundplane, ambient_scale, Arc_Ambient);

  scene->add_light(new Light(Point(-6250,-11800,15000), Color(1,1,1), 1));
  if (env_map!="")
    scene->set_background_ptr(new EnvironmentMapBackground((char*)env_map.c_str()));
  scene->select_shadow_mode( No_Shadows );
  scene->set_materials(ase_matls);
  return scene;
}

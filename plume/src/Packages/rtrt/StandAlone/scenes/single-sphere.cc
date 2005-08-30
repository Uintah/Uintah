//
// This file contains a simple scene suitable for ray tracing
// on 1 processor.
//
// It contains one sphere and a "ground" and a ring.
//

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Ring.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/Checker.h>
#include <iostream>
#include <math.h>
#include <string.h>

using namespace rtrt;

extern "C" 
Scene* make_scene(int /*argc*/, char* /*argv*/[], int /*nworkers*/)
{
  Camera cam( Point(0,30,0), Point( 0,0,0 ), Vector(0,0,1), 45.0 );

  Material* white=new LambertianMaterial( Color( 1,1,1 ) );
//  white->local_ambient_mode = Sphere_Ambient;
  Object* sphere = new Sphere( white, Point(0,0,0), 1 );

  Group * group = new Group();
  group->add( sphere );

  double ambient_scale=1.0;
  Color bgcolor(0.3, 0.3, 0.3);
  Color cdown(0.6, 0.4, 0.4);
  Color cup(0.4, 0.4, 0.6);

  rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 1) );
  Scene* scene=new Scene(group, cam,
			 bgcolor, cdown, cup, groundplane,
			 ambient_scale, Sphere_Ambient);

  EnvironmentMapBackground *emap = 
    new EnvironmentMapBackground("/home/sci/dmw/rtrt/rgb-envmap.ppm",
//    new EnvironmentMapBackground("/home/sci/dmw/sr/rtrt/SCIRun/sgi64opt/Packages/rtrt/StandAlone/utils/top.ppm",
				 Vector(0,0,1));

  scene->set_ambient_environment_map(emap);
  return scene;
}


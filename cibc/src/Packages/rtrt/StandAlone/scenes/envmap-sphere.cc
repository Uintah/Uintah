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
  Camera cam( Point(0,3,0), Point( 0,0,0 ), Vector(0,0,1), 45.0 );

  Material* silver=new MetalMaterial( Color( 1,1,1 ), 30 );

  Object* sphere = new Sphere( silver, Point(0,0,0), 1 );

  Group * group = new Group();
  group->add( sphere );

  double ambient_scale=1.0;
  Color bgcolor(0.3, 0.3, 0.3);
  Color cdown(0.6, 0.4, 0.4);
  Color cup(0.4, 0.4, 0.6);

  rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 1) );
  Scene* scene=new Scene(group, cam,
			 bgcolor, cdown, cup, groundplane,
			 ambient_scale, Arc_Ambient);

  Light * red = new Light(Point(100, 100, 100), Color(.9,.8,.8), 0);
  red->name_ = "red";
  scene->add_light( red );
  Light * green = new Light(Point(100, -100, 100), Color(.8,.9,.8), 0);
  green->name_ = "green";
  scene->add_light( green );
  Light * blue = new Light(Point(-100, 0, 100), Color(.8,.8,.9), 0);
  blue->name_ = "blue";
  scene->add_light( blue );
  Light * purple = new Light(Point(0, 0, -100), Color(.9,.8,.9), 0);
  purple->name_ = "purple";
  scene->add_light( purple );

  scene->select_shadow_mode( Hard_Shadows );
  return scene;
}


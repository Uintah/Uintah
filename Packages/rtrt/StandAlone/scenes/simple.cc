//
// This file contains a simple scene suitable for ray tracing
// on 1 processor.
//
// It contains one sphere and a "ground".
//

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/Checker.h>
#include <iostream>
#include <math.h>
#include <string.h>

using namespace rtrt;

extern "C" 
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
  Camera cam( Point(0,0,0), Point( 0,-1,0 ), Vector(0,0,1), 45.0 );

  Material* matl=new MetalMaterial( Color( .9,.1,.4 ) );
  Material* matl2=new MetalMaterial( Color( .1,.9,.1 ) );

  Object* obj  = new Sphere( matl, Point(0,-10,0), 1 );
  Object* obj1 = new Rect(matl2, Point(0,0,0), Vector(6,0,0), Vector(0,6,0));


  Group * group = new Group();
  group->add( obj );
  group->add( obj1 );

  double ambient_scale=1.0;
  Color bgcolor(0.1, 0.2, 0.45);
  Color cdown(0.82, 0.62, 0.62);
  Color cup(0.1, 0.3, 0.8);

  rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 1) );
  Scene* scene=new Scene(group, cam,
			 bgcolor, cdown, cup, groundplane,
			 ambient_scale);
  scene->add_light( new Light(Point(20,20,50), Color(1,1,1), 0.8) );
  scene->ambient_hack = true;

  scene->set_background_ptr( new LinearBackground( Color(1.0, 1.0, 1.0),
						   Color(0.0,0.0,0.0),
						   Vector(0,0,1)) );
  scene->shadow_mode=1;
  return scene;
}


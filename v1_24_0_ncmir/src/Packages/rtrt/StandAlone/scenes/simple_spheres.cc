//
// This file contains 10 spheres to drive around and a ground.
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
Scene* make_scene(int /*argc*/, char* /*argv*/[], int /*nworkers*/)
{
  Camera cam( Point(10,-10,0), Point( 10,0,0 ), Vector(0,0,1), 45.0 );

  Material* matl=new MetalMaterial( Color( .9,.1,.4 ) );
  Material* matl2=new MetalMaterial( Color( .1,.9,.1 ) );

  Material* lammat=new LambertianMaterial( Color( 0.0,1.0,1.0 ) );

  Group * group = new Group();

  for( int i = 0; i < 3; i++ )
    for( int j = 0; j < 3; j++ )
      {
	Material * matlp;
	if( j == 1 )
	  matlp = lammat;
	else
	  matlp = matl;
	Object* obj  = new Sphere( matlp, Point(i*10,j*10,0), 1 );
	group->add( obj );
      }


  Object* obj1 = new Rect(matl2, Point(0,0,0), Vector(20,0,0), Vector(0,20,0));

  group->add( obj1 );

  double ambient_scale=1.0;
  Color bgcolor(0.0, 0.0, 0.0);
  Color cdown(1.0, 1.0, 1.0);
  Color cup(1.0, 1.0, 1.0);

  rtrt::Plane groundplane ( Point(0, 0, -0.5), Vector(0, 0, 1) );
  Scene* scene=new Scene(group, cam,
			 bgcolor, cdown, cup, groundplane,
			 ambient_scale, Arc_Ambient);
  Light * mainLight = new Light(Point(20,20,50), Color(1,1,1), 0.8, 1.0 );
  mainLight->name_ = "main light";

  Light * botLight = new Light(Point(0,0,-50), Color(1,1,0), 0.8, 1.0 );
  botLight->name_ = "bottom light";

  scene->add_light( mainLight );
  scene->add_light( botLight );
  
  scene->set_background_ptr( new LinearBackground( Color(0.0, 0.0, 0.0),
						   Color(1.0,0.0,1.0),
						   Vector(0,0,1)) );
  scene->select_shadow_mode( Hard_Shadows );
  return scene;
}


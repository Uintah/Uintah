//
// This file contains a simple scene suitable for ray tracing
// on 1 processor.
//
// It contains one sphere and a "ground" and a ring.
//

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Ring.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/Checker.h>
#include <Packages/rtrt/Core/Box.h>
#include <Packages/rtrt/Core/SpinningInstance.h>
#include <Packages/rtrt/Core/DynamicInstance.h>
#ifdef HAVE_SOUND
#  include <Packages/rtrt/Sound/Sound.h>
#endif
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>

#include <iostream>
#include <vector>
#include <math.h>
#include <string.h>

using namespace rtrt;
using std::vector;

extern "C" 
Scene* make_scene(int /*argc*/, char* /*argv*/[], int /*nworkers*/)
{
  Camera cam( Point(1.9,22,7), Point( 1.9,-9.6,7 ), Vector(0,0,1), 45.0 );

  Group * group = new Group();


  Material* matl=new MetalMaterial( Color( .9,.1,.4 ) );
  Object* obj  = new Sphere( matl, Point(0,-10,0), 1 );
  group->add( obj );

#if 0
  Material* matl2=new MetalMaterial( Color( .1,.9,.1 ) );
  Object* obj1 = new Rect(matl2, Point(0,0,0), Vector(6,0,0), Vector(0,6,0));
  group->add( obj1 );

  Material* matl3=new MetalMaterial( Color( .1,.9,.9 ) );
  Object* obj2 = new Ring(matl3, Point(0, -8, 1), Vector(0,0,1), 5, 1);
  group->add( obj2 );
#endif

  ImageMaterial * painting = 
    new ImageMaterial( "/usr/sci/projects/rtrt/paintings/delaware.ppm",
		       ImageMaterial::Clamp, ImageMaterial::Clamp,
		       1, Color(0,0,0), 0 );

  Object * picture = new Rect( painting,
			       Point(5,5,5), Vector(0,1,0), Vector(0,0,-1) );


  group->add( picture );

  Material* red =new LambertianMaterial( Color( .9,.2,.2 ) );
  obj  = new Sphere( red, Point(5, 5, 6), 0.3 );
  group->add( obj );

  Transform *vtrans = new Transform();

  //Vector location(2,2,2);
  Vector location(0,0,0);

  Object * box = new Box( red, Point(-1,-1,-1), Point(1,1,1) );

  Object * objOfInterest = 
    new DynamicInstance( new InstanceWrapperObject( box ), vtrans, location );
  //Object * objOfInterest = box;
  group->add( objOfInterest );

  double ambient_scale=1.0;
  Color bgcolor(0.1, 0.2, 0.45);
  Color cdown(0.82, 0.62, 0.62);
  Color cup(0.1, 0.3, 0.8);

  rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 1) );
  Scene* scene=new Scene(group, cam,
			 bgcolor, cdown, cup, groundplane,
			 ambient_scale, Arc_Ambient);

#ifdef HAVE_SOUND
  vector<Point> loc; loc.push_back(Point(5,5,5));
  Sound * sound = new Sound( "cool_music.wav", "cool", loc, 10, true );
  scene->addSound( sound );
  loc.clear(); loc.push_back( Point( 0,0,0 ) );
  sound = new Sound( "water-flowing1.wav", "water", loc, 10, true );
  scene->addSound( sound );
#endif

  scene->addObjectOfInterest( "cube", objOfInterest, true );

  Light * light = new Light(Point(20,20,50), Color(1,1,1), 0.8);
  light->name_ = "main light";
  scene->add_light( light );

  light = new Light(Point(20,5,5), Color(1,1,1), 0.5);
  light->name_ = "picture light";
  scene->add_light( light );

  light = new Light(Point(4.9,4.9,4.9), Color(1,1,1), 0.5);
  light->name_ = "in front of pict";
  scene->add_light( light );

  scene->set_background_ptr( new LinearBackground( Color(1.0, 1.0, 1.0),
						   Color(0.0,0.0,0.0),
						   Vector(0,0,1)) );
  scene->select_shadow_mode( Hard_Shadows );
  return scene;
}


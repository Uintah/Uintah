#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Disc.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Point4D.h>
#include <Packages/rtrt/Core/CrowMarble.h>
#include <Packages/rtrt/Core/Mesh.h>
#include <Packages/rtrt/Core/Bezier.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/Checker.h>
#include <Packages/rtrt/Core/Speckle.h>
#include <Packages/rtrt/Core/Box.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/Parallelogram.h>
#include <Packages/rtrt/Core/ObjReader.h>
#include <Packages/rtrt/Core/SpinningInstance.h>
#if !defined(linux)
#  include <Packages/rtrt/Sound/Sound.h>
#endif

#include <Core/Math/MinMax.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>

#include <iostream>
#include <math.h>
#include <string.h>

using namespace rtrt;

extern "C"
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
  Point center(8, -8, 0);  

  Point eye = center + Vector(0,0,60);
  Point lookat = center + Vector(1,1,60);
  Vector up(0,0,1);

  double fov=60;

  Group * group = new Group();

  Transform room_trans;
  room_trans.pre_translate( center.vector() );
  room_trans.pre_scale( Vector(0.02,0.02,0.02) );

  Camera cam(room_trans.project(eye),room_trans.project(lookat),up,fov);

  string pathname("/usr/sci/data/Geometry/models/livingroom/livingroom-obj2_fullpaths/");
  Array1<string> names;

  cout << "argc is " << argc << "\n";
  if( argc < 2 ) {
    cout << "Adding all furniture\n";

    names.add("coffee-table");
    names.add("shelf_pictures");
    names.add("fruit-bowl");

    names.add("mirror1");
    names.add("mirror2");
    names.add("horse");
    names.add("venus-demilo");
    names.add("bookends");
    names.add("books");
    names.add("camera");
    names.add("ceiling_light");
    names.add("chair1");
    names.add("chair2");
    names.add("chess-set");
    names.add("corella");
    names.add("corella2");
    names.add("couch");
    names.add("door_jams");
    names.add("end-table1");
    names.add("end-table2");
    names.add("fern");
    names.add("glasses");
    names.add("glasses_lenses");
    names.add("grand_piano");
    names.add("greek_bowl");
    names.add("guitar");
    names.add("horse_bookends");
    names.add("orrey");
    names.add("paintings");
    names.add("phone");
    names.add("pipe");
    names.add("shell1");
    names.add("shell2");
    names.add("tea-set");
    names.add("wall-sconce1");
    names.add("wall-sconce2");
    names.add("wine_table");
    names.add("globe_base");
  }

  for (int i=0; i < names.size(); i++) {
    cerr << "Reading: " << pathname + names[i] << "\n";
    string objname( pathname + names[i]+ ".obj" );
    string mtlname( pathname + names[i]+ ".mtl" );
    if (!readObjFile(objname, mtlname, room_trans, group))
      exit(0);
  }

  names.remove_all();
  names.add("livingroom");

  for (int i=0; i < names.size(); i++) {
    cerr << "Reading: " << pathname + names[i] << "\n";
    string objname( pathname + names[i]+ ".obj" );
    string mtlname( pathname + names[i]+ ".mtl" );
    if (!readObjFile(objname, mtlname, room_trans, group))
      exit(0);
  }

  if( argc < 2 ) {
    names.remove_all();
    names.add("clock");

    for (int i=0; i < names.size(); i++) {
      cerr << "Reading: " << pathname + names[i] << "\n";
      string objname( pathname + names[i]+ ".obj" );
      string mtlname( pathname + names[i]+ ".mtl" );
      if (!readObjFile(objname, mtlname, room_trans, group, 30))
	exit(0);
    }

    names.remove_all();
    names.add("wine_glasses");
    names.add("wine_bottle");

    for (int i=0; i < names.size(); i++) {
      cerr << "Reading: " << pathname + names[i] << "\n";
      string objname( pathname + names[i]+ ".obj" );
      string mtlname( pathname + names[i]+ ".mtl" );
      if (!readObjFile(objname, mtlname, room_trans, group, 200))
	exit(0);
    }
  }

  Color cdown(0.1, 0.1, 0.7);
  Color cup(0.5, 0.5, 0.0);

  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(.9, 0.9, 0.9);
  Scene *scene = new Scene(group, cam, bgcolor, cdown, cup, groundplane, 0.3);

  Transform globetrans;
  Group * globegroup = new Group;
  string name = pathname + "globe_sphere";
  if( !readObjFile( name + ".obj", name + ".mtl", room_trans, globegroup ) )
    {
      cout << "Error reading: " << name << "\n";
    }
  Grid * globegrid = new Grid( globegroup, 30 );
  SpinningInstance * si = 
    new SpinningInstance(
	    new InstanceWrapperObject(globegrid), &globetrans, 
	    room_trans.project(Point(167.633,-8.34035,32.7797)),
	    Vector(-1.32,-46.686,-100.118), -0.1 );

  group->add( si );
  scene->addObjectOfInterest( si, true );

#if !defined(linux)
  string path = "/home/sci/dav/sounds/";
  vector<Point> loc; loc.push_back(Point(82,90,59)); // piano
  Sound * sound = new Sound( path+"player-piano-cd026_73.wav", "piano", loc, 150, true );
  scene->addSound( sound );
  loc.clear();

  loc.push_back(Point(0,0,0));  // harp back ground
  sound = new Sound( path+"harp-melodic-cd027_59.wav", "harp", loc, 150, true, .5 );
  scene->addSound( sound );
  loc.clear();

  loc.push_back(Point(-152,23,50));  // clock
  sound = new Sound( path+"ticking-clock-cd058_07.wav", "ticking", loc, 150, true );
  scene->addSound( sound );
  loc.clear();

  loc.push_back(Point(-162,23,50));  // clock
  sound = new Sound( path+"clock-tower-bells-cd025_75.wav", "chime", loc, 150, true );
  scene->addSound( sound );
  loc.clear();

  loc.push_back(Point(-8,-91,31)); // fruit
  sound = new Sound( path+"music-box-cd074_96.wav", "music-box", loc, 150, true );
  scene->addSound( sound );
  loc.clear();

  loc.push_back(Point(-121,453,62)); // outside
  sound = new Sound( path+"waves-ocean-shoreline-cd039_27.wav", "under water", loc, 200, true );
  scene->addSound( sound );
  loc.clear();

  loc.push_back(Point(154,-84,110)); // books
  sound = new Sound( path+"cool_music.wav", "cool", loc, 150, true );
  scene->addSound( sound );

#endif

  double lightBrightness = 0.5;

  scene->select_shadow_mode( Hard_Shadows );
  scene->maxdepth = 4;
  Light * light = new Light( Point(0.15,0.15,2.8), 
			     Color(1.0,1.0,1.0), 0, lightBrightness );
  light->name_ = "Overhead";
  scene->add_light( light );
  
  light = new Light( Point(-1.48,-3.76,1.68),
		     Color(1.0,1.0,1.0), 0, lightBrightness );
  light->name_ = "Right Sconce";
  scene->add_light( light );

  light = new Light( Point(1.0,-3.69,1.78),
		     Color(1.0,1.0,1.0), 0, lightBrightness );
  light->name_ = "Left Sconce";
  scene->add_light( light );

  light = new Light( Point(3.498,-3.432,2.7),
		     Color(1.0,1.0,1.0), 0, lightBrightness );
  light->name_ = "Horse";
  scene->add_light( light );

  light = new Light( Point(-3.281,-3.66,2.7),
		     Color(1.0,1.0,1.0), 0, lightBrightness );
  light->name_ = "Venus";
  scene->add_light( light );

  light = new Light( Point(32.81,16,5.96),
		     Color(1.0,1.0,1.0), 0, lightBrightness );
  light->name_ = "Sun";
  scene->add_light( light );

  scene->animate=true;

  return scene;
}

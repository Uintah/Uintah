#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/HierarchicalGrid.h>
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
#include <Packages/rtrt/Core/HierarchicalGrid.h>
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
Scene* make_scene(int argc, char* /*argv[]*/, int /*nworkers*/)
{
  string name;

  string objname;
  string mtlname;

  int    minObjs1 = 10;
  int    minObjs2 = 10;
  HierarchicalGrid * hg;

  Point center(12.5, -8.5, 0);  

  Point eye = center + Vector(0,0,60);
  Point lookat = center + Vector(-1,-1,60);
  Vector up(0,0,1);

  double fov=60;

  Group * group = new Group();

  Transform room_trans;
  room_trans.pre_scale( Vector(0.025,0.025,0.025) );
  room_trans.pre_translate( center.vector() );

  cout << "argc is " << argc << "\n";

  Camera cam(room_trans.project(eye),room_trans.project(lookat),up,fov);

  string pathname("/opt/SCIRun/data/Geometry/models/livingroom/livingroom-obj2_fullpaths/");

  double lightBrightness = 0.4;

  Light * overhead = new Light( room_trans.project(Point(7.3, 18.3, 144.5)),
				Color(1.0,1.0,1.0), 1, lightBrightness );
  overhead->name_ = "Overhead";
  
  lightBrightness = 0.3;

  Light * rSconce = new Light( room_trans.project(Point(-81.9,-172.9,85.1)),
			       Color(1.0,1.0,1.0), 0, lightBrightness );
  rSconce->name_ = "Right Sconce";
  rSconce->turnOff();

  Light * lSconce = new Light( room_trans.project(Point(40.7,-179.5,85.1)),
			       Color(1.0,1.0,1.0), 0, lightBrightness );
  lSconce->name_ = "Left Sconce";
  lSconce->turnOff();

  lightBrightness = 0.7;
  Light * horselight = 
    new Light( room_trans.project(Point(146.91,-142.5,142.0)),
	       Color(1.0,1.0,1.0), 0, lightBrightness );
  horselight->name_ = "Horse";
  horselight->turnOff();

  Light * venuslight = 
    new Light( room_trans.project(Point(-143.2,-144.5,142.0)),
	       Color(1.0,1.0,1.0), 0, lightBrightness );
  venuslight->name_ = "Venus";
  venuslight->turnOff();

  lightBrightness = 0.0;
  Light * sunlight = new Light( room_trans.project(Point(4206,2059,740)),
				Color(1.0,1.0,1.0), 0, lightBrightness );
  sunlight->name_ = "Sun";
  sunlight->turnOff();

  Array1<string>    names;
  Array1<Material*> mtls;

  //////////////// LIVING ROOM ARCHITECTURE ////////////////////////
  cout << "reading main living room architecture\n";
  Group * livGrp = new Group;
  name = "livingroom";
  objname = pathname + name + ".obj";
  mtlname = pathname + name + ".mtl";
  if (!readObjFile(objname, mtlname, room_trans, mtls, livGrp)) exit(0);
  hg = new HierarchicalGrid( livGrp, 6, 8, 10,
			     minObjs1, minObjs2, 4 );
  group->add( hg );

  for( int cnt = 0; cnt < mtls.size(); cnt++ )
    {
      mtls[cnt]->my_lights.add(sunlight);
      mtls[cnt]->my_lights.add(horselight);
      mtls[cnt]->my_lights.add(venuslight);
      mtls[cnt]->my_lights.add(overhead);
      mtls[cnt]->my_lights.add(rSconce);
      mtls[cnt]->my_lights.add(lSconce);
    }

  if( argc < 2 ) {
    cout << "Adding all furniture\n";

    names.add("wine_table");
    names.add("globe_base");
    names.add("mirror1");
    names.add("mirror2");

    names.add("coffee-table");
    names.add("shelf_pictures");
    names.add("fruit-bowl");

    names.add("bookends");
    names.add("books");
    names.add("camera");
    names.add("ceiling_light");
    names.add("chess-set");
    names.add("corella");
    names.add("corella2");
    names.add("door_jams");
    names.add("end-table1");
    names.add("end-table2");
    names.add("greek_bowl");
    names.add("guitar");
    names.add("orrey");
    names.add("phone");
    names.add("pipe");
    names.add("shell1");
    names.add("shell2");
    names.add("tea-set");
    names.add("wall-sconce1");
    names.add("wall-sconce2");
  } // end if not min livingroom

  // Grid 32 objects:
  for (int i=0; i < names.size(); i++) {
    cerr << "Reading: " << pathname + names[i] << "\n";
    string objname( pathname + names[i]+ ".obj" );
    string mtlname( pathname + names[i]+ ".mtl" );
    if (!readObjFile(objname, mtlname, room_trans, mtls, group, 32))
      exit(0);
    for( int cnt = 0; cnt < mtls.size(); cnt++ )
      {
	mtls[cnt]->my_lights.add(overhead);
      }
  }
 
  if( argc < 2 ) { // Non-Min Living Room

    // Grid 64 objects:
    names.remove_all();
    names.add("chair1");
    names.add("chair2");
    names.add("couch");
    for (int i=0; i < names.size(); i++) {
      cerr << "Reading: " << pathname + names[i] << "\n";
      string objname( pathname + names[i]+ ".obj" );
      string mtlname( pathname + names[i]+ ".mtl" );
      if (!readObjFile(objname, mtlname, room_trans, mtls, group, 64))
	exit(0);
      for( int cnt = 0; cnt < mtls.size(); cnt++ )
	{
	  mtls[cnt]->my_lights.add(sunlight);
	  mtls[cnt]->my_lights.add(overhead);
	}
    }
    // Venus
    name = "venus-demilo";
    cerr << "Reading: " << pathname + name<< "\n";
    objname = pathname + name + ".obj";
    mtlname = pathname + name + ".mtl";
    if (!readObjFile(objname, mtlname, room_trans, mtls, group, 64))
      exit(0);

    ////  Lights for Venus
    for( int cnt = 0; cnt < mtls.size(); cnt++ )
      {
	mtls[cnt]->my_lights.add(sunlight);
	mtls[cnt]->my_lights.add(venuslight);
	mtls[cnt]->my_lights.add(overhead);
      }

    // Paintings
    Group * paintingsGrp = new Group;
    name = "paintings";  // hgrid 3 4 5
    objname = pathname + name + ".obj";
    mtlname = pathname + name + ".mtl";
    cerr << "Reading: " << pathname + name << "\n";
    if (!readObjFile(objname, mtlname, room_trans, mtls, paintingsGrp)) exit(0);
    hg = new HierarchicalGrid( paintingsGrp, 3, 4, 5,
			       minObjs1, minObjs2, 3 );
    group->add( hg );
    ////  Lights for Paintings
    for( int cnt = 0; cnt < mtls.size(); cnt++ )
      {
	mtls[cnt]->my_lights.add(overhead);
      }

    // Horse
    Group * horseGrp = new Group;
    name = "horse";  // hgrid 3 4 5
    objname = pathname + name + ".obj";
    mtlname = pathname + name + ".mtl";
    cerr << "Reading: " << pathname + name << "\n";
    if (!readObjFile(objname, mtlname, room_trans, mtls, horseGrp)) exit(0);
    hg = new HierarchicalGrid( horseGrp, 3, 4, 5,
			       minObjs1, minObjs2, 3 );
    group->add( hg );
    ////  Lights for Horse
    for( int cnt = 0; cnt < mtls.size(); cnt++ )
      {
	mtls[cnt]->my_lights.add(horselight);
	mtls[cnt]->my_lights.add(overhead);
      }

  } // end non min-living room


  cout << "reading specially individually gridded stuff\n";

  if( argc < 2 ) { // Non-Min Living Room

#if 1
    for( int cnt = 1; cnt <= 1; cnt++ ) {
      char horsename[128];
      sprintf( horsename,"horse_bookend%d",cnt );
      name = horsename;
      objname = pathname + name + ".obj";
      mtlname = pathname + "horse_bookends.mtl";
      if (!readObjFile(objname, mtlname, room_trans,mtls,group,32)) exit(0);
      for( int cnt = 0; cnt < mtls.size(); cnt++ )
	{
	  mtls[cnt]->my_lights.add(overhead);
	}
    }
#endif

    //////////////// WINE BOTTLE AND GLASSES ////////////////////////
    name = "wine_glasses"; // grid 64
    objname = pathname + name + ".obj";
    mtlname = pathname + name + ".mtl";
    if(!readObjFile(objname, mtlname, room_trans, group, 64)) exit(0);
    name = "wine_bottle"; // grid 32
    objname = pathname + name + ".obj";
    mtlname = pathname + name + ".mtl";
    if(!readObjFile(objname, mtlname, room_trans, mtls, group, 32)) exit(0);
    for( int cnt = 0; cnt < mtls.size(); cnt++ )
      {
	mtls[cnt]->my_lights.add(overhead);
      }

    //////////////// FERN /////////////////////////
    Group * fernGrp = new Group;
    name = "fern";  // hgrid 4 6 8
    objname = pathname + name + ".obj";
    mtlname = pathname + name + ".mtl";
    if (!readObjFile(objname, mtlname, room_trans, mtls, fernGrp)) exit(0);
    hg = new HierarchicalGrid( fernGrp, 3, 5, 7,
			       minObjs1, minObjs2, 3 );
    group->add( hg );
    for( int cnt = 0; cnt < mtls.size(); cnt++ )
      {
	mtls[cnt]->my_lights.add(overhead);
      }

    //////////////// CLOCK ////////////////////////
    Group * clockGrp = new Group;
    name = "clock";  // hgrid 4 6 8
    objname = pathname + name + ".obj";
    mtlname = pathname + name + ".mtl";
    if (!readObjFile(objname, mtlname, room_trans, mtls, clockGrp)) exit(0);
    hg = new HierarchicalGrid( clockGrp, 4, 6, 8,
			       minObjs1, minObjs2, 3 );
    group->add( hg );
    for( int cnt = 0; cnt < mtls.size(); cnt++ )
      {
	mtls[cnt]->my_lights.add(overhead);
      }

    //////////////// PIANO ////////////////////////
    Group * pianoGrp = new Group;
    name = "grand_piano";  // hgrid 4 6 8
    objname = pathname + name + ".obj";
    mtlname = pathname + name + ".mtl";
    if (!readObjFile(objname, mtlname, room_trans, mtls, pianoGrp)) exit(0);
    hg = new HierarchicalGrid( pianoGrp, 4, 6, 8,
			       minObjs1, minObjs2, 3 );
    group->add( hg );
    for( int cnt = 0; cnt < mtls.size(); cnt++ )
      {
	mtls[cnt]->my_lights.add(overhead);
      }

    //////////////// EYE GLASSES ////////////////////////
    Group * glassesGrp = new Group;
    name = "glasses";  // hgrid 3 7 9
    objname = pathname + name + ".obj";
    mtlname = pathname + name + ".mtl";
    if (!readObjFile(objname, mtlname, room_trans, mtls, glassesGrp)) exit(0);
    hg = new HierarchicalGrid( glassesGrp, 3, 7, 9,
			       minObjs1, minObjs2, 3 );
    group->add( hg );
    for( int cnt = 0; cnt < mtls.size(); cnt++ )
      {
	mtls[cnt]->my_lights.add(overhead);
      }
    name = "glasses_lenses"; // grid 32
    objname = pathname + name + ".obj";
    mtlname = pathname + name + ".mtl";
    if(!readObjFile(objname, mtlname, room_trans, mtls, group, 32)) exit(0);
    for( int cnt = 0; cnt < mtls.size(); cnt++ )
      {
	mtls[cnt]->my_lights.add(overhead);
      }
  }

  Color cdown(0.1, 0.1, 0.7);
  Color cup(0.5, 0.5, 0.0);

  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(.9, 0.9, 0.9);
  Scene *scene = new Scene(new HierarchicalGrid(group, 8, 8, 8, 10, 10, 4), cam, bgcolor, cdown, cup, groundplane, 0.3);

  scene->select_shadow_mode( Hard_Shadows );
  scene->maxdepth = 8;

  scene->add_per_matl_light( overhead );
  scene->add_per_matl_light( rSconce );
  scene->add_per_matl_light( lSconce );
  scene->add_per_matl_light( horselight );
  scene->add_per_matl_light( venuslight );
  scene->add_per_matl_light( sunlight );

  //////////////// GLOBE INSTANCE ////////////////////////
  Transform globetrans;
  Group * globegroup = new Group;
  name = pathname + "globe_sphere";
  if( !readObjFile( name + ".obj", name + ".mtl",room_trans,mtls,globegroup ) )
    {
      cout << "Error reading: " << name << "\n";
    }
  for( int cnt = 0; cnt < mtls.size(); cnt++ )
    {
      mtls[cnt]->my_lights.add(sunlight);
      mtls[cnt]->my_lights.add(overhead);
    }
  Grid * globegrid = new Grid( globegroup, 64 );
  SpinningInstance * si = 
    new SpinningInstance(
	    new InstanceWrapperObject(globegrid), &globetrans, 
	    room_trans.project(Point(167.633,-8.34035,32.7797)),
	    Vector(-1.32,-46.686,-100.118), -0.1 );

  group->add( si );
  scene->addObjectOfInterest( si, true );

#if !defined(linux)
  string path = "/home/sci/dav/sounds/"; // piano
  vector<Point> loc; loc.push_back(room_trans.project(Point(53,82,64)));
  Sound * sound = new Sound( path+"player-piano-cd026_73.wav", "piano", 
			     loc, 3, true );
  scene->addSound( sound );
  loc.clear();

  loc.push_back(room_trans.project(Point(0,0,0))); // harp back ground
  sound = new Sound( path+"harp-melodic-cd027_59.wav", "harp", loc,
		     5, true, .5 );
  scene->addSound( sound );
  loc.clear();

  loc.push_back(room_trans.project(Point(-161,-121,67)));  // clock
  sound = new Sound( path+"ticking-clock-cd058_07.wav", "ticking", 
		     loc, 2, true );
  scene->addSound( sound );
  loc.clear();

  loc.push_back(room_trans.project(Point(-161,-121,67)));  // clock
  sound = new Sound( path+"clock-tower-bells-cd025_75.wav", "chime",
		     loc, 2, true );
  scene->addSound( sound );
  loc.clear();

  loc.push_back(room_trans.project(Point(-8,-91,31))); // fruit
  sound = new Sound( path+"music-box-cd074_96.wav", "music-box",
		     loc, 2, true );
  scene->addSound( sound );
  loc.clear();

  loc.push_back(room_trans.project(Point(-121,453,62))); // outside
  sound = new Sound( path+"waves-ocean-shoreline-cd039_27.wav",
		     "under water", loc, 4, true );
  scene->addSound( sound );
  loc.clear();

  loc.push_back(room_trans.project(Point(135,-126,60))); // books
  sound = new Sound( path+"cool_music.wav", "cool", loc, 2, true );
  scene->addSound( sound );

  //loc.push_back(room_trans.project(Point(-160,61,64))); // wine
  //sound = new Sound( path+"cool_music.wav", "cool", loc, 2, true );
  //scene->addSound( sound );

  //loc.push_back(room_trans.project(Point(-98,116,37))); // phone
  //sound = new Sound( path+"cool_music.wav", "cool", loc, 1, true );
  //scene->addSound( sound );

  //loc.push_back(room_trans.project(Point(178,36,37))); // banjo
  //sound = new Sound( path+"cool_music.wav", "cool", loc, 1, true );
  //scene->addSound( sound );

#endif

  scene->animate=true;


  return scene;
}

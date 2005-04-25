#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Disc.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/PPMImage.h>
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
#  include <Packages/rtrt/Core/Trigger.h>
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
  string name;

  string objname;
  string mtlname;

  int    minObjs1 = 10;
  int    minObjs2 = 10;
  HierarchicalGrid * hg;

  Point center(12.5, -8.5, 0);  

  Point eye = center + Vector(0,0,60); //Vector(-50,500,60);
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

  double lightBrightness = 0.5;
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

  lightBrightness = 0.8;
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

  rtrt::Array1<string>    names;
  rtrt::Array1<Material*> mtls;

  //////////////// LIVING ROOM ARCHITECTURE ////////////////////////

  names.add("livingroom");
  names.add("phone");

  if( argc < 2 ) {
    cout << "Adding all furniture\n";

    names.add("bookend1");
    names.add("bookend2");

    names.add("books");
    names.add("camera");
    names.add("ceiling_light");
    names.add("chess-set");
    names.add("coffee-table");
    names.add("corella");
    names.add("corella2");
    names.add("door_jams");
    names.add("end-table1");
    names.add("end-table2");
    names.add("fern");
    names.add("fruit-bowl");
    names.add("glasses");
    names.add("glasses_lenses");
    names.add("globe_base");
    names.add("grand_piano");
    names.add("greek_bowl");
    names.add("guitar");
    names.add("horse");
    names.add("horse_bookend1");
    names.add("horse_bookend2");
    names.add("horse_bookend3");
    names.add("horse_bookend4");
    names.add("horse_bookend5");
    names.add("horse_bookend6");
    names.add("mirror1");
    names.add("mirror2");
    names.add("orrey");
    names.add("painting1");
    names.add("painting2");
    names.add("painting3");
    names.add("painting4");
    names.add("painting5");
    names.add("painting6");
    names.add("painting7");
    names.add("paintings");
    names.add("pipe");
    names.add("shelf_pictures");
    names.add("shell1");
    names.add("shell2");
    names.add("wall-sconce1");
    names.add("wall-sconce2");
    names.add("wine_bottle");
    names.add("wine_glasses");
    names.add("wine_table");

    names.add("venus-demilo");
    names.add("chair1");
    names.add("chair2");
    names.add("couch");

  } // end if not min livingroom

  for (int i=0; i < names.size(); i++) {
    cerr << "READING OBJ FILE: " << pathname + names[i] << "\n";
    string objname( pathname + names[i]+ ".obj" );
    string mtlname( pathname + names[i]+ ".mtl" );
    if (!readObjFile(objname, mtlname, room_trans, mtls, group))
      exit(0);

    for( int cnt = 0; cnt < mtls.size(); cnt++ )
      {
	mtls[cnt]->my_lights.add(sunlight);
	mtls[cnt]->my_lights.add(horselight);
	mtls[cnt]->my_lights.add(venuslight);
	mtls[cnt]->my_lights.add(overhead);
	mtls[cnt]->my_lights.add(rSconce);
	mtls[cnt]->my_lights.add(lSconce);
      }
  }

  if( argc < 2 )
    {
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


      //////////////// TEA SET ////////////////////////
      Group * teaGrp = new Group;
      name = "tea-set";  // hgrid 4 6 8
      objname = pathname + name + ".obj";
      mtlname = pathname + name + ".mtl";
      if (!readObjFile(objname, mtlname, room_trans, mtls, teaGrp)) exit(0);
      hg = new HierarchicalGrid( teaGrp, 4, 6, 8,
				 minObjs1, minObjs2, 3 );
      group->add( hg );
      for( int cnt = 0; cnt < mtls.size(); cnt++ )
	{
	  mtls[cnt]->my_lights.add(overhead);
	}
    }

  Color cdown(0.1, 0.1, 0.7);
  Color cup(0.5, 0.5, 0.0);

  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(.9, 0.9, 0.9);
  Scene *scene = 
    //  new Scene( group, cam, bgcolor, cdown, cup, groundplane, 0.3 );
  //new Scene( new Grid(group, 64),
  //       cam, bgcolor, cdown, cup, groundplane, 0.3 );

  new Scene( new HierarchicalGrid(group, 8, 8, 8, 20, 20, 4),
         cam, bgcolor, cdown, cup, groundplane, 0.3 );

  scene->add_per_matl_light( overhead );
  scene->add_per_matl_light( rSconce );
  scene->add_per_matl_light( lSconce );
  scene->add_per_matl_light( horselight );
  scene->add_per_matl_light( venuslight );
  scene->add_per_matl_light( sunlight );

  scene->select_shadow_mode( Hard_Shadows );
  scene->maxdepth = 8;

#if !defined(linux)
  if( argc < 2 || strcmp(argv[1],"-fast") ) {
    cout << "Creating TRIGGERS:\n";

    //////////////// TRIGGERS ////////////////////////
    // livingroom main_text
    Trigger * last;
    string ifpath = "/opt/SCIRun/data/Geometry/interface/";

    PPMImage * ppm = new PPMImage(ifpath+"livingroom/main_text.ppm", true);
    vector<Point> loc; loc.push_back(room_trans.project(Point(0,0,2)));
    Trigger * trig = new Trigger( "Living Room", loc, 5, 200, ppm );
    scene->addTrigger( trig );

    ppm = new PPMImage(ifpath+"livingroom/livingroom_credits1.ppm", true);
    loc.clear(); loc.push_back(room_trans.project(Point(0,1,2)));
    trig = new Trigger( "Living Room", loc, 3, 30, ppm );
    trig->setBasePriority( Trigger::MediumTriggerPriority );
    scene->addTrigger( trig );

    // MAIN CREDITS TRIGGERS
    ppm = new PPMImage(ifpath+"credits/personnel_credits4.ppm", true);
    loc.clear(); loc.push_back(room_trans.project(Point(0,0,0)));
    trig = new Trigger( "Credits 4", loc, 0, 6, ppm, false );
    last = trig;

    ppm = new PPMImage(ifpath+"credits/personnel_credits3.ppm", true);
    loc.clear(); loc.push_back(room_trans.project(Point(0,0,0)));
    trig = new Trigger( "Credits 3", loc, 0,6,ppm,false,NULL,true,trig );

    ppm = new PPMImage(ifpath+"credits/personnel_credits2.ppm", true);
    loc.clear(); loc.push_back(room_trans.project(Point(0,0,0)));
    trig = new Trigger( "Credits 2", loc, 0,6,ppm,false,NULL,true,trig );

    ppm = new PPMImage(ifpath+"credits/personnel_credits1.ppm", true);
    loc.clear(); loc.push_back(room_trans.project(Point(0,0,0)));
    trig = new Trigger( "Credits", loc, 1,6,ppm,true,NULL,true,trig );
    scene->addTrigger( trig );

    last->setNext( trig );
    //  }

    // SCIENCE ROOM TRIGGERS
    ///// INTRO
    ppm = new PPMImage(ifpath+"scienceroom/intro.ppm", true);
    loc.clear(); loc.push_back(Point(-8,8,1.9));
    trig = new Trigger( "Science Intro", loc, 5,30,ppm,true );
    scene->addTrigger( trig );
    ///// WALL 1
    ppm = new PPMImage(ifpath+"scienceroom/science_wall1.ppm", true);
    loc.clear(); loc.push_back(Point(-11,8,1.9));
    trig = new Trigger( "Science Wall 1", loc, 1,30,ppm,true );
    trig->setBasePriority( Trigger::MediumTriggerPriority );
    scene->addTrigger( trig );
    ///// WALL 2
    ppm = new PPMImage(ifpath+"scienceroom/science_wall2.ppm", true);
    loc.clear(); loc.push_back(Point(-8,11,1.9));
    trig = new Trigger( "Science Wall 2", loc, 1,30,ppm,true );
    trig->setBasePriority( Trigger::MediumTriggerPriority );
    scene->addTrigger( trig );

    // GALAXY TRIGGERS
    ///// Intro
    ppm = new PPMImage(ifpath+"galaxy/intro.ppm", true);
    loc.clear(); loc.push_back(Point(29,29,2.0));
    trig = new Trigger( "Galaxy Intro", loc, 33,200,ppm,true );
    scene->addTrigger( trig );

    ///// jpl credit
    ppm = new PPMImage(ifpath+"galaxy/galaxy_credits1.ppm", true);
    loc.clear(); loc.push_back(Point(29,29,2.0));
    trig = new Trigger( "Galaxy Intro", loc, 15,30,ppm,true );
    trig->setBasePriority( Trigger::MediumTriggerPriority );
    scene->addTrigger( trig );

    // ATLANTIS TRIGGERS
    ///// Intro
    ppm = new PPMImage(ifpath+"atlantis/intro.ppm", true);
    loc.clear(); 
    loc.push_back(Point(  2, -5, 2)); // south
    loc.push_back(Point(  1, 10, 2)); // north
    loc.push_back(Point( -9,  2, 2)); // west
    loc.push_back(Point( 10,  1, 2)); // east
    trig = new Trigger( "Atlantis Intro", loc, 2,100,ppm,true );
    scene->addTrigger( trig );
  }
#endif
  
  //////////////// GLOBE INSTANCE ////////////////////////
  if( argc < 2 ) {
    Transform globetrans;
    Group * globegroup = new Group;
    name = pathname + "globe_sphere";
    if( !readObjFile(name + ".obj",
		     name + ".mtl",room_trans,mtls, globegroup ))
      {
	cout << "Error reading: " << name << "\n";
      }
    for( int cnt = 0; cnt < mtls.size(); cnt++ )
      {
	mtls[cnt]->my_lights.add(sunlight);
	mtls[cnt]->my_lights.add(overhead);
      }
    HierarchicalGrid * globegrid = 
      new HierarchicalGrid( globegroup, 4, 6, 8, 10, 10, 4 );
    SpinningInstance * si = 
      new SpinningInstance(
			   new InstanceWrapperObject(globegrid), &globetrans, 
			   room_trans.project(Point(167.633,-8.34035,32.7797)),
			   Vector(-1.32,-46.686,-100.118), -0.1 );

    group->add( si );
    scene->addObjectOfInterest( si, true );
  }

#if !defined(linux)
  {
    string path = "/opt/SCIRun/data/Geometry/sounds/";
    vector<Point> loc;
    loc.push_back(room_trans.project(Point(53,82,64)));  // piano
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

    // MUSEUM ROOM SOUND
    loc.clear();
    loc.push_back(Point(-12,-16,2)); // Center of Museum
    sound = new Sound( path+"violinist-cd074_98.wav", "Violin", 
		       loc, 12, true, 0.5 );
    scene->addSound( sound );

    // GALAXY ROOM SOUND
    loc.clear();
    loc.push_back(Point(29,29,2)); // Center of Galaxy Room.
    sound = new Sound( path+"eerie-sounds-deep-space-cd006_05.wav", "Space", 
		       loc, 30, true, 0.5 );
    scene->addSound( sound );


  }
#endif

  scene->animate=true;

  return scene;
}

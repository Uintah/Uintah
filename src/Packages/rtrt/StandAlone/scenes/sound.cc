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
#if !defined(linux)
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
  Camera cam( Point(-1,0,2), Point( -1,1,2 ), Vector(0,0,1), 45.0 );

  Material* matl = new LambertianMaterial( Color( .1,.9,.9 ) );

  Group * group = new Group();

  double ambient_scale=1.0;
  Color bgcolor(0.1, 0.2, 0.45);
  Color cdown(0.82, 0.62, 0.62);
  Color cup(0.1, 0.3, 0.8);

  rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 1) );
  Scene* scene=new Scene(group, cam,
			 bgcolor, cdown, cup, groundplane,
			 ambient_scale, Arc_Ambient);

  vector<Point> loc;
  Point pos;

  vector<string> soundNames;

  //soundNames.push_back("welcome.wav");
  soundNames.push_back("water-flowing1.wav");
  soundNames.push_back("cool_music.wav");
  soundNames.push_back("aquarium-aerator-cd039_59.wav");
  soundNames.push_back("clock-tower-bells-cd025_75.wav");
  soundNames.push_back("computer-scrambling-sound-cd005_80.wav");
  //soundNames.push_back("cue-ball-break-cd046_73.wav");
  soundNames.push_back("eerie-sounds-deep-space-cd006_05.wav");
  soundNames.push_back("electronic-sounding-alarm-cd006_60.wav");
  soundNames.push_back("evaporating-sound-cd006_29.wav");
  soundNames.push_back("fire-crackling-cd078_36.wav");
  soundNames.push_back("generator-on-off-cd059_01.wav");
  soundNames.push_back("glass-crackling-falling-cd011_11.wav");
  soundNames.push_back("harp-ascend-decend-cd027_36.wav");
  soundNames.push_back("harp-ascending-cd027_27.wav");
  soundNames.push_back("harp-heaven-rolls-cd027_41.wav");
  soundNames.push_back("harp-melodic-cd027_59.wav");
  soundNames.push_back("harp-melodic-intro-cd027_56.wav");
  soundNames.push_back("helicopter-idle-cd030_01.wav");
  soundNames.push_back("kettle-on-stove-cd008_79.wav");
  soundNames.push_back("laser-gun-shots-cd005_56.wav");
  soundNames.push_back("music-box-cd074_96.wav");
  soundNames.push_back("operating-generator-cd016_07.wav");
  soundNames.push_back("player-piano-cd026_73.wav");
  soundNames.push_back("pond-sounds-cd065_46.wav");
  soundNames.push_back("rocket-phasing-left-to-right-cd005_47.wav");
  soundNames.push_back("seagulls-and-surf-cd003_22.wav");
  soundNames.push_back("seagulls-ocean-waves-cd042_86.wav");
  soundNames.push_back("submarine-sonar-cd045_32.wav");
  soundNames.push_back("ticking-clock-cd058_07.wav");
  soundNames.push_back("transporter-sound-cd006_09.wav");
  soundNames.push_back("trumpet-reveille-cd045_14.wav");
  soundNames.push_back("underwater-sound-cd100_24.wav");
  soundNames.push_back("violinist-cd074_98.wav");
  soundNames.push_back("waves-ocean-shoreline-cd039_27.wav");

#if !defined(linux)
  string path = "/home/sci/dav/sounds/";

  Sound * sound;
  int cnt = 0;

  for( int i = 0; i < 4; i++ ) {
    for( int j = 0; j < 4; j++ ) {

      pos = Point(i*5,j*5,0);
      loc.push_back(pos);
      Object* obj  = new Sphere( matl, pos, 1 );
      sound = 
	new Sound( path + soundNames[cnt], soundNames[cnt], loc, 5, true );
      scene->addSound( sound );
      loc.clear();
      group->add( obj );
      cnt++;
    }
  }

#endif

  Light * light = new Light(Point(20,20,50), Color(1,1,1), 0.8);
  light->name_ = "main light";
  scene->add_light( light );

  scene->set_background_ptr( new LinearBackground( Color(1.0, 1.0, 1.0),
						   Color(0.0,0.0,0.0),
						   Vector(0,0,1)) );
  scene->select_shadow_mode( Hard_Shadows );
  return scene;
}


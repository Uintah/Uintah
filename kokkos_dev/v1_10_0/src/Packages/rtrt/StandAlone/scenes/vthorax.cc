
#include <Packages/rtrt/Core/SelectableGroup.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/CutPlane.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/HVolumeVG.h>
#include <Packages/rtrt/Core/HVolume.h>
#include <Packages/rtrt/Core/HVolumeBrickColor.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/PlaneDpy.h>
#include <Packages/rtrt/Core/Hist2DDpy.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Thread/Thread.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string.h>

using SCIRun::Thread;
using namespace rtrt;
using namespace std;

void usage(char* me) {
  cerr << "Valid options for scene: " << me << '\n';
  cerr << " -rate [time between frames]\n";
  cerr << " -depth [for volume creation]\n";
  cerr << " -t,-texture [file for rgb texture data]\n";
  cerr << " -vg [file for bivariate data]\n";
  cerr << " -s,-scalar [file for third scalar vis]\n";
  cerr << " -nofixlight  turns off light being fixed with camera\n";
  cerr << " -seperate  the scalar and bivariate field will be rendered sperately\n";
}
  
extern "C" 
Scene* make_scene(int argc, char* argv[], int nworkers)
{
  char* tex_file = 0;
  char* vg_file = 0;
  char* scalar_file = 0;
  
  int depth=3;
  float rate = -1;

  bool use_cut = false;
  bool fix_lights = true;
  bool together = true;
  
  // Need to have at least one thing
  if (argc < 2) {
    usage(argv[0]);
    return 0;
  }
  
  for(int i=1;i<argc;i++){
    if(strcmp(argv[i], "-depth")==0){
      i++;
      depth=atoi(argv[i]);
    } else if(strcmp(argv[i], "-t") == 0 ||
	      strcmp(argv[i], "-texture") == 0){
      tex_file = argv[++i];
    } else if(strcmp(argv[i], "-vg")==0){
      vg_file = argv[++i];
    } else if(strcmp(argv[i], "-s")==0 ||
	      strcmp(argv[i], "-scalar")==0){
      scalar_file = argv[++i];
    } else if(strcmp(argv[i], "-rate")==0){
      rate = atof(argv[++i]);
    } else if(strcmp(argv[i], "-cut")==0){
      use_cut = true;
    } else if(strcmp(argv[i], "-nofixlight")==0){
      fix_lights = false;
    } else if(strcmp(argv[i], "-seperate")==0){
      together = false;
    } else {
      cerr << "Unknown option: " << argv[i] << '\n';
      usage(argv[0]);
      return 0;
    }
  }

  if (vg_file == 0 && scalar_file == 0) {
    cerr << "Need at least one field to render.\n";
    // need at least one field
    usage(argv[0]);
    return 0;
  }
  
  Color surf(.5, 0.1, 0.15);
  Material* matl0 = 0;
  if (tex_file) {
    // get the color data
    matl0=new HVolumeBrickColor(tex_file, nworkers, .9, .9, .9, 50,  0);
  } else {
    // use a default color
    matl0=new Phong(surf, Color(1,1,1), 100, 0);
  }

  Group *group;

  if (together) {
    group = new Group();
  } else {
    if (rate > 0) {
      group = new SelectableGroup(rate);
    } else {
      group = new SelectableGroup();
      // This turns off autoswitch
      ((SelectableGroup*)group)->toggleAutoswitch();
    }
  }

  group->set_name("Thorax Scalar Fields");

  if (vg_file) {
    Hist2DDpy* dpy=new Hist2DDpy();
  
    cerr << "Reading " << vg_file << "\n";
    Object *vg;
    vg = new HVolumeVG<VG<unsigned char, unsigned char>, BrickArray3<VG<unsigned char, unsigned char> >, BrickArray3<VMCell<VG<unsigned char, unsigned char> > > >
      (matl0, dpy, vg_file, depth, nworkers);
    group->add(vg);
  
    (new Thread(dpy, "Histogram GUI thread"))->detach();
  }

  if (scalar_file) {
    VolumeDpy *dpy = new VolumeDpy();
    cerr << "Reading " << scalar_file << "\n";
    Object * sf;
    sf=new HVolume<unsigned char, BrickArray3<unsigned char>, BrickArray3<VMCell<unsigned char> > > (matl0, dpy, scalar_file, depth, nworkers);
    group->add(sf);
    (new Thread(dpy, "Volume GUI thread"))->detach();
  }

  Object *top;
  if (use_cut) {
    // add the top object
    top = group;
  } else {
    top = group;
  }
  
  double bgscale=0.3;
  Color groundcolor(0,0,0);
  Color averagelight(1,1,1);
  double ambient_scale=.5;
  
  Color bgcolor(0,0,0);
  
  Plane groundplane ( Point(0, 0, 0), Vector(1, 0, 0) );

  // Let's try and make a good guess as to where the light and eye should be.
  BBox bounds;
  group->compute_bounds(bounds, 0);
  // Take the corners and then extend one of them
  Point light0_loc = bounds.max() + (bounds.max()-bounds.min())*0.2;
  Point eye_loc(light0_loc.x(), 0, light0_loc.z() );
  if (fix_lights)
    light0_loc = eye_loc;
  
  Camera cam(eye_loc, ((bounds.max() - bounds.min())*.5).point(),
	     Vector(0,1,0), 60);
  
  Scene* scene=new Scene(top, cam,
			 bgcolor, groundcolor*averagelight, bgcolor, groundplane, 
			 ambient_scale);
  Light *light0 = new Light(light0_loc, Color(1,1,.8),
			    (bounds.max()-bounds.min()).length()*0.01);
  light0->name_ = "light 0";
  light0->fixed_to_eye = fix_lights;
  scene->add_light(light0);
  scene->shadow_mode=No_Shadows;
  scene->addObjectOfInterest(group, true);
  scene->maxdepth=0;
  
  return scene;
}

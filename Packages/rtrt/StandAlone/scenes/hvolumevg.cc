
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/CutPlane.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/HVolumeVG.h>
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

extern "C" 
Scene* make_scene(int argc, char* argv[], int nworkers)
{
  char* file=0;
  int depth=3;
  for(int i=1;i<argc;i++){
    if(strcmp(argv[i], "-depth")==0){
      i++;
      depth=atoi(argv[i]);
    } else {
      if(file){
	cerr << "Unknown option: " << argv[i] << '\n';
	cerr << "Valid options for scene: " << argv[0] << '\n';
	cerr << " -rate\n";
	cerr << " -depth\n";
	return 0;
      }
      file=argv[i];
    }
  }
  
  Camera cam(Point(5,0,0), Point(0,0,0),
	     Vector(0,1,0), 60);
  
  Color surf(.5, 0.1, 0.15);
  Material* matl0=new Phong(surf, Color(1,1,1), 100, 0);
  Hist2DDpy* dpy=new Hist2DDpy();
  
  cerr << "Reading " << file << "\n";
  HVolumeVG<VG<unsigned short, unsigned short>, BrickArray3<VG<unsigned short, unsigned short> >, BrickArray3<VMCell<VG<unsigned short, unsigned short> > > >* o
    =new HVolumeVG<VG<unsigned short, unsigned short>, BrickArray3<VG<unsigned short, unsigned short> >, BrickArray3<VMCell<VG<unsigned short, unsigned short> > > >
    (matl0, dpy, file, depth, nworkers);
  
  (new Thread(dpy, "Histogram GUI thread"))->detach();
  
  double bgscale=0.3;
  Color groundcolor(0,0,0);
  Color averagelight(1,1,1);
  double ambient_scale=.5;
  
  Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);
  
  Plane groundplane ( Point(0, 0, 0), Vector(1, 0, 0) );
  Scene* scene=new Scene(o, cam,
			 bgcolor, groundcolor*averagelight, bgcolor, groundplane, 
			 ambient_scale);
  // Let's try and make a good guess as to where the light should be.
  BBox bounds;
  o->compute_bounds(bounds, 0);
  // Take the corners and then extend one of them
  Point light0_loc = bounds.max() + (bounds.max()-bounds.min())*0.2;
  
  Light *light0 = new Light(light0_loc, Color(1,1,.8),
			    (bounds.max()-bounds.min()).length()*0.01);
  light0->name_ = "light 0";
  scene->add_light(light0);
  scene->shadow_mode=No_Shadows;
  scene->addObjectOfInterest(o, true);
  scene->maxdepth=0;
  
  return scene;
}

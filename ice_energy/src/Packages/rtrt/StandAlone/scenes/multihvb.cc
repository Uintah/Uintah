

#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/CutPlane.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/HVolume.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/PlaneDpy.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Slice.h>
#include <Packages/rtrt/Core/TimeObj.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>

using namespace rtrt;
using namespace std;
using SCIRun::Thread;
using SCIRun::Mutex;

extern "C" 
Scene* make_scene(int argc, char* argv[], int nworkers)
{
    double rate=3;
    char* file=0;
    int depth=3;
    for(int i=1;i<argc;i++){
	if(strcmp(argv[i], "-rate")==0){
	    i++;
	    rate=atof(argv[i]);
	} else if(strcmp(argv[i], "-depth")==0){
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

    Color surf(.50000, 0.0, 0.00);
    Material* matl0=new Phong(surf*0.6, surf*0.6, 100, .4);
    VolumeDpy* dpy=new VolumeDpy();
    ifstream in(file);
    PlaneDpy* pdpy=new PlaneDpy(Vector(1,0,0), Point(0,0,0));

    TimeObj* timeobj1=new TimeObj(rate);
    TimeObj* timeobj2=new TimeObj(rate);
    while(in){
	char file[1000];
	in >> file;
	if(in){
	    cerr << "Reading " << file << "\n";
	    HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > >* o
		=new HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > >
		(matl0, dpy, file, depth, nworkers);
	    timeobj1->add(o);
	    Object* o2=new Slice<float, BrickArray3<float>, BrickArray3<VMCell<float> > >
		(dpy, pdpy, o);
	    timeobj2->add(o2);
	}
    }
    Group* group=new Group();
    Object* obj1=timeobj1;
    obj1=new CutPlane(obj1, Point(0,.75,0), Vector(0,-1,0));
    obj1=new CutPlane(obj1, Point(0,.25,0), Vector(0,1,0));
    group->add(new CutPlane(obj1, pdpy));
    group->add(timeobj2);
    (new Thread(pdpy, "CutPlane GUI thread"))->detach();
    (new Thread(dpy, "Volume GUI thread"))->detach();

    double bgscale=0.3;
    Color groundcolor(0,0,0);
    Color averagelight(1,1,1);
    double ambient_scale=.5;

    Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);

    Plane groundplane ( Point(0, 0, 0), Vector(1, 0, 0) );
    Scene* scene=new Scene(group, cam,
			   bgcolor, groundcolor*averagelight, bgcolor, groundplane, 
			   ambient_scale);
    scene->add_light(new Light(Point(5,-3,3), Color(1,1,.8)*2, 0));
    scene->select_shadow_mode( No_Shadows );
    scene->maxdepth=0;
    scene->attach_display(dpy);
    //scene->attach_display(pdpy);
    return scene;
}

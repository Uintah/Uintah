#define GROUP_IN_TIMEOBJ 1

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Cylinder.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/HVolume.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/HVolumeBrickColor.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/CutPlane.h>
#include <Packages/rtrt/Core/PlaneDpy.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/SketchMaterial.h>
#ifdef GROUP_IN_TIMEOBJ
#include <Packages/rtrt/Core/TimeObj.h>
#endif
#include <Core/Thread/Thread.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdlib.h>

using namespace rtrt;
using SCIRun::Thread;

extern "C" 
Scene* make_scene(int argc, char* argv[], int nworkers)
{
    int depth=3;
    Array1<char*> files;
    bool cut=false;
#ifdef GROUP_IN_TIMEOBJ
    double rate=3;
#endif
    for(int i=1;i<argc;i++){
	if(strcmp(argv[i], "-depth")==0){
	    i++;
	    depth=atoi(argv[i]);
	} else if(strcmp(argv[i], "-cut")==0){
	    cut=true;
#ifdef GROUP_IN_TIMEOBJ
	} else if(strcmp(argv[i], "-rate")==0){
	  rate = atof(argv[++i]);
#endif
	} else if(argv[i][0] != '-'){
	    files.add(argv[i]);
	} else {
	    cerr << "Unknown option: " << argv[i] << '\n';
	    cerr << "Valid options for scene: " << argv[0] << '\n';
	    cerr << " -depth n   - set depth of hierarchy\n";
	    cerr << " file       - raw file name\n";
	    return 0;
	}
    }

    if(files.size()==0){
	cerr << "Must specify at least one file\n";
	return 0;
    }

    Camera cam(Point(1,3,1), Point(0.5,0.5,0.5), Vector(0,1,0), 40);

    Material* matl0;
    matl0=new Phong(Color(.6,1,.4), Color(0,0,0), 100, 0);

    VolumeDpy* dpy=new VolumeDpy(40);
    Object* obj;

    if(files.size()==1){
	HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > > *hvol = new HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > > (matl0, dpy, files[0], depth, nworkers);
#if 1
	// Now to add a new material
	BBox bbox;
	hvol->compute_bounds(bbox, 0);
	Material *sm = new SketchMaterial<BrickArray3<short>, short>(hvol->blockdata, bbox);
	hvol->set_matl(sm);
#endif
	obj = hvol;
    } else {
#ifdef GROUP_IN_TIMEOBJ
	TimeObj* group = new TimeObj(rate);
	cout << "using time changing objects\n";
#else
	Group* group=new Group();
	cout << "Grouping all objects together\n";
#endif
	obj=group;
	for(int i=0;i<files.size();i++){
	  HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > > * hvol = new HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > > (matl0, dpy, files[i], depth, nworkers);
	  group->add(hvol);
	}
    }

    if(cut){
	PlaneDpy* pd=new PlaneDpy(Vector(0,0,1), Point(0,0,100));
	obj=new CutPlane(obj, pd);
	obj->set_matl(matl0);
	(new Thread(pd, "Cutting plane display thread"))->detach();
    }

    // Start up the thread to handle the slider
    (new Thread(dpy, "Volume GUI thread"))->detach();
	
    //double bgscale=0.5;
    double ambient_scale=.5;

    Color bgcolor(0.01, 0.05, 0.3);
    Color cup(1, 0, 0);
    Color cdown(0, 0, 0.2);

    rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, -1, 0) );
    Scene* scene=new Scene(obj, cam,
			   bgcolor, cdown, cup, groundplane,
			   ambient_scale);
    scene->addObjectOfInterest(obj, true);
    //scene->add_light(new Light(Point(50,-30,30), Color(1.0,0.8,0.2), 0));
    Light *light0 = new Light(Point(1100,-600,3000), Color(1.0,1.0,1.0), 0);
    light0->name_ = "light 0";
    scene->add_light(light0);
    scene->set_background_ptr( new LinearBackground(
                               Color(0.2, 0.4, 0.9),
                               Color(0.0,0.0,0.0),
                               Vector(1, 0, 0)) );

    scene->select_shadow_mode( No_Shadows );
    scene->attach_display(dpy);
    return scene;
}


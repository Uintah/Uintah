
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/HVolume.h>
#include <Packages/rtrt/Core/HVolumeBrickColor.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <Core/Thread/Thread.h>
#include <iostream>
#include <math.h>
#include <string.h>

using namespace rtrt;
using namespace std;
using SCIRun::Thread;

extern "C" 
Scene* make_scene(int argc, char* argv[], int nworkers)
{
    bool bone=false;
    bool tex=false;
    bool transp=false;
    bool rect=false;
    char* texfile=0;
    int depth=3;
    char* camera=0;
    for(int i=1;i<argc;i++){
	if(strcmp(argv[i], "-tex")==0){
	    tex=true;
	} else if(strcmp(argv[i], "-depth")==0){
	    i++;
	    depth=atoi(argv[i]);
	} else if(strcmp(argv[i], "-bone")==0){
	    bone=true;
	} else if(strcmp(argv[i], "-texfile")==0){
	    i++;
	    texfile=argv[i];
	} else if(strcmp(argv[i], "-rect")==0){
	    rect=true;
	} else if(strcmp(argv[i], "-transp")==0){
	    transp=true;
	} else if(strcmp(argv[i], "-camera")==0){
	    i++;
	    camera=argv[i];
	} else {
	    cerr << "Unknown option: " << argv[i] << '\n';
	    cerr << "Valid options for scene: " << argv[0] << '\n';
	    cerr << " -tex\n";
	    cerr << " -bone\n";
	    cerr << " -texfile\n";
	    cerr << " -transp\n";
	    cerr << " -rect\n";
	    cerr << " -camera\n";
	    return 0;
	}
    }

    double bgscale=0.3;
    Color groundcolor(0,0,0);
    Color averagelight(1,1,1);
    double ambient_scale=.5;

    Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);

    Camera cam(Point(744.682, -1506.04, -63.1467),
	       Point(856.916, 32.9168, 536.683),
	       Vector(-0.99723, 0.0742708, -0.00396236),
	       67.6197);

    if(camera){
	if(strcmp(camera, "feet")==0){
	    cam=Camera(Point(744.682, -1506.04, -63.1467),
		       Point(856.916, 32.9168, 536.683),
		       Vector(-0.99723, 0.0742708, -0.00396236),
		       67.6197);
	} else if(strcmp(camera, "torso")== 0){
	    cam=Camera(Point(494.801, -1314.91, -506.777),
		       Point(607.035, 224.051, 93.0523),
		       Vector(-0.99723, 0.0742708, -0.00396236),
		       28.0168);
	} else if(strcmp(camera, "fingers")==0){
	    cam=Camera(Point(701.261, -1373.05, -396.229),
		       Point(813.495, 165.907, 203.601),
		       Vector(-0.99723, 0.0742708, -0.00396236),
		       2.45506);
	} else if(strcmp(camera, "slowfeet")==0){
	    cam=Camera(Point(2602.31, -631.56, -88.9464),
		       Point(1228.06, 285.507, 17.0367),
		       Vector(-0.553438, -0.832454, 0.0269558),
		       6.47977);
	} else if(strcmp(camera, "knee")==0){
	    cam=Camera(Point(1015.8, -1086.82, -1004.09),
		       Point(1253.67, 115.682, 108.64),
		       Vector(-0.989588, 0.111193, 0.0913829),
		       6.25971);
	} else {
	    cerr << "Unknown camera: " << camera << '\n';
	}
    }

    Plane groundplane ( Point(1000, 0, 0), Vector(-1, 0.0, 0) );
    Scene* scene=new Scene(0, cam,
			   bgcolor, groundcolor*averagelight, bgcolor, groundplane, 
			   ambient_scale);

    Material* matl0;
    if(!tex){
	if(bone){
	    Color bone(0.9608, 0.8706, 0.7020);
	    matl0=new Phong(bone*.6, bone*.6, 100, 0);
	} else {
	    Color flesh(1.0000, 0.4900, 0.2500);
	    if(!transp)
		matl0=new Phong(flesh*.6, flesh*.6, 100, 0);
	    else
		matl0=new Phong(flesh*.1, flesh*.1, 100, 0);
	}
    } else {
	if(!texfile)
	    texfile="/scratch/sparker/vfemcolor";
	matl0=new HVolumeBrickColor(texfile, nworkers,
				    .1, .5, .6, 50,  0);
    }
    VolumeDpy* dpy=new VolumeDpy(bone?1224.5:600.5);
    scene->attach_display(dpy);
    HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >* obj0=
	new HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >
	(matl0, dpy, "vfem/vfem16_0", depth, nworkers);
    HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >* obj1=
	new HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >
	(matl0, dpy, "vfem/vfem16_1", depth, nworkers);
    HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >* obj2=
	new HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >
	(matl0, dpy, "vfem/vfem16_2", depth, nworkers);
    HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >* obj3=
	new HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >
	(matl0, dpy, "vfem/vfem16_3", depth, nworkers);
    HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >* obj4=
	new HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >
	(matl0, dpy, "vfem/vfem16_4", depth, nworkers);
    HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >* obj5=
	new HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >
	(matl0, dpy, "vfem/vfem16_5", depth, nworkers);
    HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >* obj6=
	new HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >
	(matl0, dpy, "vfem/vfem16_6", depth, nworkers);
    (new Thread(dpy, "Volume GUI thread"))->detach();

    Group* group=new Group();
    group->add(obj0);
    group->add(obj1);
    group->add(obj2);
    group->add(obj3);
    group->add(obj4);
    group->add(obj5);
    group->add(obj6);
    if(transp){
	Material* matl1;
	if(!tex){
	    Color bone(0.9608, 0.8706, 0.7020);
	    matl1=new Phong( bone*.6, bone*.6, 100, 0);
	} else {
	    matl1=matl0;
	}
	VolumeDpy* dpy2=new VolumeDpy(1459.5);
	scene->attach_display(dpy2);
	HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >* obj00=
	    new HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >
	    (matl1, dpy, obj0);
	HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >* obj11=
	    new HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >
	    (matl1, dpy, obj1);
	HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >* obj22=
	    new HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >
	    (matl1, dpy, obj2);
	HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >* obj33=
	    new HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >
		(matl1, dpy, obj3);
	HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >* obj44=
	    new HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >
	    (matl1, dpy, obj4);
	HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >* obj55=
	    new HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >
	    (matl1, dpy, obj5);
	HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >* obj66=
	    new HVolume<short, BrickArray3<short>, BrickArray3<VMCell<short> > >
	    (matl1, dpy, obj6);
	Group* group2=new Group();
	group2->add(obj00);
	group2->add(obj11);
	group2->add(obj22);
	group2->add(obj33);
	group2->add(obj44);
	group2->add(obj55);
	group2->add(obj66);
	group->add(group2);
	scene->shadowobj=group2;
	(new Thread(dpy2, "Volume GUI thread"))->detach();
    }
    if(rect){
	group->add(new Rect(matl0, Point(1734/2.,5,5),
			    Vector(1732/2.,0,0), Vector(0,250,0)));
	group->add(new Rect(matl0, Point(1734/2.,5,5),
			    Vector(0,250,0), Vector(0,0,250)));
	group->add(new Rect(matl0, Point(1734/2.,5,5),
			    Vector(1732/2.,0,0), Vector(0,0,250)));
    }

    BV1* obj=new BV1(group);

    scene->set_object(obj);
    scene->addObjectOfInterest(group, true);
    scene->add_light(new Light(Point(1000,-3000,0), Color(1,1,1), 0));
    scene->select_shadow_mode( No_Shadows );
    return scene;
}


#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/HVolumeBrick16.h>
#include <Packages/rtrt/Core/HVolumeBrickColor.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
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
    bool use_iso=false;
    double isoval=0;
    char* texfile=0;
    int depth=3;
    char* camera=0;
    for(int i=1;i<argc;i++){
	if(strcmp(argv[i], "-tex")==0){
	    tex=true;
	} else if(strcmp(argv[i], "-depth")==0){
	    i++;
	    depth=atoi(argv[i]);
 	} else if(strcmp(argv[i], "-isoval")==0){
	    i++;
	    use_iso=true;
	    isoval=atof(argv[i]);
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


    Plane groundplane ( Point(5000, 0, 0), Vector(-1, -3, 3) );
    Color cdown(0.1, 0.2, 0.8);
    Color cup(0.02, 0.06, 0.16);
    Color bgcolor(0.3, 0.3, 0.3);
    double ambient_scale=1.0;

    Scene* scene=new Scene(0, cam,
			   bgcolor, cdown, cup, groundplane,
			   ambient_scale, Arc_Ambient);
    scene->add_light(new Light(Point(-3000,-2000,1500), Color(2.0,1.2,0.4), 0));
    scene->select_shadow_mode( No_Shadows );
/*
    scene->set_background_ptr( new LinearBackground(
                               Color(0.0,0.0,0.0),
                               Color(0.5, 0.5, 0.5),
                               Vector(1,0,0)) );
*/

    Material* matl0;
    if(!tex){
	//Color flesh(1.0000, 0.4900, 0.2500);
         if(!transp)
               //matl0 = new Phong(Color(0,0,0), Color(1,1,1), Color(0,0,0), 10, 0);
               matl0=new LambertianMaterial(Color(0.5,0.5,0.5));
         else
              matl0=new PhongMaterial(Color(1.0, 0.73, 0.54), 0.1);

    } else {
	if(!texfile)
	    texfile="/scratch/sparker/vfemcolor";
	matl0=new HVolumeBrickColor(texfile, nworkers,
				    .1, .5, .6, 50, 0);
    }
    VolumeDpy* dpy=new VolumeDpy(use_iso?isoval:bone?1224.5:600.5);
    scene->attach_display(dpy);
    HVolumeBrick16* obj0=new HVolumeBrick16(matl0, dpy,
					    "vfem/vfem16_0",
					    depth, nworkers);
    HVolumeBrick16* obj1=new HVolumeBrick16(matl0, dpy,
					    "vfem/vfem16_1",
					    depth, nworkers);
    HVolumeBrick16* obj2=new HVolumeBrick16(matl0, dpy,
					    "vfem/vfem16_2",
					    depth, nworkers);
    HVolumeBrick16* obj3=new HVolumeBrick16(matl0, dpy,
					    "vfem/vfem16_3",
					    depth, nworkers);
    HVolumeBrick16* obj4=new HVolumeBrick16(matl0, dpy,
					    "vfem/vfem16_4",
					    depth, nworkers);
    HVolumeBrick16* obj5=new HVolumeBrick16(matl0, dpy,
					    "vfem/vfem16_5",
					    depth, nworkers);
    HVolumeBrick16* obj6=new HVolumeBrick16(matl0, dpy,
					    "vfem/vfem16_6",
					    depth, nworkers);
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
	    matl1=new LambertianMaterial(Color(1.0,1.0,1.0));
	} else {
	    matl1=matl0;
	}
	VolumeDpy* dpy2=new VolumeDpy(1459.5);
	scene->attach_display(dpy2);
	HVolumeBrick16* obj00=new HVolumeBrick16(matl1, dpy2, obj0);
	HVolumeBrick16* obj11=new HVolumeBrick16(matl1, dpy2, obj1);
	HVolumeBrick16* obj22=new HVolumeBrick16(matl1, dpy2, obj2);
	HVolumeBrick16* obj33=new HVolumeBrick16(matl1, dpy2, obj3);
	HVolumeBrick16* obj44=new HVolumeBrick16(matl1, dpy2, obj4);
	HVolumeBrick16* obj55=new HVolumeBrick16(matl1, dpy2, obj5);
	HVolumeBrick16* obj66=new HVolumeBrick16(matl1, dpy2, obj6);

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
    return scene;
}

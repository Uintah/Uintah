#define GROUP_IN_TIMEOBJ 1

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Cylinder.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/HVolumeBrick16.h>
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
#ifdef GROUP_IN_TIMEOBJ
#include <Packages/rtrt/Core/TimeObj.h>
#endif
#include <Core/Thread/Thread.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdlib.h>

using namespace rtrt;
using namespace std;
using SCIRun::Thread;

extern "C" 
Scene* make_scene(int argc, char* argv[], int nworkers)
{
    int depth=3;
    char* texfile=0;
    bool showgrid=false;
    bool xyslice=false;
    bool xzslice=false;
    bool yzslice=false;
    Array1<char*> files;
    bool cut=false;
#ifdef GROUP_IN_TIMEOBJ
    double rate=3;
#endif
    for(int i=1;i<argc;i++){
	if(strcmp(argv[i], "-depth")==0){
	    i++;
	    depth=atoi(argv[i]);
	} else if(strcmp(argv[i], "-showgrid")==0){
	    showgrid=true;
	} else if(strcmp(argv[i], "-texture")==0){
	    i++;
	    texfile=argv[i];
	} else if(strcmp(argv[i], "-xyslice")==0){
	    xyslice=true;
	} else if(strcmp(argv[i], "-xzslice")==0){
	    xzslice=true;
	} else if(strcmp(argv[i], "-yzslice")==0){
	    yzslice=true;
	} else if(strcmp(argv[i], "-cut")==0){
	    cut=true;
	} else if(argv[i][0] != '-'){
	    files.add(argv[i]);
#ifdef GROUP_IN_TIMEOBJ
	} else if(strcmp(argv[i], "-rate")==0){
	  rate = atof(argv[++i]);
#endif
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

#if 0
    Camera cam(Point(991.526, 506.642, 179.416),
	       Point(0, 0, 0),
	       // Vector(0.47655, -0.8301, -0.28954),
	       Vector(1, 0, 0),
	       42.566);
#else
#if 0
    Camera cam(Point(1173.01, -537.953, -287.888),
		Point(1121, 0, 0),
		Vector(-1,0,0),
		18.91);
#else
#if 0
    Camera cam(Point(1292.31, -538.193, 208.269),
		Point(1439.96, 38.0231, 62.8303),
		Vector(-1,0,0),
		64.8);
#else
    Camera cam(Point(1501.35, -482.14, -257.168),
		Point(1461.09, 56.3614, 31.5762),
		Vector(-1,0,0),
		34.62);
#endif
#endif
#endif

    Material* matl0;
    if(texfile){
        matl0=new HVolumeBrickColor(texfile, nworkers,
				    .6, .7, .6, 50,  0);
    } else {
	matl0=new Phong(Color(.6,1,.4), Color(0,0,0), 100, 0);
    }

/*
    Material* matl0=new CoupledMaterial( Color(0.5, 0.5, 0.8), 0.05);
*/
    VolumeDpy* dpy=new VolumeDpy(779);
    Object* obj;
    if(files.size()==0 && !showgrid){
	obj=new HVolumeBrick16(matl0, dpy, files[0],
			       depth, nworkers);
    } else {
#ifdef GROUP_IN_TIMEOBJ
	TimeObj* group = new TimeObj(rate);
#else
	Group* group=new Group();
#endif
	obj=group;
	for(int i=0;i<files.size();i++){
	    HVolumeBrick16* hvol=new HVolumeBrick16(matl0, dpy, files[i],
						    depth, nworkers);
	    group->add(hvol);
	    if(showgrid){
		// This is lame - it only works for one data file...
#if 0
		Material* cylmatl=new Phong(Color(.4,.4,.4),
					    Color(.5,.5,.5), 100, 0);
		Material* cylmatl=new DielectricMaterial(1,1,.1,100,Color(.45,.97,.7), Color(1,1,1));
#endif
		Material* cylmatl=new LambertianMaterial( Color(0.3,0.3,0.3) );
		int nx=hvol->get_nx();
		int ny=hvol->get_ny();
		int nz=hvol->get_nz();
		BBox bbox;
		hvol->compute_bounds(bbox, 0);
		Point min(bbox.min());
		Point max(bbox.max());
		Vector diag(max-min);
		double radius=Min(diag.x()/nx, diag.y()/ny, diag.z()/nz);
		radius/=16;
		for(int x=0;x<nx;x++){
		    for(int y=0;y<ny;y++){
			double xn=double(x)/double(nx-1)*diag.x()+min.x();
			double yn=double(y)/double(ny-1)*diag.y()+min.y();
			group->add(new Cylinder(cylmatl,
						Point(xn,yn,min.z()),
						Point(xn,yn,max.z()),
						radius));
		    }
		}
		for(int x=0;x<nx;x++){
		    for(int z=0;z<nz;z++){
			double xn=double(x)/double(nx-1)*diag.x()+min.x();
			double zn=double(z)/double(nz-1)*diag.z()+min.z();
			group->add(new Cylinder(cylmatl,
						Point(xn,min.y(),zn),
						Point(xn,max.z(),zn),
						radius));
		    }
		}
		for(int z=0;z<nz;z++){
		    for(int y=0;y<ny;y++){
			double zn=double(z)/double(nz-1)*diag.z()+min.z();
			double yn=double(y)/double(nz-1)*diag.y()+min.y();
			group->add(new Cylinder(cylmatl,
						Point(min.z(),yn,zn),
						Point(max.z(),yn,zn),
						radius));
		    }
		}
		for(int x=0;x<nx;x++){
		    for(int y=0;y<ny;y++){
			for(int z=0;z<nz;z++){
			    double xn=double(x)/double(nx-1)*diag.x()+min.x();
			    double yn=double(y)/double(nz-1)*diag.y()+min.y();
			    double zn=double(z)/double(nz-1)*diag.z()+min.z();
			    group->add(new Sphere(cylmatl, Point(xn,yn,zn), radius));
			}
		    }
		}
	    }
	}
#if 0
	group=new Group();
	Point p1(0,0,0);
	Point p2(1,0,0);
	double rad=1;
	group->add(new Sphere(cylmatl, p1, .2));
	group->add(new Sphere(cylmatl, p2, .2));
	group->add(new Sphere(cylmatl, p2+Vector(1,0,0)*rad, .1));
	group->add(new Sphere(cylmatl, p2+Vector(-1,0,0)*rad, .1));
	group->add(new Sphere(cylmatl, p2+Vector(0,-1,0)*rad, .1));
	group->add(new Sphere(cylmatl, p2+Vector(0,1,0)*rad, .1));
	group->add(new Cylinder(cylmatl, p1, p2, rad));
	obj=group;
#endif
    }
    if(xyslice || xzslice || yzslice){
	Group* group=new Group();
	group->add(obj);
	BBox bbox;
	obj->compute_bounds(bbox, 0);
	Vector diag(bbox.diagonal()*0.5);
	Point mid(bbox.min()+diag);
	if(xyslice){
	    group->add(new Rect(matl0, mid, Vector(diag.x(), 0, 0),
				Vector(0, diag.y(), 0)));
	}
	if(xzslice){
	    group->add(new Rect(matl0, mid, Vector(diag.x(), 0, 0),
				Vector(0, 0, diag.z())));
	}
	if(yzslice){
	    group->add(new Rect(matl0, mid, Vector(0, diag.y(), 0),
				Vector(0, 0, diag.z())));
	}
	obj=group;
    }

    PlaneDpy *pd = 0;
    if(cut){
	pd=new PlaneDpy(Vector(0,0,1), Point(0,0,100));
	obj=new CutPlane(obj, pd);
	(new Thread(pd, "Cutting plane display thread"))->detach();
    }

    // Start up the thread to handle the slider
    new Thread(dpy, "Volume GUI thread");
	
    //double bgscale=0.5;
    double ambient_scale=.5;

    Color bgcolor(0.01, 0.05, 0.3);
    Color cup(1, 0, 0);
    Color cdown(0, 0, 0.2);

    rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, -1, 0) );
    Scene* scene=new Scene(obj, cam,
			   bgcolor, cdown, cup, groundplane,
			   ambient_scale);
    //scene->add_light(new Light(Point(50,-30,30), Color(1.0,0.8,0.2), 0));
    scene->add_light(new Light(Point(1100,-600,3000), Color(1.0,1.0,1.0), 0));
    scene->set_background_ptr( new LinearBackground(
                               Color(0.2, 0.4, 0.9),
                               Color(0.0,0.0,0.0),
                               Vector(1, 0, 0)) );

    scene->select_shadow_mode( No_Shadows );
    scene->attach_display(dpy);
    if (cut)
      scene->attach_display(pd);
    scene->addObjectOfInterest(obj, true);
    return scene;
}


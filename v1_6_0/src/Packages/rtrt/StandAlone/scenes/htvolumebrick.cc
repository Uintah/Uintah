#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Cylinder.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/HVolumeBrick.h>
#include <Packages/rtrt/Core/HTVolumeBrick.h>
#include <Packages/rtrt/Core/HVolumeBrick16.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Core/Thread/Thread.h>
#include <fstream>
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
    char* file=0;

    double density=1.0;
    double camerax=6800, cameray=2150, cameraz=4000;
    double lookatx=0, lookaty=0, lookatz=320;
    double upx=-0.5, upy=0, upz=1;
    double foview=5;
    double light_point_x = 50, light_point_y = -30, light_point_z = 30;
    float isoval;
    bool use_iso=false;
    bool use_tri_file=false;
    char *tri_file;
    char *pts_file;
    char *tri1_file;
    char *pts1_file;
    char *tri2_file;
    char *pts2_file;

    for(int i=1;i<argc;i++){
	if(strcmp(argv[i], "-depth")==0){
	    i++;
	    depth=atoi(argv[i]);
        } else if(strcmp(argv[i], "-density")==0) {
            i++;
            density=atof(argv[i]);
        } else if(strcmp(argv[i], "-camera")==0 || strcmp(argv[i], "-eye")==0) {
            i++;
            camerax = atof(argv[i]); i++;
            cameray = atof(argv[i]); i++;
            cameraz = atof(argv[i]);
        } else if(strcmp(argv[i], "-lookat")==0) {
            i++;
            lookatx = atof(argv[i]); i++;
            lookaty = atof(argv[i]); i++;
            lookatz = atof(argv[i]);
        } else if(strcmp(argv[i], "-up")==0) {
            i++;
            upx = atof(argv[i]); i++;
            upy = atof(argv[i]); i++;
            upz = atof(argv[i]);
        } else if(strcmp(argv[i], "-fov")==0) {
            i++;
            foview = atof(argv[i]);
        } else if(strcmp(argv[i], "-iso")==0) {
            i++;
            isoval = atof(argv[i]);
            use_iso = true;
        } else if(strcmp(argv[i], "-tri")==0) {
            i++;
            tri_file = argv[i++];
            pts_file = argv[i++];
            tri1_file = argv[i++];
            pts1_file = argv[i++];
            tri2_file = argv[i++];
            pts2_file = argv[i];
            use_tri_file = true;
        } else if(strcmp(argv[i], "-light")==0) {
            i++;
            light_point_x = atof(argv[i]); i++;
            light_point_y = atof(argv[i]); i++;
            light_point_z = atof(argv[i]);
	} else if(!file){
	    file=argv[i];
	} else {
	    cerr << "Unknown option: " << argv[i] << '\n';
	    cerr << "Valid options for scene: " << argv[0] << '\n';
	    cerr << " -depth n   - set depth of hierarchy\n";
	    cerr << " file       - raw file name\n";
	    return 0;
	}
    }


    Camera cam(Point(camerax,cameray,cameraz),
               Point(lookatx,lookaty,lookatz),
               Vector(upx,upy,upz), foview);

    Material* matl0=new Phong(Color(.0,.6,.6),
                              Color(.6,.6,.6),
                              100, 0);

    VolumeDpy* dpy;
    if (use_iso) {
        dpy=new VolumeDpy(isoval);
    } else {
        dpy=new VolumeDpy();
    }
    
    HTVolumeBrick* htvol=new HTVolumeBrick(matl0, dpy, file, depth,
                                           nworkers, density);
    (new Thread(dpy, "HTVolume GUI thread"))->detach();

    Object* obj;

    if(use_tri_file) {
        Group *group=new Group();
        obj=group;

        Material* matl1=new Phong(Color(.6,.2,.1), Color(.6,.2,.1), 100, 0);

        group->add(htvol);

        // add all of the triangles to the group...
        ifstream tri_in(tri_file);
        if(!tri_in){
            cerr << "Error reading triangle file: " << tri_file << '\n';
            exit(1);
        }

        ifstream in_pts(pts_file);
        if(!in_pts){
            cerr << "Error opening points file: " << pts_file << '\n';
            exit(1);
        }
        int npts;
        in_pts >> npts;
        float *points=new float[3*npts];

        int i;

        for(i=0; i < 3*npts; i++) {
            in_pts >> points[i];
        }

        for(;;) {
            // read the points
            int v1, v2, v3;
            tri_in >> v1 >> v2 >> v3;
            if(!tri_in) break;
            float x1=points[3*(v1-1)];
            float y1=points[3*(v1-1)+1];
            float z1=points[3*(v1-1)+2];
            float x2=points[3*(v2-1)];
            float y2=points[3*(v2-1)+1];
            float z2=points[3*(v2-1)+2];
            float x3=points[3*(v3-1)];
            float y3=points[3*(v3-1)+1];
            float z3=points[3*(v3-1)+2];

            // make a new triangle, and add it to the group
            group->add(new Tri(matl1,
                               Point(x1, y1, z1),
                               Point(x2, y2, z2),
                               Point(x3, y3, z3)));

        }

        delete points;

//        Material* matl3=new DielectricMaterial(1.0, 1.0, .54, 400.0,
//                                               Color(.93,.93,.00),
//                                               Color(1,1,1));
        Material* matl2=new PhongMaterial(Color(1,1,0),.10);

        // add all of the triangles to the group...
        ifstream tri1_in(tri1_file);
        if(!tri1_in){
            cerr << "Error reading triangle file: " << tri1_file << '\n';
            exit(1);
        }

        ifstream in1_pts(pts1_file);
        if(!in1_pts){
            cerr << "Error opening points file: " << pts1_file << '\n';
            exit(1);
        }
        in1_pts >> npts;
        points=new float[3*npts];

        for(i=0; i < 3*npts; i++) {
            in1_pts >> points[i];
        }

        for(;;) {
            // read the points
            int v1, v2, v3;
            tri1_in >> v1 >> v2 >> v3;
            if(!tri1_in) break;
            float x1=points[3*(v1-1)];
            float y1=points[3*(v1-1)+1];
            float z1=points[3*(v1-1)+2];
            float x2=points[3*(v2-1)];
            float y2=points[3*(v2-1)+1];
            float z2=points[3*(v2-1)+2];
            float x3=points[3*(v3-1)];
            float y3=points[3*(v3-1)+1];
            float z3=points[3*(v3-1)+2];

            // make a new triangle, and add it to the group
            group->add(new Tri(matl2,
                               Point(x1, y1, z1),
                               Point(x2, y2, z2),
                               Point(x3, y3, z3)));

        }

        delete points;

        // add all of the triangles to the group...
        ifstream tri2_in(tri2_file);
        if(!tri2_in){
            cerr << "Error reading triangle file: " << tri2_file << '\n';
            exit(1);
        }

        ifstream in2_pts(pts2_file);
        if(!in2_pts){
            cerr << "Error opening points file: " << pts2_file << '\n';
            exit(1);
        }
        in2_pts >> npts;
        points=new float[3*npts];

        for(int i=0; i < 3*npts; i++) {
            in2_pts >> points[i];
        }

        for(;;) {
            // read the points
            int v1, v2, v3;
            tri2_in >> v1 >> v2 >> v3;
            if(!tri2_in) break;
            float x1=points[3*(v1-1)];
            float y1=points[3*(v1-1)+1];
            float z1=points[3*(v1-1)+2];
            float x2=points[3*(v2-1)];
            float y2=points[3*(v2-1)+1];
            float z2=points[3*(v2-1)+2];
            float x3=points[3*(v3-1)];
            float y3=points[3*(v3-1)+1];
            float z3=points[3*(v3-1)+2];

            // make a new triangle, and add it to the group
            group->add(new Tri(matl2,
                               Point(x1, y1, z1),
                               Point(x2, y2, z2),
                               Point(x3, y3, z3)));

        }

        delete points;


    } else {
        obj=htvol;
    }

    //double bgscale=0.5;
    double ambient_scale=.5;

    Color bgcolor(0.01, 0.05, 0.3);
    Color cup(1, 0, 0);
    Color cdown(0, 0, 0.2);

    rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, -1, 0) );
    Scene* scene=new Scene(obj, cam,
			   bgcolor, cdown, cup, groundplane,
			   ambient_scale);
    scene->add_light(new Light(Point(light_point_x, light_point_y, light_point_z),
                     Color(1.0,1.0,1.0), 0));
    scene->set_background_ptr( new LinearBackground(
                               Color(0.2, 0.4, 0.9),
                               Color(0.0,0.0,0.0),
                               Vector(1, 0, 0)) );

    scene->select_shadow_mode( No_Shadows );
    scene->attach_display(dpy);
    return scene;
}


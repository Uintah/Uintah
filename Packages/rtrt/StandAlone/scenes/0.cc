

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/CutPlane.h>
#include <Packages/rtrt/Core/PlaneDpy.h>
#include <Core/Thread/Thread.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/Checker.h>

#include <iostream>

#include <math.h>
#include <string.h>

using namespace rtrt;
using namespace std;

using SCIRun::Thread;

static const double SCALE = 1./3.;
static const double BV_RADIUS = 1.0;

static void create_dirs(Vector* objset)
{
    double dist=1./sqrt(2.0);
    Vector dir[3];
    dir[0]=Vector(dist, dist, 0);
    dir[1]=Vector(dist, 0, -dist);
    dir[2]=Vector(0, dist, -dist);

    Vector axis(1, -1, 0);
    axis.normalize();

    double rot=asin(2.0/sqrt(6.0));
    Transform t;
    t.load_identity();
    t.post_rotate(rot, axis);

    for(int n=0;n<3;n++){
        dir[n]=t.project(dir[n]);
    }

    for(int ns=0;ns<3;ns++){
	Transform t;
	t.load_identity();
	t.post_rotate(ns*2.*M_PI/3., Vector(0,0,1));
        for(int nv=0;nv<3;nv++){
            objset[ns*3+nv]=t.project(dir[nv]);
        }
    }
}

static void create_objs(Group* group, const Point& center,
		 double radius, const Vector& dir, int depth,
                 Vector* objset, Material* matl)
{
    group->add(new Sphere(matl, center, radius));

    // Check if children should be generated
    if(depth > 0){
        depth--;

        // Rotation matrix to new axis from +Z axis
        Transform mx;
	mx.load_identity();
	mx.rotate(Vector(0,0,1), dir);

        double scale = radius * (1+SCALE);

        for(int n=0;n<9;n++){
            Vector child_vec(mx.project(objset[n]));
            Point child_pt(center+child_vec*scale);
            double child_rad=radius*SCALE; Vector child_dir = child_pt-center;
            child_dir *= 1./scale;
            create_objs(group, child_pt, child_rad, child_dir, depth, objset, matl);
        }
    }
}

static void make_box(Group* group, Material* matl,
	      const Point& corner, const Vector& x, const Vector& y, const Vector& z)
{
  //Group *boxgroup = new Group();
  group->add(new Rect(matl, corner+x+z, x, z));
  group->add(new Rect(matl, corner+x+y*2+z, z, x));
  group->add(new Rect(matl, corner+y+z, y, z));
  group->add(new Rect(matl, corner+x*2+y+z, z, y));
  group->add(new Rect(matl, corner+x+y+z*2, x, y));
  //group->add(boxgroup);

  //CutPlane *cut = new CutPlane(boxgroup, Point(0,0,0), Vector(1,0,0));
  

  //PlaneDpy* pdpy=new PlaneDpy(Vector(1,0,0), Point(0,0,0));
  //CutPlane *cut = new CutPlane(boxgroup, pdpy);
  //new Thread(pdpy, "CutPlane GUI thread");
  
  //group->add(cut);
}

static Object* make_obj(int size)
{
    Group* world=new Group();
    Vector objset[9];
    create_dirs(objset);
    Material* matl0=new LambertianMaterial (Color(.4,.4,.4));
    //Material* matl0=new Phong(Color(0.32,0.32,0.32), Color(0.32,0.32,0.32), Color(0.32,0.32,0.32), 40, 0);
    create_objs(world, Point(0,0,.5), BV_RADIUS/2.0, Vector(0,0,1),
		size, objset, matl0);

    Vector diag1(1,1,0);
    diag1.normalize();
    Vector diag2(-1,1,0);
    diag2.normalize();
    Material* matl1=new LambertianMaterial (Color(.2,.4,.2));
    diag1*=1.5;
    diag2*=1.5;
    Vector z(0,0,.4);
    Point corner(-1.8,-.3,0);
    make_box(world, matl1, corner, diag1, diag2, z);

    Material* matl3=new MetalMaterial( Color(.7,.7,.7));
    world->add(new Sphere(matl3, corner+diag1*1.25+diag2*.6+z*2+Vector(0,0,.6), .6));
#if 1
    Material* matl2=new Checker(new LambertianMaterial(Color(.95,.95,.95)),
                                new LambertianMaterial(Color(.7,.3,.3)),
				Vector(1,1.1,0), Vector(-1.1,1,0));
/*
    Material* matl2=new Checker(new Phong(Color(.05,.05,.05), Color(.95,.95,.95), Color(.6,.6,.6), 10),
				new Phong(Color(.05,.0,0), Color(.7,.3,.3), Color(.6,.6,.6), 10),
				Vector(1,1.1,0), Vector(-1.1,1,0));
*/
#else
    Material* matl2=new Phong(Color(.05,.05,.05), Color(.95,.95,.95), Color(.6,.6,.6), 10);
#endif
    double planesize=15;
    Object* obj1=new Rect(matl2, Point(0,0,0), Vector(planesize,planesize*1.1,0), Vector(-planesize*1.1,planesize,0));
    world->add(obj1);
    return world;
}

extern "C" 
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
    int scenesize=2;
    double light_radius=0.8;
    for(int i=1;i<argc;i++){
	if(strcmp(argv[i], "-size")==0){
	    i++;
	    scenesize=atoi(argv[i]);
	} else if(strcmp(argv[i], "-light")==0){
	    i++;
	    light_radius=atof(argv[i]);
	} else {
	    cerr << "Unknown option: " << argv[i] << '\n';
	    cerr << "Valid options for scene: " << argv[0] << '\n';
	    cerr << " -size n   - Sets depth of sphereflake\n";
	    cerr << " -light r  - Sets radius of light source for soft shadows\n";
	    return 0;
	}
    }

    Camera cam(Point(1.8,-5.53,1.25), Point(0.0,-.13,1.22),
                        Vector(0,0,1), 28.2);

    Object* obj=make_obj(scenesize);
    double ambient_scale=1.0;
    Color bgcolor(0.1, 0.2, 0.45);
    Color cdown(0.82, 0.62, 0.62);
    Color cup(0.1, 0.3, 0.8);


    rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 3) );
    Scene* scene=new Scene(obj, cam,
			   bgcolor, cdown, cup, groundplane,
			   ambient_scale, Arc_Ambient);
    scene->add_light(new Light(Point(5,-3,3), Color(1,1,.8)*2, light_radius));

    scene->set_background_ptr( new LinearBackground(
                               Color(0.2, 0.4, 0.9),
                               Color(0.0,0.0,0.0),
                               Vector(0,0,1)) );


    scene->select_shadow_mode( Single_Soft_Shadow );
    return scene;
}

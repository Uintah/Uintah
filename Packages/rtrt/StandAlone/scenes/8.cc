

#include "Camera.h"
#include "Light.h"
#include "Scene.h"
#include "Point.h"
#include "Vector.h"
#include <Core/Geometry/Transform.h>
#include "Group.h"
#include "Sphere.h"
#include "Rect.h"
#include "Phong.h"
#include "MetalMaterial.h"
#include "LambertianMaterial.h"
#include "CoupledMaterial.h"
#include "DielectricMaterial.h"
#include "Checker.h"
#include "Box.h"
#include <iostream>
#include <math.h>
#include "string.h"

using namespace rtrt;

static void create_objs(Group* group, const Vector& offset,
		 int size, Material* mat)
{
     for (int i = 0; i < size; i++) {
          double x = 10*drand48()*offset.x();
          double y = 10*drand48()*offset.y();
          double z = 10*drand48()*offset.z();
          Point point(x, y, z);
          group->add( new Box(mat, point, point+offset) );
     }

}

static Object* make_obj(int size)
{
    Group* world=new Group();
    Material* matl0=new LambertianMaterial (Color(.4,.4,.4));

    Material* matl1= new DielectricMaterial(1.5, 1.0, 0.04, 100.0, Color(.75, .98 , .93), Color(1,1,1));
    create_objs(world, Vector(0.3, 0.3, 0.05), size, matl1);


	#if 1
	    Material* matl2=new Checker(new Phong(Color(.05,.05,.05), Color(.95,.95,.95), Color(.6,.6,.6), 10),
					new Phong(Color(.05,.0,0), Color(.7,.3,.3), Color(.6,.6,.6), 10),
				Vector(1,1.1,0), Vector(-1.1,1,0));
#else
    Material* matl2=new Phong(Color(.05,.05,.05), Color(.95,.95,.95), Color(.6,.6,.6), 10);
#endif
    double planesize=15;
    Object* obj1=new Rect(matl2, Point(0,0,0), Vector(planesize,planesize*1.1,0), Vector(-planesize*1.1,planesize,0));
    world->add(obj1);
    return world;
}

extern "C" 
Scene* make_scene(int argc, char* argv[])
{
    int scenesize=100;
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
    Color cdown(0.82, 0.62, 0.62);
    Color cup(0.1, 0.3, 0.8);

    Color bgcolor(0.3, 0.5, 0.9);

    Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 6) );

    Scene* scene=new Scene(obj, cam,
                           bgcolor, cdown, cup, groundplane,
                           ambient_scale, Arc_Ambient);
    scene->add_light(new Light(Point(50,-30,30), Color(1,1,.8), 10*light_radius));
    scene->set_background_ptr( new LinearBackground(
                               Color(0.2, 0.4, 0.9),
                               Color(0.0,0.0,0.0),
                               Vector(0,0,1)) );

    scene->shadow_mode=1;
    return scene;
}



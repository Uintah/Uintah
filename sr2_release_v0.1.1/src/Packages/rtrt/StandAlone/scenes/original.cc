
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Checker.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <iostream>
#include <math.h>
#include <string.h>

// This is a benchmark scene.  Please do not modify it... - Steve

using namespace rtrt;
using namespace std;

extern "C" 
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
    for(int i=1;i<argc;i++){
	cerr << "Unknown option: " << argv[i] << '\n';
	cerr << "Valid options for scene: " << argv[0] << '\n';
	return 0;
    }

    Camera cam(Point(3,3,2), Point(0,0,.3),
		 Vector(0,0,1), 60.0);

    double bgscale=0.5;
    double ambient_scale=.5;

    Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);

    Material* white=new Phong(Color(.6,.6,.6), Color(.6,.6,.6), 20, 0.2);
    Material* black=new Phong(Color(0,0,0), Color(.6,.6,.6), 20, 0.5);
    Material* plane_matl=new Checker(white, black, Vector(1,0,0), Vector(0,1,0));
    Object* p=new Rect(plane_matl, Point(0,0,0), Vector(20,0,0), Vector(0,20,0));
    Group* group = new Group();
    group->add(p);
    Material* red=new Phong(Color(.6,0,0), Color(.6,.6,.6), 20, 0.4);
    Sphere* s=new Sphere(red, Point(0,0,1.2),1.0);
    group->add(s);

    Plane groundplane ( Point(0, 0, 0), Vector(1, 0, 0) );
    Scene* scene=new Scene(group, cam,
			   bgcolor, Color(0,0,0), bgcolor, groundplane,
			   ambient_scale);
    Light* l1=new Light(Point(0,5,8), Color(.6,.1,.1), 0);
    Light* l2=new Light(Point(5,0,8), Color(.1,.6,.1), 0);
    Light* l3=new Light(Point(5,5,2), Color(.2,.2,.2), 0);
    scene->add_light(l1);
    scene->add_light(l2);
    scene->add_light(l3);

    scene->select_shadow_mode( Uncached_Shadows );
    scene->maxdepth=5;
    return scene;
}

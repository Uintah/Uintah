

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <iostream>
#include <math.h>
#include <string.h>

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

    Camera cam(Point(4,4,4), Point(0,0,0),
	       Vector(0,0,1), 60.0);

    Color groundcolor(.82, .62, .62);
    Color averagelight(1,1,.8);
    double ambient_scale=.5;

    Color bgcolor(.5,0,.5);

    Group* group=new Group();
    Material* matl0=new Phong(Color(.2,.2,.2), Color(.3,.3,.3), 10, 0);
    group->add(new Sphere(matl0, Point(0,0,0), 50));
    Plane groundplane ( Point(0, 0, 0), Vector(1, 0, 0) );
    Scene* scene=new Scene(group, cam,
			   bgcolor, groundcolor*averagelight, bgcolor,  groundplane, 
			   ambient_scale);
    scene->select_shadow_mode( Single_Soft_Shadow );
    return scene;
}

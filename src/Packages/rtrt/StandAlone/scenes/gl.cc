/* Draws a simple stuff to test opengl */


#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Plane.h>
#include <iostream>
#include <math.h>
#include <string.h>

using namespace rtrt;
extern void run_gl_test();

extern "C" 
Scene* make_scene(int /*argc*/, char* /*argv[]*/, int /*nworkers*/)
{
  // don't need to parse arguments
#if 0
    for(int i=1;i<argc;i++){
	cerr << "Unknown option: " << argv[i] << '\n';
	cerr << "Valid options for scene: " << argv[0] << '\n';
	return 0;
    }
#endif

    // here's where we take over the rtrt and do our own thing
    run_gl_test();

    // junk code that won't actually run, but makes the compiler happy
    Camera cam(Point(4,4,4), Point(0,0,0),
	       Vector(0,0,1), 60.0);

    Color groundcolor(.82, .62, .62);
    Color averagelight(1,1,.8);
    double ambient_scale=.5;

    Color bgcolor(.5,0,.5);

    Group* empty=new Group();
    rtrt::Plane groundplane ( Point(0, 0, 0), Vector(1, 0, 0) );
    Scene* scene=new Scene(empty, cam,
			   bgcolor, groundcolor*averagelight, bgcolor, groundplane, 
			   ambient_scale);
    scene->select_shadow_mode( Single_Soft_Shadow );
    return scene;
}

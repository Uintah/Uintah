

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Scene.h>
#include <iostream>
#include <math.h>
#include <string.h>


extern "C" 
Scene* make_scene(int argc, char* argv[])
{
    for(int i=1;i<argc;i++){
	if(strcmp(argv[i], "-option...")==0){
	    ...;
	} else {
	    cerr << "Unknown option: " << argv[i] << '\n';
	    cerr << "Valid options for scene: " << argv[0] << '\n';
	    ...;
	    return 0;
	}
    }

    Camera cam(Point(1,0,0), Point(0,0,0),
	       Vector(0,0,1), 40);
    Object* obj=...;

    double bgscale=0.95;
    Color groundcolor(.82, .62, .62);
    Color averagelight(1,1,.8);
    double ambient_scale=.5;

    Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);

    Plane groundplane ( Point(0, 0, 0), Vector(1, 0, 0) );
    Scene* scene=new Scene(obj, cam,
			   bgcolor, groundcolor, groundplane, averagelight,
			   ambient_scale);
    scene->shadow_mode=1;
    return scene;
}

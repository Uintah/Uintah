

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/MIPHVB16.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdlib.h>

using namespace rtrt;
using namespace std;

extern "C" 
Scene* make_scene(int argc, char* argv[], int nworkers)
{
    int depth=3;
    char* mip16file=0;
    for(int i=1;i<argc;i++){
	if(strcmp(argv[i], "-depth")==0){
	    i++;
	    depth=atoi(argv[i]);
	} else if(!mip16file){
	    mip16file=argv[i];
	} else {
	    cerr << "Unknown option: " << argv[i] << '\n';
	    cerr << "Valid options for scene: " << argv[0] << '\n';
	    cerr << " -depth n   - set depth of hierarchy\n";
	    cerr << " file       - raw file name\n";
	    return 0;
	}
    }

    Camera cam(Point(991.526, 506.642, 179.416),
	       Point(0, 0, 0),
	       Vector(0.47655, -0.8301, -0.28954),
	       42.566);

    Object* obj=new MIPHVB16(mip16file, depth, nworkers);

    double bgscale=0.5;
    Color groundcolor(0,0,0);
    //Color averagelight(0,0,0);
    double ambient_scale=.5;

    Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);

    Plane groundplane ( Point(0, 0, 0), Vector(1, 0, 0) );
    Scene* scene=new Scene(obj, cam,
			   bgcolor, groundcolor, bgcolor, groundplane,
			   ambient_scale);
    scene->select_shadow_mode( No_Shadows );
    return scene;
}


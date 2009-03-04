/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/




#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/MIPHVB16.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>

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


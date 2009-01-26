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


/* Draws a simple stuff to test opengl */


#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Plane.h>
#include <iostream>
#include <cmath>
#include <cstring>

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

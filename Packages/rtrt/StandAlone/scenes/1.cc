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



#include <Packages/rtrt/Core/BouncingSphere.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Checker.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/Scene.h>
#include <iostream>
#include <cmath>
#include <cstring>

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

    Camera cam(Point(4,4,1.7), Point(0,0,0),
		 Vector(0,0,1), 60.0);

    double bgscale=0.5;
    double ambient_scale=.5;

    Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);

    Material* matl0  = new Phong(Color(.2,.2,.2), Color(.3,.3,.3), 100, .5);
    Material* matl00 = new Phong(Color(.2,.2,.2), Color(.3,.3,.3), 10, 0);
    Material* matl1  = 
      new Checker( new Phong(Color(.2,.2,.5), Color(.1,.1,.1), 0, .1),
		   new Phong(Color(.2,.2,.2), Color(.1,.1,.1), 0, .1),
		   Vector(1,1.1,0), Vector(-1.1,1,0));
    Object* obj1=new Rect(matl1, Point(0,0,0), Vector(20,0,0), Vector(0,20,0));
    
    Group* group = new Group();
    group->add(obj1);
    group->add(new BouncingSphere(matl00, Point(0,0,1.5), .5, Vector(0,0,1.2)));
    group->add(new BouncingSphere(matl0, Point(0,0,2.5), .5, Vector(0,0,1.4)));
    group->add(new BouncingSphere(matl00, Point(0,0,3.5), .5, Vector(0,0,1.6)));
    group->add(new BouncingSphere(matl0, Point(0,0,.5), .5, Vector(0,0,1)));

    Plane groundplane ( Point(0, 0, 0), Vector(1, 0, 0) );
    Scene* scene=new Scene(group, cam,
			   bgcolor, Color(0,0,0), bgcolor, groundplane,
			   ambient_scale);
    scene->select_shadow_mode(Hard_Shadows);
    return scene;
}

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
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/CrowMarble.h>
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

    Camera cam(Point(4,4,4), Point(0,0,0),
	       Vector(0,0,1), 60.0);

    Color groundcolor(.5,.3,.2);
    Color averagelight(1,1,.8);
    double ambient_scale=.5;

    Color bgcolor(.2,.2,.2);
    Group* group=new Group();
    Material* matl0=new Phong(Color(.2,.2,.2), Color(.3,.3,.3), 10, .5);
    Material* matl2=new Phong(Color(.05,.05,.05), Color(.05,.05,.05), 10, .05);
    Material* matl1=new CrowMarble(10, Vector(1,1,1), Color(0.5,0.5,0.5), Color(0.4,0.2, 0.1),Color(0.1,0.05,0.1));
    Object* obj1=new Rect(matl1, Point(0,0,0), Vector(6,0,0), Vector(0,6,0));

    group->add(obj1);
    group->add(new BouncingSphere(matl0, Point(0,0,.4), .5, Vector(0,0,1)));
    group->add(new BouncingSphere(matl2, Point(0,0,1.4), .5, Vector(0,0,1.2)));
    group->add(new BouncingSphere(matl0, Point(0,0,2.4), .5, Vector(0,0,1.4)));
    group->add(new BouncingSphere(matl2, Point(0,0,3.4), .5, Vector(0,0,1.6)));

    Plane groundplane ( Point(0, 0, 0), Vector(1, 0, 0) );
    Scene* scene=new Scene(group, cam,
			   bgcolor, groundcolor*averagelight, bgcolor, groundplane,
			   ambient_scale);
    scene->select_shadow_mode(Single_Soft_Shadow);
    return scene;
}

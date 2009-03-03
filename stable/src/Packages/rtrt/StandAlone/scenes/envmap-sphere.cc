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


//
// This file contains a simple scene suitable for ray tracing
// on 1 processor.
//
// It contains one sphere and a "ground" and a ring.
//

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Ring.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/Checker.h>
#include <iostream>
#include <cmath>
#include <cstring>

using namespace rtrt;

extern "C" 
Scene* make_scene(int /*argc*/, char* /*argv*/[], int /*nworkers*/)
{
  Camera cam( Point(0,3,0), Point( 0,0,0 ), Vector(0,0,1), 45.0 );

  Material* silver=new MetalMaterial( Color( 1,1,1 ), 30 );

  Object* sphere = new Sphere( silver, Point(0,0,0), 1 );

  Group * group = new Group();
  group->add( sphere );

  double ambient_scale=1.0;
  Color bgcolor(0.3, 0.3, 0.3);
  Color cdown(0.6, 0.4, 0.4);
  Color cup(0.4, 0.4, 0.6);

  rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 1) );
  Scene* scene=new Scene(group, cam,
			 bgcolor, cdown, cup, groundplane,
			 ambient_scale, Arc_Ambient);

  Light * red = new Light(Point(100, 100, 100), Color(.9,.8,.8), 0);
  red->name_ = "red";
  scene->add_light( red );
  Light * green = new Light(Point(100, -100, 100), Color(.8,.9,.8), 0);
  green->name_ = "green";
  scene->add_light( green );
  Light * blue = new Light(Point(-100, 0, 100), Color(.8,.8,.9), 0);
  blue->name_ = "blue";
  scene->add_light( blue );
  Light * purple = new Light(Point(0, 0, -100), Color(.9,.8,.9), 0);
  purple->name_ = "purple";
  scene->add_light( purple );

  scene->select_shadow_mode( Hard_Shadows );
  return scene;
}


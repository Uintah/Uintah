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



#include <Packages/rtrt/Core/BumpObject.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Stats.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace rtrt;
using namespace std;
using namespace SCIRun;


BumpObject::BumpObject(Vector &vector, UVMapping* uv)
  : Object(NULL, uv) {
  norm = vector;
}


BumpObject::~BumpObject()
{
}

void BumpObject::intersect(Ray& /*ray*/, HitInfo& /*hit*/, 
			   DepthStats* /*st*/,
			   PerProcessorContext*) {
  cout << "If you're interesecting this object - stop it. This is the Bump Object and should only be used for its normal." << endl;
}



/*
void BumpObject::light_intersect(Light* light, Ray& ray, HitInfo&,
			     double dist, Color& atten, DepthStats* st,
			     PerProcessorContext*)
{

  cout << "If you're light interesecting this object - stop it. This is the Bump Object and should only be used for its normal." << endl;
}
*/

void BumpObject::multi_light_intersect(Light*, const Point& /*orig*/,
				       const Array1<Vector>& /*dirs*/,
				       const Array1<Color>& /*attens*/,
				       double,
				       DepthStats*, PerProcessorContext*) {
  cout << "If you're multi light interesecting this object - stop it. This is the Bump Object and should only be used for its normal." << endl;
}

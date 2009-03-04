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



#include <Packages/rtrt/Core/Checker.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/HitInfo.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* checker_maker() {
  return new Checker();
}

// initialize the static member type_id
PersistentTypeID Checker::type_id("Checker", "Material", checker_maker);


Checker::Checker(Material* matl0, Material* matl1,
		 const Vector& u, const Vector& v)
    : matl0(matl0), matl1(matl1), u(u), v(v)
{
}

Checker::~Checker()
{
}

void Checker::shade(Color& result, const Ray& ray,
		    const HitInfo& hit, int depth, 
		    double atten, const Color& accumcolor,
		    Context* cx)
{
    Point p(ray.origin()+ray.direction()*hit.min_t);
    double uu=Dot(u, p);
    double vv=Dot(v, p);
    if(uu<0)
	uu=-uu+1;
    if(vv<0)
	vv=-vv+1;
    int i=(int)uu;
    int j=(int)vv;
    int m=(i+j)%2;
    (m==0?matl0:matl1)->shade(result, ray, hit, depth, atten,
			      accumcolor, cx);
}

const int CHECKER_VERSION = 1;

void 
Checker::io(SCIRun::Piostream &str)
{
  str.begin_class("Checker", CHECKER_VERSION);
  Material::io(str);
  SCIRun::Pio(str, matl0);
  SCIRun::Pio(str, matl1);
  SCIRun::Pio(str, u);
  SCIRun::Pio(str, v);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Checker*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Checker::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Checker*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun

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


#include <Packages/rtrt/Core/TimeCycleMaterial.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Context.h>
#include <cmath>
#include <Core/Thread/Time.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* timeCycleMaterial_maker() {
  return new TimeCycleMaterial();
}

// initialize the static member type_id
PersistentTypeID TimeCycleMaterial::type_id("TimeCycleMaterial", "Material", 
					    timeCycleMaterial_maker);

TimeCycleMaterial::TimeCycleMaterial( void )
    : CycleMaterial(),
      cur_time_( 0.0 )
{
    time_array_.initialize( 0.0 );
    time_ = SCIRun::Time::currentSeconds();
}

TimeCycleMaterial::~TimeCycleMaterial( void )
{
}

void 
TimeCycleMaterial::add( Material* mat, double time ) 
{
    members.add( mat );
    time_array_.add( time );
    cur_time_ = time_array_[0];
}

void 
TimeCycleMaterial::shade( Color& result, const Ray& ray,
			  const HitInfo& hit, int depth, 
			  double atten, const Color& accumcolor,
			  Context* cx )
{
    double etime = SCIRun::Time::currentSeconds() - time_;
    
    if( etime > cur_time_ ) {
	next();
	time_ += cur_time_;
	cur_time_ = time_array_[current];
    }
    members[current]->shade(result, ray, hit, depth, atten, accumcolor, cx);
}

const int TIMECYCLEMATERIAL_VERSION = 1;

void 
TimeCycleMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("TimeCycleMaterial", TIMECYCLEMATERIAL_VERSION);
  Material::io(str);
  SCIRun::Pio(str, time_array_);
  SCIRun::Pio(str, time_);
  SCIRun::Pio(str, cur_time_);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::TimeCycleMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::TimeCycleMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::TimeCycleMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun

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



#ifndef SHADOWBASE_H
#define SHADOWBASE_H

#include <Core/Persistent/Persistent.h>

namespace SCIRun {
  class Point;
  class Vector;
}

namespace rtrt {

using SCIRun::Point;
using SCIRun::Vector;

class Light;
class Color;
class Context;
class Scene;
class ShadowBase;


enum ShadowType { No_Shadows = 0, Single_Soft_Shadow, Hard_Shadows,
		  Glass_Shadows, Soft_Shadows, Uncached_Shadows };
}

namespace SCIRun {
void Pio(Piostream&, rtrt::ShadowBase*&);
}

namespace rtrt {

class ShadowBase : public SCIRun::Persistent {
  const char* name;
public:
  ShadowBase();
  virtual ~ShadowBase();

  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, rtrt::ShadowBase*&);
  
  virtual void preprocess(Scene* scene, int& pp_offset, int& scratchsize);
  virtual bool lit(const Point& hitpos, Light* light,
		   const Vector& light_dir, double dist, Color& atten,
		   int depth, Context* cx) = 0;
  void setName(const char* name) {
    this->name=name;
  }
  const char* getName() {
    return name;
  }

  static char * shadowTypeNames[];

  static int increment_shadow_type(int shadow_type);
  static int decrement_shadow_type(int shadow_type);

};

} // end namespace rtrt

#endif


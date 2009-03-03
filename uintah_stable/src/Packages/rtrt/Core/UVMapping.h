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



#ifndef UVMAPPING_H
#define UVMAPPING_H 1


#include <Core/Persistent/Persistent.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/HitInfo.h>

namespace rtrt {
class UVMapping;
}
namespace SCIRun {
class Point;
class Vector;
void Pio(Piostream&, rtrt::UVMapping*&);
}

namespace rtrt {

using SCIRun::Point;
using SCIRun::Vector;

class HitInfo;
class UV;

class UVMapping : public virtual SCIRun::Persistent {
public:
  UVMapping();
  virtual ~UVMapping();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, UVMapping*&);

  virtual void uv(UV& uv, const Point&, const HitInfo& hit)=0;
  virtual void get_frame(const Point &, const HitInfo&,const Vector &norm,  
			 Vector &v2, Vector &v3);
};

} // end namespace rtrt

#endif

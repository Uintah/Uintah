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


/*
Name:		James Bigler, Christiaan Gribble
Location:	University of Utah
Email:		bigler@cs.utah.edu; cgribble@cs.utah.edu
*/

#ifndef RTRT_UVSPHERE2_H
#define RTRT_UVSPHERE2_H 1

#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/UV.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace rtrt {
class UVSphere2;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::UVSphere2*&);
}

namespace rtrt {

class UVSphere2 : public Sphere, public UVMapping {

 protected:

  inline double _DET2(const Vector &v0, const Vector &v1, int i0, int i1) {
    return (v0[i0] * v1[i1] + v0[i1] * -v1[i0]);
  }

  inline void VXV3(Vector &to, const Vector &v1, const Vector &v2) {
    to[0] =  _DET2(v1,v2, 1,2);
    to[1] = -_DET2(v1,v2, 0,2);
    to[2] =  _DET2(v1,v2, 0,1);
  }
  
 public:
  UVSphere2(Material *m, const Point &center, double radius);
  UVSphere2() {} // for Pio.
  virtual ~UVSphere2();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, UVSphere2*&);

  virtual void uv(UV& uv, const Point&, const HitInfo& hit);
  virtual void get_frame(const Point &hitpos, const HitInfo&hit,
                         const Vector &norm,  Vector &pu, Vector &pv);
};
 

} // end namespacertrt
#endif



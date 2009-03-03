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


#ifndef DYNAMICINSTANCE_H
#define DYNAMICINSTANCE_H

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Instance.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/InstanceWrapperObject.h>

namespace rtrt {
  class DynamicInstance;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::DynamicInstance*&);
}

namespace rtrt {

class Camera;
class Stealth;

class DynamicInstance : public Instance
{
public:
  
  DynamicInstance( InstanceWrapperObject * obj,
		   Transform             * trans, 
		   const Vector          & location );
  ~DynamicInstance();
  
  DynamicInstance() : Instance() {} // for Pio.
  
  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, DynamicInstance*&);

  // Places the newTransform into the currentTransform.
  // ** should we update the bounding box / verify the transform is good? **
  inline void useNewTransform()
  {
    Transform * temp = currentTransform;
    currentTransform = newTransform;
    newTransform     = temp;
    o->compute_bounds(bbox,1E-5);
    bbox.transform_inplace(currentTransform);
  }
	    
  void updatePosition( const Stealth * stealth, const Camera * cam );
  void updateNewTransform( const float trans[4][4],
			   Transform & viewpoint );

  // Pointer to some generic original transform that may be used for may
  // instances.
  Transform * origTransform;

  // Pointers to Transforms, "new'ed" by this object
  Transform * newTransform;

  // location to move the center of the objects model to.
  Vector      location_; 

}; // end class DynamiceInstance

} // end namespace rtrt

#endif

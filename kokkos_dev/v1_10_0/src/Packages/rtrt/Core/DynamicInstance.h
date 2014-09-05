#ifndef DYNAMICINSTANCE_H
#define DYNAMICINSTANCE_H

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Instance.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
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

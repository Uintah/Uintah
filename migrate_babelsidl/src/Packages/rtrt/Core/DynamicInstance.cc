
#include <Packages/rtrt/Core/DynamicInstance.h>

#include <Packages/rtrt/Core/Stealth.h>
#include <Packages/rtrt/Core/Camera.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* dynamicinstance_maker() {
  return new DynamicInstance();
}

// initialize the static member type_id
PersistentTypeID DynamicInstance::type_id("DynamicInstance", "Instance", 
					  dynamicinstance_maker);

DynamicInstance::DynamicInstance(InstanceWrapperObject * obj,
				 Transform * trans,
				 const Vector & location ) :
  Instance(obj, trans),
  origTransform(trans),
  location_(location)
{
  currentTransform = new Transform(*trans); // Parent's variable.
  newTransform     = new Transform(*trans);

  bbox.extend( Point(0,0,0), 10 );
}

DynamicInstance::~DynamicInstance()
{
}

void
DynamicInstance::updateNewTransform( const float trans[4][4], 
				     Transform & viewpoint )
{
  double mat[4][4];

  double tempmat[4][4];
  for(int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      mat[i][j]= trans[i][j];
      if(i == j) tempmat[i][j] = 1;
      else       tempmat[i][j] = 0;
    }
  }

  Transform temptrans;
  temptrans.set( (double*)tempmat );

  newTransform->set( (double*)mat );
  newTransform->pre_translate( location_ );
  newTransform->pre_trans( viewpoint );
}

void
DynamicInstance::updatePosition( const Stealth * stealth, const Camera * cam )
{
  double rotational_speed_damper = 100;
  double speed;

  Vector up, side;
  //Vector forward = cam->get_lookat() - cam->get_eye();
  cam->get_viewplane( up, side );

  *newTransform = *origTransform;

  if( ( speed = stealth->getSpeed(3) ) != 0 ) // Pitching
    {
      newTransform->post_rotate( speed/rotational_speed_damper, side );
    }

  if( ( speed = stealth->getSpeed(4) ) != 0 ) // Rotating
    {
      newTransform->post_rotate( -speed/rotational_speed_damper, up );
    }

}

const int DYNAMICINSTANCE_VERSION = 1;

void 
DynamicInstance::io(SCIRun::Piostream &str)
{
  str.begin_class("DynamicInstance", DYNAMICINSTANCE_VERSION);
  Instance::io(str);
  SCIRun::Pio(str, origTransform);
  SCIRun::Pio(str, newTransform);
  SCIRun::Pio(str, location_);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::DynamicInstance*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::DynamicInstance::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::DynamicInstance*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun

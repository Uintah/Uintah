#include <Packages/rtrt/Core/PortalMaterial.h>

using namespace rtrt;
using namespace SCIRun;

Persistent* portalmaterial_maker() {
  return new PortalMaterial();
}

// initialize the static member type_id
PersistentTypeID PortalMaterial::type_id("PortalMaterial", "Material", 
					   portalmaterial_maker);

const int PORTALMATERIAL_VERSION = 1;

void 
PortalMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("PortalMaterial", PORTALMATERIAL_VERSION);
  Material::io(str);
  SCIRun::Pio(str, p_);
  SCIRun::Pio(str, u_);
  SCIRun::Pio(str, v_);
  SCIRun::Pio(str, oe_p_);
  SCIRun::Pio(str, oe_u_);
  SCIRun::Pio(str, oe_v_);
  SCIRun::Pio(str, portal_);
  SCIRun::Pio(str, other_end_);
  SCIRun::Pio(str, attached_);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::PortalMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::PortalMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::PortalMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun











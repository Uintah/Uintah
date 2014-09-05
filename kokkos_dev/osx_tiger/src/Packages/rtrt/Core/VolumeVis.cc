
#include <Packages/rtrt/Core/VolumeVis.h>

namespace rtrt {

// Persistent* vv_maker() {
//   return new VolumeVis();
// }

// // initialize the static member type_id
// PersistentTypeID VolumeVis::type_id("VolumeVis", "Object", vv_maker);



} // end namespace rtrt

// namespace SCIRun {
// void SCIRun::Pio(SCIRun::Piostream& stream, rtrt::VolumeVis*& obj)
// {
//   SCIRun::Persistent* pobj=obj;
//   stream.io(pobj, rtrt::VolumeVis::type_id);
//   if(stream.reading()) {
//     obj=dynamic_cast<rtrt::VolumeVis*>(pobj);
//     ASSERT(obj != 0)
//   }
// }
// } // end namespace SCIRun

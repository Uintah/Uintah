#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/VolumeVis.h>
#include <Packages/rtrt/Core/VolumeVisDpy.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Core/Math/MiscMath.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <float.h>
#include <iostream>

namespace rtrt {
  using namespace std;

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

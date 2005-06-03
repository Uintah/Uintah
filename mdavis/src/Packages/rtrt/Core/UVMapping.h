
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

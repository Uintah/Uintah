
#ifndef UVMAPPING_H
#define UVMAPPING_H 1

namespace SCIRun {
  class Point;
}

namespace rtrt {

  using namespace SCIRun;
  
class HitInfo;
class UV;

class UVMapping {
public:
    UVMapping();
    virtual ~UVMapping();
    virtual void uv(UV& uv, const Point&, const HitInfo& hit)=0;
};

} // end namespace rtrt

#endif

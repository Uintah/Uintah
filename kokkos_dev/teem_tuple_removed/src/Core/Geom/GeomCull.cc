#include <Core/Geom/GeomCull.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;
using std::ostream;

namespace SCIRun {

Persistent* make_GeomCull()
{
  return scinew GeomCull(0,0);
}
  
PersistentTypeID GeomCull::type_id("GeomCull", "GeomObj", make_GeomCull);

GeomCull::GeomCull(GeomHandle child, Vector *normal) :
  GeomContainer(child), normal_(0) 
{
  if (normal) normal_ = scinew Vector(*normal);
}

GeomCull::GeomCull(const GeomCull &copy) :
  GeomContainer(copy), normal_(0) 
{
  if (copy.normal_) normal_ = scinew Vector(*copy.normal_);
}

GeomObj *
GeomCull::clone() 
{
  return scinew GeomCull(*this);
}
  
void
GeomCull::set_normal(Vector *normal) {
  if (normal_) {
    delete normal_;
    normal_ = 0;
  }
  
  if (normal) {
    normal_ = scinew Vector(*normal);
  }
}

void
GeomCull::io(Piostream&stream) {
    stream.begin_class("GeomCull", 1);
    GeomContainer::io(stream); // Do the base class first...
    if (normal_) { 
      Pio(stream,*normal_);
    }
    stream.end_class();
  }

}

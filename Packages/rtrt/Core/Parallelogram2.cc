
#include <Packages/rtrt/Core/Parallelogram2.h>

namespace rtrt {
  extern double _HOLO_STATE_;
}

using namespace rtrt;

void Parallelogram2::animate(double /*t*/, bool& /*changed*/)
{
  MultiMaterial *mat = dynamic_cast<MultiMaterial*>(matl);
  if (mat) {
    mat->set(0,_HOLO_STATE_);
  }
}  


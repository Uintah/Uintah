
#include <Packages/rtrt/Core/Parallelogram2.h>

extern double _HOLO_STATE_;

namespace rtrt {

void Parallelogram2::animate(double t, bool& changed)
{
  MultiMaterial *mat = dynamic_cast<MultiMaterial*>(matl);
  if (mat) {
    mat->set(0,_HOLO_STATE_);
  }
}  

} // end namespace

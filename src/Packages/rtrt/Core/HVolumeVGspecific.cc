
#include <Packages/rtrt/Core/HVolumeVG.h>
#include <limits.h>

using rtrt::VG;

template<>
void VG<unsigned short, unsigned short>::setmin()
{
  v=0;
  g=0;
}

template<>
void VG<unsigned short, unsigned short>::setmax()
{
  v=USHRT_MAX;
  g=USHRT_MAX;
}

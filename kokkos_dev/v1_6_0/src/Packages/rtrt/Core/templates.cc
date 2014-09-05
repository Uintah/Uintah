
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Array2.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Heightfield.h>
#include <Packages/rtrt/Core/BrickArray2.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1468
#endif

using rtrt::Array1;
using rtrt::Array2;
using rtrt::Heightfield;
using rtrt::HMCell; 
using rtrt::BrickArray2;
namespace rtrt {
  class Object;
  class VolumeBase;
  class Volume;
  class Material;
}

template struct HMCell<float>;
template class rtrt::BrickArray2<float>;
template class Array2 < HMCell<float> >;
template class Heightfield<BrickArray2<float>,Array2<HMCell<float > > >;

template class Array1<Light*>;

template class Array1<Object*>;

template class Array1<VolumeBase*>;

template class Array1<Volume*>;

#include <Core/Geometry/Point.h>
template class Array1<SCIRun::Point>;

#include <Core/Geometry/Vector.h>
template class Array1<SCIRun::Vector>;

template class Array1<int>;
template class Array1<double>;

template class Array1<Material*>;
#include <Packages/rtrt/Core/Shadows/ShadowBase.h>
template class Array1<ShadowBase*>;
namespace SCIRun {
template void Pio<rtrt::ShadowBase *>(Piostream &, 
				      rtrt::Array1<rtrt::ShadowBase *> &);
}
#include <Packages/rtrt/Core/Random.h>
#include <Packages/rtrt/Core/HashTable.cc>
#include <Packages/rtrt/Core/HashTableEntry.cc>

template class rtrt::HashTable<rtrt::RandomTable::TableInfo, double*>;
template class rtrt::HashTableEntry<rtrt::RandomTable::TableInfo, double*>;
template class rtrt::HashTable<rtrt::RandomTable::TableInfo, int*>;
template class rtrt::HashTableEntry<rtrt::RandomTable::TableInfo, int*>;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1468
#endif

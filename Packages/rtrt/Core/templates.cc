
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Array2.h>
#include <Packages/rtrt/Core/Heightfield.h>
#include <Packages/rtrt/Core/BrickArray2.h>

using rtrt::Array1;
using rtrt::Array2;
using rtrt::Heightfield;
using rtrt::HMCell;
using rtrt::BrickArray2;

template struct HMCell<float>;
template class BrickArray2<float>;
template class Array2 < HMCell<float> >;
template class rtrt::Heightfield<BrickArray2<float>,Array2<HMCell<float > > >;

class Light;
template class Array1<Light*>;

class Object;
template class Array1<Object*>;

class BoundedObject;
template class Array1<BoundedObject*>;

class VolumeBase;
template class Array1<VolumeBase*>;

class Volume;
template class Array1<Volume*>;

#include <Core/Geometry/Point.h>
template class Array1<SCIRun::Point>;

#include <Core/Geometry/Vector.h>
template class Array1<SCIRun::Vector>;

template class Array1<int>;
template class Array1<double>;

class Material;
template class Array1<Material*>;

#include <Packages/rtrt/Core/Random.h>
#include <Packages/rtrt/Core/HashTable.cc>
#include <Packages/rtrt/Core/HashTableEntry.cc>

template class rtrt::HashTable<rtrt::RandomTable::TableInfo, double*>;
template class rtrt::HashTableEntry<rtrt::RandomTable::TableInfo, double*>;
template class rtrt::HashTable<rtrt::RandomTable::TableInfo, int*>;
template class rtrt::HashTableEntry<rtrt::RandomTable::TableInfo, int*>;





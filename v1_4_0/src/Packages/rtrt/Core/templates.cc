
#include <Packages/rtrt/Core/Array1.cc>

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
template class Array1<Point>;

#include <Core/Geometry/Vector.h>
template class Array1<Vector>;

template class Array1<int>;
template class Array1<double>;

class Material;
template class Array1<Material*>;

#include <Packages/rtrt/Core/Random.h>
#include <Packages/rtrt/Core/HashTable.cc>
#include <Packages/rtrt/Core/HashTableEntry.cc>

template class HashTable<RandomTable::TableInfo, double*>;
template class HashTableEntry<RandomTable::TableInfo, double*>;
template class HashTable<RandomTable::TableInfo, int*>;
template class HashTableEntry<RandomTable::TableInfo, int*>;

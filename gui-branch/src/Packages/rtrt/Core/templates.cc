
#include "Array1.cc"

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

#include "Point.h"
template class Array1<Point>;

#include "Vector.h"
template class Array1<Vector>;

template class Array1<int>;
template class Array1<double>;

class Material;
template class Array1<Material*>;

#include "Random.h"
#include "HashTable.cc"
#include "HashTableEntry.cc"

template class HashTable<RandomTable::TableInfo, double*>;
template class HashTableEntry<RandomTable::TableInfo, double*>;
template class HashTable<RandomTable::TableInfo, int*>;
template class HashTableEntry<RandomTable::TableInfo, int*>;

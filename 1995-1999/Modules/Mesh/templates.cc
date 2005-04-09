
/*
 * Manual template instantiations for g++
 */

#include <Classlib/Array1.cc>
#include <Classlib/HashTable.cc>
#include <Datatypes/SurfacePort.h>

class PointWidget;
template class Array1<PointWidget*>;
class TCLint;
template class Array1<TCLint*>;
class TCLdouble;
template class Array1<TCLdouble*>;
template class Array1<SurfaceIPort*>;
template class Array1<SurfaceHandle>;
class GeomGroup;
template class Array1<GeomGroup*>;

template class HashTable<unsigned long long, int>;
template class HashTable<sci::Edge, int>;
template class HashTableIter<sci::Edge, int>;

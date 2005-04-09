
/*
 * Manual template instantiations for g++
 */

#include <Classlib/Array1.cc>
#include <Classlib/FastHashTable.cc>
#include <Datatypes/Mesh.h>

template class Array1<unsigned char*>;

template class FastHashTable<sci::Face>;
template class FastHashTableIter<sci::Face>;

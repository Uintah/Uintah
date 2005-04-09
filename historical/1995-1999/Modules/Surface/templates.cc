
/*
 * Manual template instantiations for g++
 */

#include <Classlib/Array1.cc>
#include <Classlib/Array2.cc>
#include <Datatypes/ContourSetPort.h>

template class Array1<ContourSetIPort*>;
template class Array1<ContourSetHandle>;
class GeomTrianglesP;
template class Array1<GeomTrianglesP*>;
class GeomTrianglesVP;
template class Array1<GeomTrianglesVP*>;
class GeomTriangles;
template class Array1<GeomTriangles*>;
class GeomLines;
template class Array1<GeomLines*>;

template class Array2<Array1<int> >;

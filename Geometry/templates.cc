
/*
 * Manual template instantiations for g++
 */

#include <Classlib/Array3.cc>
#include <Classlib/Array1.cc>
#include <Geometry/Point.h>

template class Array3<Array1<int>*>;
template class Array1<Point>;

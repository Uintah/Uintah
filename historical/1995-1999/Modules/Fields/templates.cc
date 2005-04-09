
/*
 * Manual template instantiations for g++
 */

#include <Classlib/Array1.cc>
#include <Classlib/Array2.cc>

template class Array1<char>;
class GeomPts;
template class Array1<GeomPts*>;
class PointWidget;
template class Array1<PointWidget*>;

template class Array2<char>;

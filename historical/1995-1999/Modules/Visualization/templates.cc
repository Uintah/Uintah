
/*
 * Manual template instantiations for g++
 */

#include <Classlib/Array1.cc>
#include <Classlib/Array2.cc>
#include <Classlib/FastHashTable.cc>
#include <Classlib/HashTable.cc>
#include <Classlib/Ring.cc>
#include <Datatypes/OctreePort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>
#include <Geom/Color.h>
#include <Geometry/Plane.h>
#include <Modules/Visualization/GenStandardColorMaps.h>

template class Array1<SurfaceOPort*>;
template class Array1<SurfaceHandle>;
template class Array1<TriSurface*>;
class TCLint;
template class Array1<TCLint*>;
template class Array1<GeomGroup*>;
class GeomPts;
template class Array1<GeomPts*>;
template class Array1<Plane>;
class GeomLines;
template class Array1<GeomLines*>;
template class Array1<SCI::Visualization::GenStandardColorMaps::StandardColorMap*>;
template class Array1<Array1<TSElement*> >;
class GeomTrianglesP;
template class Array1<GeomTrianglesP*>;
class Semaphore;
template class Array1<Semaphore*>;
template class Array1<Ring<int>*>;
template class Array1<HashTable<int, int>*>;
template class Array1<SimpleIPort<OctreeTopHandle>*>;
template class Array1<SimpleIPort<ScalarFieldHandle>*>;
class SLTracer;
template class Array1<SLTracer*>;
class GeomPolyline;
template class Array1<GeomPolyline*>;

class TCLint;
template class Array2<TCLint*>;
template class Array2<CharColor>;

template class FastHashTable<sci::Face>;
template class FastHashTableIter<sci::Face>;

template class Ring<int>;

class SLSourceInfo;
template class HashTable<int, SLSourceInfo*>;
template class HashTableIter<int, SLSourceInfo*>;


/*
 * Manual template instantiations for g++
 */

#include <Classlib/Array1.cc>
#include <Classlib/Array2.cc>
#include <Classlib/Array3.cc>
#include <Classlib/HashTable.cc>
#include <Classlib/LockingHandle.cc>
#include <Classlib/String.h>
#include <Geometry/BSphere.h>
#include <Geometry/Vector.h>
#include <Geom/Light.h>
#include <Geom/Material.h>
#include <Geom/Pt.h>
#include <Geom/VertexPrim.h>

template class Array1<BSphere>;
template class Array1<GeomObj*>;
template class Array1<MaterialHandle>;
struct ITree;
template class Array1<ITree*>;
template class Array1<Vector>;
class Light;
template class Array1<Light*>;
template class Array1<unsigned char>;
template class Array1<Colorub>;
template class Array1<TimedParticle>;
template class Array1<Array1<float> >;
template class Array1<InstTimedParticle>;
template class Array1<float*>;
template class Array1<GeomVertex*>;

template class Array2<double>;
template class Array2<MaterialHandle>;
template class Array2<Vector>;

template class Array3<char>;

template class HashTable<int, GeomObj*>;
template class HashTableIter<int, GeomObj*>;

template class LockingHandle<Material>;

template void Pio(Piostream&, MaterialHandle&);
template void Pio(Piostream&, Array1<MaterialHandle>&);
template void Pio(Piostream&, Array1<Point>&);
template void Pio(Piostream&, Array1<Vector>&);
template void Pio(Piostream&, Array2<double>&);
template void Pio(Piostream&, Array2<MaterialHandle>&);
template void Pio(Piostream&, Array2<Vector>&);
template void Pio(Piostream&, Array1<GeomObj*>&);
template void Pio(Piostream&, Array1<double>&);
template void Pio(Piostream&, HashTable<int, GeomObj*>&);
template void Pio(Piostream&, Array1<Light*>&);
template void Pio(Piostream&, Array1<int>&);
template void Pio(Piostream&, Array1<GeomVertex*>&);


/*
 * Manual template instantiations for g++
 */

#include <Classlib/Array1.cc>
#include <Classlib/Array2.cc>
#include <Classlib/HashTable.cc>
#include <Classlib/LockingHandle.cc>
#include <Classlib/Stack.cc>
#include <Modules/CS684/Pixel.h>
#include <Modules/CS684/RTPrims.h>
#include <Modules/CS684/RadPrims.h>
#include <Modules/CS684/Scene.h>
#include <Modules/CS684/Spectrum.h>

class RingWidget;
template class Array1<RingWidget*>;
class FrameWidget;
template class Array1<FrameWidget*>;
class ScaledBoxWidget;
template class Array1<ScaledBoxWidget*>;
class RadLink;
template class Array1<RadLink*>;
class RadObj;
template class Array1<RadObj*>;
template class Array1<RadMeshHandle>;
template class Array1<RTObjectHandle>;
template class Array1<RTLight>;
template class Array1<Spectrum>;

template class Array2<Pixel>;

template class LockingHandle<RTObject>;
template class LockingHandle<RTMaterial>;
template class LockingHandle<BRDF>;
template class LockingHandle<RadMesh>;

template void Pio(Piostream&, BRDFHandle&);
template void Pio(Piostream&, RTMaterialHandle&);
template void Pio(Piostream&, RTObjectHandle&);
template void Pio(Piostream&, Array1<RTObjectHandle>&);
template void Pio(Piostream&, Array1<RadMeshHandle>&);
template void Pio(Piostream&, Array2<Pixel>&);
template void Pio(Piostream&, Array1<Spectrum>&);

template class HashTable<int, RadObj*>;

template class Stack<RadObj*>;


/*
 * Manual template instantiations for g++
 */

#include <Classlib/Array1.cc>
#include <Classlib/Array2.cc>
#include <Classlib/Array3.cc>
#include <Classlib/HashTable.cc>
#include <Classlib/LockingHandle.cc>
#include <Classlib/Queue.cc>
#include <Multitask/AsyncReply.cc>
#include <Multitask/Mailbox.cc>
#include <Datatypes/cMatrix.h>
#include <Datatypes/cVector.h>
#include <Datatypes/Boolean.h>
#include <Datatypes/ColorMap.h>
#include <Datatypes/ColumnMatrix.h>
#include <Datatypes/ContourSet.h>
#include <Datatypes/DenseMatrix.h>
#include <Datatypes/GeometryComm.h>
#include <Datatypes/HexMesh.h>
#include <Datatypes/Image.h>
#include <Datatypes/Interval.h>
#include <Datatypes/LockArray3.cc>
#include <Datatypes/MultiMesh.h>
#include <Datatypes/Octree.h>
#include <Datatypes/ParticleGridReader.h>
#include <Datatypes/ParticleSet.h>
#include <Datatypes/ParticleSetExtension.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/SegFld.h>
#include <Datatypes/SigmaSet.h>
#include <Datatypes/Surface.h>
#include <Datatypes/SurfTree.h>
#include <Datatypes/TensorFieldBase.h>
#include <Datatypes/VectorField.h>
#include <Datatypes/VoidStar.h>

template class Array1<sci::NodeHandle>;
template class Array1<Color>;
template class Array1<Array1<Point> >;
template class Array1<sci::Element*>;
template class Array1<Array1<double> >;
template class Array1<sci::NodeVersion1>;
template class Array1<sci::ElementVersion1>;
template class Array1<sci::MeshHandle>;
template class Array1<DenseMatrix>;
template class Array1<AugElement>;
template class Array1<SampInfo>;
template class Array1<ScalarFieldHandle>;
class tripleInt;
template class Array1<Array1<tripleInt>*>;
template class Array1<tripleInt>;
class TSElement;
template class Array1<TSElement*>;
struct TSEdge;
template class Array1<TSEdge*>;
template class Array1<SurfInfo>;
template class Array1<FaceInfo>;
template class Array1<EdgeInfo>;
template class Array1<NodeInfo>;
template class Array1<Array1<int> >;
template class Array1<Array1<Array1<int> > >;
template class Array1<VectorFieldHandle>;
template class Array1<Hexahedron*>;
class MEFluid;
template class Array1<MEFluid*>;
template class Array1<Array1<Vector> >;
class cfdlibTimeStep;
template class Array1<cfdlibTimeStep*>;
struct SrchLst;
template class Array1<SrchLst*>;
template class Array1<Array3<short> >;
template class Array1<VectorFieldRG*>;
template class Array1<ScalarFieldRGdouble*>;

template class Array2<Color>;

template class Array3<Array1<int> >;
template class Array3<double>;
template class Array3<short>;
template class Array3<float>;
template class Array3<unsigned char>;
template class Array3<Vector>;

template void Pio(Piostream&, Array1<Color>&);
template void Pio(Piostream&, Array1<Array1<Point> >&);
template void Pio(Piostream&, Array3<Array1<int> >&);
template void Pio(Piostream&, Array1<sci::NodeVersion1>&);
template void Pio(Piostream&, Array1<sci::NodeHandle>&);
template void Pio(Piostream&, Array1<sci::ElementVersion1>&);
template void Pio(Piostream&, Array1<sci::Element*>&);
template void Pio(Piostream&, Array1<Array1<double> >&);
template void Pio(Piostream&, SurfaceHandle&);
template void Pio(Piostream&, Array1<sci::MeshHandle>&);
template void Pio(Piostream&, Array1<DenseMatrix>&);
template void Pio(Piostream&, Array3<double>&);
template void Pio(Piostream&, Array3<char>&);
template void Pio(Piostream&, Array3<short>&);
template void Pio(Piostream&, Array3<int>&);
template void Pio(Piostream&, Array3<float>&);
template void Pio(Piostream&, Array3<unsigned char>&);
template void Pio(Piostream&, Array1<ScalarFieldHandle>&);
template void Pio(Piostream&, Array1<Array1<tripleInt>*>&);
template void Pio(Piostream&, Array1<clString>&);
template void Pio(Piostream&, Array1<TSElement*>&);
template void Pio(Piostream&, Array1<TSEdge*>&);
template void Pio(Piostream&, Array1<SurfInfo>&);
template void Pio(Piostream&, Array1<FaceInfo>&);
template void Pio(Piostream&, Array1<EdgeInfo>&);
template void Pio(Piostream&, Array1<NodeInfo>&);
template void Pio(Piostream&, Array1<Array1<int> >&);
template void Pio(Piostream&, Array1<Array1<Array1<int> > >&);
template void Pio(Piostream&, Array1<VectorFieldHandle>&);
template void Pio(Piostream&, Array3<Vector>&);
template void Pio(Piostream&, HashTable<int, HexNode*>&);
template void Pio(Piostream&, HashTable<int, HexFace*>&);
template void Pio(Piostream&, HashTable<int, Hexahedron*>&);
template void Pio(Piostream&, Array1<Array1<Vector> >&);

template class Mailbox<GeomReply>;
struct SoundComm;
template class Mailbox<SoundComm*>;

template class AsyncReply<int>;
template class AsyncReply<GeometryData*>;

template class HashTable<int, int>;
template class HashTable<int, HexNode*>;
template class HashTableIter<int, HexNode*>;
template class HashTable<int, Hexahedron*>;
template class HashTableIter<int, Hexahedron*>;
template class HashTable<int, HexFace*>;
template class HashTableIter<int, HexFace*>;
template class HashTable<FourHexNodes, HexFace*>;


template class Queue<tripleInt>;

/*
 * These aren't used by Datatypes directly, but since they are used in
 * a lot of different modules, we instantiate them here to avoid bloat
 */
#include <Datatypes/SimplePort.cc>
template class SimpleIPort<cMatrixHandle>;
template class SimpleIPort<cVectorHandle>;
template class SimpleIPort<sci::MeshHandle>;
template class SimpleIPort<sciBooleanHandle>;
template class SimpleIPort<ColorMapHandle>;
template class SimpleIPort<ColumnMatrixHandle>;
template class SimpleIPort<ContourSetHandle>;
template class SimpleIPort<HexMeshHandle>;
template class SimpleIPort<ImageHandle>;
template class SimpleIPort<IntervalHandle>;
template class SimpleIPort<MatrixHandle>;
template class SimpleIPort<MultiMeshHandle>;
template class SimpleIPort<OctreeTopHandle>;
template class SimpleIPort<ParticleGridReaderHandle>;
template class SimpleIPort<ParticleSetHandle>;
template class SimpleIPort<ParticleSetExtensionHandle>;
template class SimpleIPort<ScalarFieldHandle>;
template class SimpleIPort<SegFldHandle>;
template class SimpleIPort<SigmaSetHandle>;
template class SimpleIPort<SurfaceHandle>;
template class SimpleIPort<TensorFieldHandle>;
template class SimpleIPort<VectorFieldHandle>;
template class SimpleIPort<VoidStarHandle>;

template class SimpleOPort<cVectorHandle>;
template class SimpleOPort<sci::MeshHandle>;
template class SimpleOPort<sciBooleanHandle>;
template class SimpleOPort<ColorMapHandle>;
template class SimpleOPort<ColumnMatrixHandle>;
template class SimpleOPort<ContourSetHandle>;
template class SimpleOPort<HexMeshHandle>;
template class SimpleOPort<ImageHandle>;
template class SimpleOPort<IntervalHandle>;
template class SimpleOPort<MatrixHandle>;
template class SimpleOPort<MultiMeshHandle>;
template class SimpleOPort<OctreeTopHandle>;
template class SimpleOPort<ParticleGridReaderHandle>;
template class SimpleOPort<ParticleSetHandle>;
template class SimpleOPort<ParticleSetExtensionHandle>;
template class SimpleOPort<ScalarFieldHandle>;
template class SimpleOPort<SegFldHandle>;
template class SimpleOPort<SigmaSetHandle>;
template class SimpleOPort<SurfaceHandle>;
template class SimpleOPort<TensorFieldHandle>;
template class SimpleOPort<VectorFieldHandle>;
template class SimpleOPort<VoidStarHandle>;

template class LockingHandle<cMatrix>;
template class LockingHandle<sci::Mesh>;
template class LockingHandle<sci::Node>;
template class LockingHandle<sciBoolean>;
template class LockingHandle<ColorMap>;
template class LockingHandle<ColumnMatrix>;
template class LockingHandle<ContourSet>;
template class LockingHandle<HexMesh>;
template class LockingHandle<Image>;
template class LockingHandle<Interval>;
template class LockingHandle<LockArray3<Point> >;
template class LockingHandle<Matrix>;
template class LockingHandle<MultiMesh>;
template class LockingHandle<OctreeTop>;
template class LockingHandle<ParticleGridReader>;
template class LockingHandle<ParticleSet>;
template class LockingHandle<ParticleSetExtension>;
template class LockingHandle<ScalarField>;
template class LockingHandle<SegFld>;
template class LockingHandle<SigmaSet>;
template class LockingHandle<Surface>;
template class LockingHandle<TensorFieldBase>;
template class LockingHandle<VectorField>;
template class LockingHandle<VoidStar>;

template class Mailbox<SimplePortComm<ScalarFieldHandle> >;

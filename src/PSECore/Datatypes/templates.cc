/*
 * Manual template instantiations for g++
 */


/*
 * These aren't used by Datatypes directly, but since they are used in
 * a lot of different modules, we instantiate them here to avoid bloat
 */

#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Datatypes/ScalarField.h>

#include <PSECore/Datatypes/SimplePort.h>

using namespace SCICore::Containers;
using namespace SCICore::Datatypes;
using namespace SCICore::GeomSpace;

using namespace PSECore::Datatypes;

#if 0

template class SimpleIPort<cMatrixHandle>;
template class SimpleIPort<cVectorHandle>;
template class SimpleIPort<MeshHandle>;
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
template class SimpleOPort<MeshHandle>;
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
#endif

//template class LockingHandle<ScalarField>;
template void Pio<>(Piostream&, LockingHandle<ScalarField>&);

// Pio__SCICore::PersistentSpace::Piostream(
//					    SCICore::Containers::LockingHandle,
//					    SCICore::Datatypes::ScalarField
//					   )

#if 0
template class LockingHandle<cMatrix>;
template class LockingHandle<Node>;
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
template class LockingHandle<SegFld>;
template class LockingHandle<SigmaSet>;
template class LockingHandle<Surface>;
template class LockingHandle<TensorFieldBase>;
template class LockingHandle<VectorField>;
template class LockingHandle<VoidStar>;

template class Mailbox<SimplePortComm<ScalarFieldHandle> >;

#endif

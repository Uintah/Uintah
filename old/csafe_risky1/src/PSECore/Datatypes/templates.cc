/*
 * Manual template instantiations
 */


/*
 * These aren't used by Datatypes directly, but since they are used in
 * a lot of different modules, we instantiate them here to avoid bloat
 *
 * Find the bloaters with:
find . -name "*.ii" -print | xargs cat | sort | uniq -c | sort -nr | more
 */

#include <PSECore/Datatypes/SimplePort.h>

using namespace SCICore::Datatypes;
using namespace PSECore::Datatypes;

#include <SCICore/Datatypes/Surface.h>
template class SimpleIPort<SurfaceHandle>;
template class SimpleOPort<SurfaceHandle>;

#include <SCICore/Datatypes/ScalarField.h>
template class SimpleIPort<ScalarFieldHandle>;
template class SimpleOPort<ScalarFieldHandle>;

#include <SCICore/Datatypes/Matrix.h>
template class SimpleIPort<MatrixHandle>;
template class SimpleOPort<MatrixHandle>;

#include <SCICore/Datatypes/ColorMap.h>
template class SimpleIPort<ColorMapHandle>;
template class SimpleOPort<ColorMapHandle>;

#include <SCICore/Datatypes/ColumnMatrix.h>
template class SimpleIPort<ColumnMatrixHandle>;
template class SimpleOPort<ColumnMatrixHandle>;

#include <SCICore/Datatypes/VectorField.h>
template class SimpleIPort<VectorFieldHandle>;
template class SimpleOPort<VectorFieldHandle>;

#include <SCICore/Datatypes/Mesh.h>
template class SimpleIPort<MeshHandle>;
template class SimpleOPort<MeshHandle>;


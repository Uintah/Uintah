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

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/ColorMap.h>
#include <Core/Datatypes/VectorField.h>
#include <Core/Datatypes/Mesh.h>

using namespace SCIRun;

template class SimpleIPort<ScalarFieldHandle>;
template class SimpleOPort<ScalarFieldHandle>;

template class SimpleIPort<MatrixHandle>;
template class SimpleOPort<MatrixHandle>;

template class SimpleIPort<ColorMapHandle>;
template class SimpleOPort<ColorMapHandle>;

template class SimpleIPort<VectorFieldHandle>;
template class SimpleOPort<VectorFieldHandle>;

template class SimpleIPort<MeshHandle>;
template class SimpleOPort<MeshHandle>;


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

#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/LatticeGeom.h>
#include <Core/Datatypes/ScalarField.h>

using namespace SCIRun;
#ifdef __sgi
#pragma set woff 1468
#endif

template class LockingHandle<ScalarField>;

#include <Core/Datatypes/ColumnMatrix.h>
template class LockingHandle<ColumnMatrix>;

#include <Core/Datatypes/Matrix.h>
template class LockingHandle<Matrix>;

#include <Core/Datatypes/Mesh.h>
template class LockingHandle<Mesh>;

#include <Core/Datatypes/Surface.h>
template class LockingHandle<Surface>;


#include <Core/Datatypes/FieldRG.h>
template class FieldRG<double>;

#include <Core/Datatypes/FieldTet.h>
template class FieldTet<double>;

#ifdef __sgi
#pragma reset woff 1468
#endif











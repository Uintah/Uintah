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

#include <SCICore/Containers/LockingHandle.h>

#include <SCICore/Datatypes/ScalarField.h>
using namespace SCICore::Datatypes;

#ifdef __sgi
#pragma set woff 1468
#endif

template class LockingHandle<ScalarField>;

#include <SCICore/Datatypes/ColumnMatrix.h>
template class LockingHandle<ColumnMatrix>;

#include <SCICore/Datatypes/Matrix.h>
template class LockingHandle<Matrix>;

#include <SCICore/Datatypes/Mesh.h>
template class LockingHandle<Mesh>;

#include <SCICore/Datatypes/Surface.h>
template class LockingHandle<Surface>;

#ifdef __sgi
#pragma reset woff 1468
#endif

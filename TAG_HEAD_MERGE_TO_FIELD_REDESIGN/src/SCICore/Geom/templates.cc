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

#include <SCICore/Containers/Array1.h>

#include <SCICore/Geom/Color.h>

using namespace SCICore::GeomSpace;
using namespace SCICore::Containers;

#ifdef __sgi
#pragma set woff 1468
#endif

template class Array1<Color>;

#ifdef __sgi
#pragma reset woff 1468
#endif

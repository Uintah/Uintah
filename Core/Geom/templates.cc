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

#include <Core/Containers/Array1.h>

#include <Core/Geom/Color.h>


#ifdef __sgi
#pragma set woff 1468
#endif

using namespace SCIRun;

template class Array1<Color>;

#ifdef __sgi
#pragma reset woff 1468
#endif

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
#include <SCICore/Containers/Array2.h>

#ifdef __sgi
#pragma set woff 1468
#endif

using namespace SCICore::Containers;
template class Array1<int>;
template class Array1<double>;

template class Array2<int>;
template class Array2<double>;


#ifdef __sgi
#pragma reset woff 1468
#endif

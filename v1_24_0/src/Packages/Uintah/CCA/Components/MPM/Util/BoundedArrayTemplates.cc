#ifdef __GNUG__

#include "BoundedArray.h"
#include "BoundedArray.cc"

// Instantiate the BoundedArray class for integers and matrix of ints


template class BoundedArray<int>;
template class BoundedArray<BoundedArray<int> *>;
template class BoundedArray<BoundedArray<BoundedArray<int> *> *>;

#endif


#ifndef HASH_H
#define HASH_H 1

#include <Core/Util/Assert.h>

namespace rtrt {

template<class Key> inline unsigned int Hash(const Key& k)
{
    return k.hash();
}

inline unsigned int Hash(const int& k)
{
    return k;
}

inline unsigned int Hash(const unsigned long& k)
{
    return (unsigned int)k;
}   

inline unsigned int Hash(const void*& k)
{
    return (unsigned int)(unsigned long)k;
}

} // end namespace rtrt

#endif


#ifndef HASH_H
#define HASH_H 1

#include <Core/Util/Assert.h>
#include <string>

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

inline unsigned int Hash(const std::string& k)
{
  using namespace std;
  unsigned int sum=0;
  for(string::const_iterator iter = k.begin(); iter != k.end(); iter++)
    sum=(sum<<3)^(sum>>2)^(*iter<<1);
  return sum;
}

} // end namespace rtrt

#endif

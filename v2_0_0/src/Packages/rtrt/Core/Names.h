
#ifndef OBJNAMES_H
#define OBJNAMES_H

#include <string>

namespace rtrt {
  class Object;
  using namespace std;
  class Names {
  public:
    static void nameObject(const string& name, const Object* ptr);
    static const string& getName(const Object* ptr);
    static bool hasName(const Object* ptr);
  };
}

#endif

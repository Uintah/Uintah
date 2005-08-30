
#ifndef OBJNAMES_H
#define OBJNAMES_H

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace rtrt {
  class Object;
  class Material;
  using namespace std;
  class Names {
  public:
    // Please note that nameObjectWithUnique and nameObject are not
    // thread safe.
    
    // This will generate a unique name for the object and then use it
    // to name the object.
    static void nameObjectWithUnique(const Object* ptr);

    // Names the object with the name.  No checking is done to make
    // sure the object has already been named.
    static void nameObject(const string& name, const Object* ptr);

    // Returns the name of the object.  If the object has no name then
    // an empty string is returned ("").
    static const string& getName(const Object* ptr);

    // Return true if a name was found for the object, false otherwise.
    static bool hasName(const Object* ptr);

    // The same for materials
    static void nameMaterialWithUnique(const Material* ptr);
    static void nameMaterial(const string& name, const Material* ptr);
    static const string& getName(const Material* ptr);
    static bool hasName(const Material* ptr);
  };
}

#endif

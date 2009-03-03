/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



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

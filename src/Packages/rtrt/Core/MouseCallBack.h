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



#ifndef MOUSECALLBACK_H
#define MOUSECALLBACK_H

#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/HitInfo.h>

namespace rtrt {
  class Object;
  class Ray;
  class HitInfo;

  typedef void (*cbFunc)( Object *obj, const Ray&, const HitInfo& );

/*    typedef struct MouseCBGroup { */
/*      cbFunc mouseDown; */
/*      cbFunc mouseUp; */
/*      cbFunc mouseMotion; */
/*    } MouseCBGroup; */
  
  using namespace std;
  
  class MouseCallBack {
  public:
    
    // assigns a callback to an object
/*      static void assignCB( MouseCBGroup &cbGroup, const Object* object); */
    static void assignCB_MD( cbFunc fp, const Object* object );
    static void assignCB_MU( cbFunc fp, const Object* object );
    static void assignCB_MM( cbFunc fp, const Object* object );
    // retrieves a callback from an object
    static cbFunc getCB_MD( const Object* object );
    static cbFunc getCB_MU( const Object* object );
    static cbFunc getCB_MM( const Object* object );
    // determines whether or not an object has been assigned a callback
    static bool hasCB_MD( const Object* object );
    static bool hasCB_MU( const Object* object );
    static bool hasCB_MM( const Object* object );
  };
}

#endif

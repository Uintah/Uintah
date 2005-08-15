/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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



/*
 *  Light.h: Base class for light sources
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Light_h
#define SCI_Geom_Light_h 1

#include <sci_defs/ogl_defs.h>

#include <Core/Containers/LockingHandle.h>
#include <Core/Thread/Mutex.h>
#include <Core/Persistent/Persistent.h>

namespace SCIRun {

class Point;
class Vector;
class Color;
class GeomObj;
class OcclusionData;
class View;
struct DrawInfoOpenGL;


class Light : public Persistent {
protected:
  Light(const string& name, bool on = true, bool tranformed = true);
    
public:
  int ref_cnt;
  Mutex &lock;
  string name;
  bool on;
  bool transformed;  // defaults to true meaning you do want the light 
                     // to be transformed by the current modelview matrix.  
                     // Set to false for headlights and directional lights 
                     // fixed in the viewing hemisphere.
  virtual ~Light();
  virtual void io(Piostream&);

  friend void Pio( Piostream&, Light*& );

  void opengl_reset_light( int i );

  static PersistentTypeID type_id;
#ifdef SCI_OPENGL
  virtual void opengl_setup(const View& view, DrawInfoOpenGL*, int& idx)=0;
#endif
};

typedef LockingHandle<Light> LightHandle;

} // End namespace SCIRun


#endif /* SCI_Geom_Light_h */


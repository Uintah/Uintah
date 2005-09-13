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
 *  GeomDL.h: ?
 *
 *  Written by:
 *   Author Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Date July 2000
 *
 *  Copyright (C) 2000  SCI Group
 */

#ifndef SCI_Geom_GeomDL_h 
#define SCI_Geom_GeomDL_h 1

#ifdef SCI_OPENGL
#include <Core/Geom/GeomOpenGL.h>
#endif
#include <Core/Geom/GeomContainer.h>

#include <list>

namespace SCIRun {
    
class DrawInfoOpenGL;
using std::list;

class GeomDL : public GeomContainer {
protected:
  int polygons_;
  list<DrawInfoOpenGL *> drawinfo_;

public:
  GeomDL(GeomHandle);
  GeomDL(const GeomDL &copy);
  virtual ~GeomDL();
      
  virtual GeomObj* clone();
  virtual void reset_bbox();

  void dl_register(DrawInfoOpenGL *info);
  void dl_unregister(DrawInfoOpenGL *info);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;	
};
    
} // End namespace SCIRun

#endif

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
 *  GeomObj.h: Displayable Geometry
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_GeomObj_h
#define SCI_Geom_GeomObj_h 1

#include <sci_defs/ogl_defs.h>

#include <Core/Containers/Array1.h>
#include <Core/Thread/Mutex.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Persistent/Persistent.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>
#include <Core/Geom/share.h>

namespace SCIRun {

using std::vector;

struct DrawInfoOpenGL;
class  Material;
class  GeomSave;
class  Hit;
class  BBox;
class  Vector;
class  Point;
class  IntVector;

class SHARE GeomObj : public Persistent {
public:
  int ref_cnt;
  Mutex &lock;

  GeomObj();
  GeomObj(const GeomObj&);
  virtual ~GeomObj();
  virtual GeomObj* clone() = 0;

  virtual void reset_bbox();
  virtual void get_bounds(BBox&) = 0;

  // For OpenGL
#ifdef SCI_OPENGL
  int pre_draw(DrawInfoOpenGL*, Material*, int lit);
  virtual void draw(DrawInfoOpenGL*, Material*, double time)=0;
  int post_draw(DrawInfoOpenGL*);
#endif
  virtual void get_triangles( Array1<float> &v );
  static PersistentTypeID type_id;

  virtual void io(Piostream&);    

  // we want to return false if value is the default value
  virtual void setId(int id) { id_int_ = id; }
  virtual void setId(const IntVector &id) { id_intvector_ = id; }
  virtual void setId(long long id) { id_longlong_ = id; }

  virtual bool getId( int &id );
  virtual bool getId( IntVector &id );
  virtual bool getId( long long &id );

private:

  int       id_int_;
  IntVector id_intvector_;
  long long  id_longlong_;
};

void Pio(Piostream&, GeomObj*&);

typedef LockingHandle<GeomObj> GeomHandle;

} // End namespace SCIRun

#endif // ifndef SCI_Geom_GeomObj_h





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
 *  Drawable.h: Displayable 2D object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_Drawable_h
#define SCI_Drawable_h 

#include <Core/Containers/Array1.h>
#include <Core/2d/BBox2d.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Thread/Mutex.h>

#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

class  BBox2D;
class  OpenGLWindow;

class Drawable : public Datatype {
private:
  Mutex *lock_;
  string name_;
  bool enabled_;
  Drawable *parent_;

 protected:
  OpenGLWindow *ogl_;

public:
  Drawable(const string &name="");
  virtual ~Drawable();

  void lock() { lock_->lock(); }
  void unlock() { lock_->unlock(); }
  string name() { return name_;}
  void set_name ( const string &name) { name_=name;}
  Drawable *parent() { return parent_; }
  void set_parent( Drawable *p ) { parent_ = p; }
  bool is_enabled() { return enabled_; }
  void enable( bool state ) { enabled_ = state; }
  virtual void set_opengl ( OpenGLWindow *w ) { ogl_ = w ; }
  virtual void need_redraw() {}
  virtual void reset_bbox();
  virtual void get_bounds(BBox2d&) = 0;
  

  // For OpenGL
#ifdef SCI_OPENGL
  virtual void draw()=0;
#endif
  static PersistentTypeID type_id;

  virtual void io(Piostream&);    
};

void Pio(Piostream&, Drawable*&);

} // namespace SCIRun

#endif // SCI_Drawable_h

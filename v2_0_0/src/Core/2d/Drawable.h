/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
#include <sci_config.h>

#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

class  BBox2D;
class  OpenGLWindow;

class SCICORESHARE Drawable : public Datatype {
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

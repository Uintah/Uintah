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
 *  DrawObj.h: Displayable 2D object
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

#include <Core/Datatypes/Color.h>
#include <Core/Containers/Array1.h>
#include <Core/2d/BBox2d.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Thread/CrowdMonitor.h>
#include <sci_config.h>
#include <stdio.h>

#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

class  BBox2D;
class  OpenGLWindow;
using std::vector;

class SCICORESHARE DrawObj : public Datatype {
private:
  string name_;
  DrawObj *parent_;
  CrowdMonitor *lock_;

protected:
  OpenGLWindow *ogl_;
  Color color_;

public:
  DrawObj(const string &name="");
  virtual ~DrawObj();

  void set_lock( CrowdMonitor *lock ) { lock_ = lock; }
  void read_lock() { if (lock_) lock_->readLock(); }
  void read_unlock() { lock_->readUnlock(); }

  void write_lock() { if (lock_) lock_->writeLock(); }
  void write_unlock() { lock_->writeUnlock(); }

  string name() { return name_;}
  string tcl_color();
  void set_name ( const string &name) { name_=name;}
  DrawObj *parent() { return parent_; }
  void set_parent( DrawObj *p ) { parent_ = p; }

  virtual void set_opengl ( OpenGLWindow *w ) { ogl_ = w ; }
  virtual void need_redraw() {}
  virtual void reset_bbox();
  virtual void get_bounds(BBox2d&) = 0;
  virtual void child_changed( DrawObj *) {}

  virtual void add(const vector<double>&) {}

  // For OpenGL
#ifdef SCI_OPENGL
  virtual void draw( bool = false )=0;
#endif
  static PersistentTypeID type_id;

  virtual void io(Piostream&);    
};

void Pio(Piostream&, DrawObj*&);

} // namespace SCIRun

#endif // SCI_DrawObj_h

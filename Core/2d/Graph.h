/*
 *  Graph.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   July 20001
 *
 */


#ifndef Graph_h
#define Graph_h

#include <string>
#include <map>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>

#include <Core/2d/Drawable.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/Array1.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/GuiInterface/TclObj.h>
#include <Core/Thread/Mutex.h>
#include <Core/Geom/Color.h>
#include <Core/2d/OpenGLWindow.h>



namespace SCIRun {

struct ObjInfo {
  string name_;
  Drawable *obj_;
  bool mapped_;

  ObjInfo( const string &name, Drawable *d);
  ~ObjInfo() { if (obj_) delete obj_; }

  void draw() { obj_->draw(); }
  void set_window( const string &);
  void set_id( const string & );
};

class Graph : public TclObj, public Drawable, public OpenGLWindow {
private:
  Mutex *lock_;
  ObjInfo *obj_;

public:
  Graph( const string & );
  virtual ~Graph() {}

  void add( const string &, Drawable *);
  virtual void tcl_command(TCLArgs&, void*);
  virtual void set_window( const string &);
  void update();

  void lock() { lock_->lock(); }
  void unlock() { lock_->unlock(); }

  virtual void need_redraw();
  virtual void get_bounds( BBox2d &) {}
  virtual void draw() {}
  virtual void io(Piostream& stream);

};


} // End namespace SCIRun

#endif Graph_h



/*
 *  Graph.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   July 20001
 *
 */



#include <iostream>
using std::cin;
using std::endl;
#include <sstream>
using std::ostringstream;

#include <tcl.h>
#include <tk.h>

#include <Core/2d/TclObj.h>
#include <Core/2d/Graph.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/TCL.h>

namespace SCIRun {


ObjInfo::ObjInfo( const string &name, Drawable *d)  
{
  name_ = name;
  obj_ = d;
  mapped_ = false;
}

void 
ObjInfo::set_window( const string &window)
{
  TclObj *t = dynamic_cast<TclObj *>(obj_);
  if ( t )
    t->set_window( window );
}

void 
ObjInfo::set_id( const string &id)
{
  TclObj *t = dynamic_cast<TclObj *>(obj_);
  if ( t )
    t->set_id(id);
}


Graph::Graph( const string &id )
  : TclObj( "Graph" ), Drawable( id )
{
  lock_ = scinew Mutex((string("Graph::")+id).c_str());
  obj_ = 0;

  ostringstream tmp;
  tmp << id << "-" << generation;
  set_id( tmp.str() );
}

void
Graph::set_window( const string &window )
{
  TclObj::set_window( window);
  
  if ( obj_ ) {
    obj_->set_window( window + ".ctrl");
  }
}

void
Graph::add( const string &name, Drawable *d )
{
  d->set_parent( this );

  if ( obj_ ) delete obj_;

  obj_ = scinew ObjInfo (name, d );

  obj_->set_id( id() + "-obj" );
  if ( initialized_ &&  window() != "" )
    obj_->set_window( window() + ".ctrl" );
}


void
Graph::need_redraw() 
{
  update();
}

void
Graph::update()
{
  if ( !initialized_ || !obj_) {
    return; 
  }
  
  pre();
  clear();
  obj_->draw();
  post();
}

void
Graph::tcl_command(TCLArgs& args, void* userdata)
{
  if ( OpenGLWindow::tcl_command( args, userdata ) ) 
   return;

  if ( args[1] == "redraw" ) {
    if ( initialized_ ) 
      update();
  }

}


#define GRAPH_VERSION 1

void
Graph::io(Piostream& stream)
{
  stream.begin_class( "Graph", GRAPH_VERSION);


  stream.end_class();
}

} // End namespace SCIRun



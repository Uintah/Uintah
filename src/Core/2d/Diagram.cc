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
 *  Diagram.cc: Displayable 2D object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */


#include <stdio.h>
#include <iostream>
using std::cerr;
using std::ostream;
#include <sstream>
using std::ostringstream;

#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/2d/Diagram.h>
#include <Core/2d/Hairline.h>
#include <Core/2d/BBox2d.h>
#include <Core/2d/OpenGLWindow.h>

namespace SCIRun {

Persistent* make_Diagram()
{
  return scinew Diagram;
}

PersistentTypeID Diagram::type_id("diagram", "DrawObj", make_Diagram);

Diagram::Diagram( const string &name)
  : TclObj( "Diagram"), DrawObj(name)
{
  select_mode_ = 2;
  scale_mode_ = 1;
  draw_mode_ = Draw;
  selected_widget_ = -1;
  selected_ = 0;
}


Diagram::~Diagram()
{
}

void
Diagram::add( DrawObj *d )
{
  graph_.add(d);
  if ( window() != "" ) {
    tcl_ << " add " << (graph_.size()-1) << " ";
    if ( d->name() != "" )
      tcl_ << d->name();
    else
      tcl_ << (graph_.size()-1);
    tcl_exec();
  }
}


int
Diagram::add_widget( Widget *w )
{
  widget_.add(w);
  return widget_.size()-1;
}

void
Diagram::reset_bbox()
{
  graphs_bounds_.reset();
  for (int i=0; i<graph_.size(); i++)
    if ( graph_[i]->is_enabled() )
      graph_[i]->get_bounds( graphs_bounds_ );
}
  
void
Diagram::get_bounds( BBox2d &bb )
{
  reset_bbox();
  bb.extend( graphs_bounds_ );
}


void
Diagram::tcl_command(TCLArgs& args, void* userdata)
{
  if ( args[1] == "select" ) {
    int plot = atoi( args[2].c_str() );
    bool state = args[3] == "0" ? false : true;
    graph_[plot]->enable( state );
    redraw();
  } 
  else if ( args[1] == "select-one" ) {
    selected_ = atoi( args[2].c_str() );
    redraw();
  } 
  else if ( args[1] == "redraw" ) {
    reset_vars();
    select_mode_ = gui_select->get();
    scale_mode_ = gui_scale->get();
    redraw();
  }
  else if ( args[1] == "ButtonPress" ) {
    int x, y, b;
    string_to_int(args[2], x);
    string_to_int(args[3], y);
    string_to_int(args[4], b);
    button_press( x, y, b );
  }
  else if ( args[1] == "Motion" ) {
    int x, y, b;
    string_to_int(args[2], x);
    string_to_int(args[3], y);
    string_to_int(args[4], b);
    button_motion( x, y, b );
  }
  else if ( args[1] == "ButtonRelease" ) {
    int x, y, b;
    string_to_int(args[2], x);
    string_to_int(args[3], y);
    string_to_int(args[4], b);
    button_release( x, y, b );
  }
  else if ( args[1] == "widget" ) {
    // start a widget
    if ( args[2] == "hairline" ) {
      add_hairline();
    }
    else {
      cerr << "Diagram[tcl_command]: unknown widget requested" << endl;
    }
  }
  else
    cerr << "Diagram[tcl_command]: unknown tcl command requested" << endl;
}


void
Diagram::add_hairline() 
{
  BBox2d b1, b2;
  get_bounds( b1 );
  b2.extend( Point2d( b1.min().x(), 0 ) );
  b2.extend( Point2d( b1.max().x(), 1 ) );

  Hairline *hair = scinew Hairline( b2, "Hairline");
  int w = add_widget( hair );

  string window_name;
  tcl_eval( "new-opt " , window_name );
  hair->set_id( id()+"-hairline-" + to_string(w) );
  hair->set_window( window_name, string("hair")+ to_string(w));

  for (int i=0; i<graph_.size(); i++) {
    Polyline *p = dynamic_cast<Polyline *>(graph_[i]);
    if ( p ) 
      hair->add( p );
  }
  redraw();
}


void
Diagram::set_id( const string & id )
{
  ostringstream tmp;
  tmp << id << "-" << generation;

  gui_select = scinew GuiInt("select", tmp.str(), this );
  gui_scale = scinew GuiInt("scale", tmp.str(), this );

  TclObj::set_id( tmp.str() );
}

void
Diagram::set_window( const string & window )
{
  TclObj::set_window( window, name() );

  for (int i=0; i<graph_.size(); i++) {
    tcl_ << " add " << i << " ";
    if ( graph_[i]->name() != "" )
      tcl_ << graph_[i]->name();
    else
      tcl_ << i;
    tcl_exec();
  }
}


void
Diagram::button_press( int x, int y, int button )
{
  if ( ogl_ ) {
    draw_mode_ = Pick;

    GLuint buffer[10]; 

    ogl_->pre();

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();

    GLint viewport[4];
    glGetIntegerv( GL_VIEWPORT, viewport );

    gluPickMatrix(x, y, 10, 10, viewport );

    glSelectBuffer( 10, buffer );
    glRenderMode( GL_SELECT );
    glInitNames();
    glPushName(2);

    draw();

    int n = glRenderMode( GL_RENDER );

    if ( n > 0 ) {
      selected_widget_ = buffer[3];
      widget_[ selected_widget_ ]->select( x, y, button );
    }

    glPopMatrix();
    ogl_->post();
    
    draw_mode_ = Draw;
    redraw();
  }
  else
    cerr << "diagram no ogl \n";
}

void
Diagram::button_motion( int x, int y, int button )
{
  if ( selected_widget_ != -1 ) {
    widget_[selected_widget_]->move( x, y, button );
    redraw();
  }
}

void
Diagram::button_release( int x, int y, int button )
{
  if ( selected_widget_ != -1) {
    widget_[ selected_widget_ ]->release(x, y, button );
    selected_widget_ = -1;
    redraw();
  }
}

#define DIAGRAM_VERSION 1

void 
Diagram::io(Piostream& stream)
{
  stream.begin_class("Diagram", DIAGRAM_VERSION);
  DrawObj::io(stream);
  stream.end_class();
}


} // namespace SCIRun

  

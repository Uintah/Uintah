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

#include <Core/Malloc/Allocator.h>
#include <Core/2d/Diagram.h>
#include <Core/2d/BBox2d.h>


namespace SCIRun {

Persistent* make_Diagram()
{
  return scinew Diagram;
}

PersistentTypeID Diagram::type_id("diagram", "Drawable", make_Diagram);

Diagram::Diagram( const string &name)
  : TclObj( "Diagram"), Drawable(name)
{
  select_mode = 2;
  scale_mode = 1;
}


Diagram::~Diagram()
{
}

void
Diagram::add( Drawable *d )
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
   if ( parent() ) parent()->need_redraw();
  } 
  else if ( args[1] == "select-one" ) {
    selected_ = atoi( args[2].c_str() );
   if ( parent() ) parent()->need_redraw();
  } 
  else if ( args[1] == "redraw" ) {
    reset_vars();
    select_mode = gui_select->get();
    scale_mode = gui_scale->get();
    if ( parent() ) parent()->need_redraw();
  }
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
  cerr << "Diagram name = " << name() << endl;
  TclObj::set_window( window, name() );

  for (int i=0; i<graph_.size(); i++) {
    tcl_ << id() << " add " << i << " ";
    if ( graph_[i]->name() != "" )
      tcl_ << graph_[i]->name();
    else
      tcl_ << i;
    tcl_exec();
  }
}
  
#define DIAGRAM_VERSION 1

void 
Diagram::io(Piostream& stream)
{
  stream.begin_class("Diagram", DIAGRAM_VERSION);
  Drawable::io(stream);
  stream.end_class();
}


} // namespace SCIRun

  

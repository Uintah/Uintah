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
 *  TclObj.cc: Displayable 2D object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/GuiInterface/TclObj.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/TCL.h>

#include <Core/Malloc/Allocator.h>

#include <stdio.h>
#include <iostream>
#include <sstream>

#include <tcl.h>
#include <tk.h>

using namespace std;

namespace SCIRun {


TclObj::TclObj( const string &script) 
  : id_(""), window_(""), script_(script), has_window_(false)
{
} 

TclObj::TclObj( const string &script, const string &id ) 
  : id_(""), window_(""), script_(script), has_window_(false)
{
  set_id( id );
} 

TclObj::~TclObj()
{
  if ( id_ != "" ) {
    ostringstream cmd;
    cmd << "delete object " << id_;
    TCL::execute( cmd.str().c_str() );
  }
}

void
TclObj::command( const string &s )
{
  ostringstream cmd;
  cmd << id_ << " " << s;
  //cerr << "TclObj::exec " << cmd.str() << endl;
  TCL::execute( cmd.str().c_str() );
}

int
TclObj::tcl_eval( const string &s, string &result )
{
  ostringstream cmd;
  cmd << id_ << " " << s;
  //cerr << "TclObj::eval " << cmd.str() << endl;

  return TCL::eval( cmd.str(), result );
}

void
TclObj::tcl_exec()
{
  TCL::execute( tcl_.str().c_str() );
  tcl_.str( " ");
  tcl_ << id_ << " ";
}

void
TclObj::set_id( const string & id )
{
  if ( id[0] == ':' )
    id_ = id;
  else
    id_ = string("::")+id;

  TCL::add_command( id_+"-c", this, 0 );
  
  ostringstream cmd;
  cmd << script_ << " " << id_;
  //cerr << "TclObj::set_id  " << cmd.str() << endl;
  TCL::execute( cmd.str().c_str() );
  
  tcl_ << id_ << " ";}

void
TclObj::set_window( const string & window, const string& args,  bool exec )
{
  window_ = window;
  if ( exec ) {
    ostringstream cmd;
    cmd << id_ << " ui " << window << " " << args;
    TCL::execute( cmd.str().c_str() );
  }
  has_window_ = true;
}


} // namespace SCIRun

  

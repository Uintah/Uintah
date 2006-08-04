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
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiInterface.h>

#include <stdio.h>
#include <iostream>
#include <sstream>

#include <tcl.h>
#include <tk.h>

using namespace std;

namespace SCIRun {


TclObj::TclObj(GuiInterface* gui, const string &script)
  : id_(""), window_(""), script_(script), has_window_(false), gui(gui)
{
} 

TclObj::TclObj(GuiInterface* gui, const string &script, const string &id ) 
  : id_(""), window_(""), script_(script), has_window_(false), gui(gui)
{
  set_id( id );
} 

TclObj::~TclObj()
{
  if ( id_ != "" ) {
    ostringstream cmd;
    cmd << "delete object " << id_;
    gui->execute( cmd.str() );
  }
}

void
TclObj::command( const string &s )
{
  ostringstream cmd;
  cmd << id_ << " " << s;
  gui->execute( cmd.str() );
}

int
TclObj::tcl_eval( const string &s, string &result )
{
  ostringstream cmd;
  cmd << id_ << " " << s;
  //cerr << "TclObj::eval " << cmd.str() << endl;

  return gui->eval( cmd.str(), result );
}

void
TclObj::tcl_exec()
{
  gui->execute( tcl_.str() );
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

  gui->add_command( id_+"-c", this, 0 );
  
  ostringstream cmd;
  cmd << script_ << " " << id_;
  //cerr << "TclObj::set_id  " << cmd.str() << endl;
  gui->execute( cmd.str() );
  
  tcl_ << id_ << " ";}

void
TclObj::set_window( const string & window, const string& args,  bool exec )
{
  window_ = window;
  if ( exec ) {
    ostringstream cmd;
    cmd << id_ << " ui " << window << " " << args;
    gui->execute( cmd.str() );
  }
  has_window_ = true;
}


} // namespace SCIRun

  

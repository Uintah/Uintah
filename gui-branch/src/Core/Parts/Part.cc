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
 *  Part.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <iostream>
#include <Core/Persistent/Pstreams.h>
#include <Core/Parts/Part.h>
#include <Core/Framework/CoreFramework.h>
#include <Core/Parts/GuiVar.h>
#include <Core/GuiInterface/GuiManager.h>

namespace SCIRun {

using namespace std;
  
Signal1<const string &> Part::tcl_execute;
Signal2<const string &, string &> Part::tcl_eval;
Signal3<const string &, Part *, void *> Part::tcl_add_command;
Signal1<const string &> Part::tcl_delete_command;


Part::Part( Part *parent, const string &name, const string &type )
  : name_(name), type_(type), parent_(parent)
{
}


void
Part::init()
{
  framework->add_part( name_, this );
  port_ = new PartPort( this );
  framework->add_port( name_+"::default", port_ );

  if ( parent_ )
    parent_->add_child( this );
}
 
Part::~Part()
{
  // remove defaults port
  framework->rem_port( name_+"::default" );

  // delete all children
  for (unsigned i=0; i<children_.size(); i++)
    delete children_[i];

  // inform parent
  if ( parent_ )
    parent_->rem_child( this );

  // inform framework
  framework->rem_part( name_ );
}

void
Part::add_child( Part *child )
{
  children_.push_back(child);
}


void
Part::rem_child( Part *child )
{
  vector<Part *>::iterator i;
  for (i=children_.begin(); i!=children_.end(); i++)
    if ( *i == child ) {
      children_.erase( i );
      return;
    }
}

void 
Part::emit_vars( ostream& out, string &midx )
{
  for (unsigned i=0; i!= vars_.size(); i++)
    vars_[i]->emit(out,midx);
}

void 
Part::add_gui_var( GuiVar *v )
{
  vars_.push_back(v);
}

void 
Part::rem_gui_var( GuiVar *v )
{
  for (unsigned i=0; i<vars_.size(); i++)
    if ( vars_[i] == v ) {
      vars_.erase( vars_.begin()+i );
      return;
    }
}

void
Part::reset_vars()
{
  for (unsigned i=0; i<vars_.size(); i++)
    vars_[i]->reset();
}
     

// tcl compatibity

const string &
Part::get_var( const string &obj, const string &var )
{
  static string res;
  //  tcl_eval( obj + " get-var " + var, res );
  tcl_eval( obj + " cget  -" + var, res );
  cerr << "Part::get_var = " << obj << " cget -" << var << " =  " <<res<<endl;
  return res;
}

void
Part::set_var( const string &obj, const string &var, const string &value )
{
  cerr << "Part::set_var = " << obj << " configure -" << var << " " << value << endl;
  tcl_execute( obj + " configure -" + var + " " + value );
}

int 
Part::get_gui_stringvar(const string &base, const string &name, string &value )
{
  return gm->get_gui_stringvar( base, name, value );
}

int 
Part::get_gui_doublevar(const string &base, const string &name, double &value )
{
  return gm->get_gui_doublevar( base, name, value );
}

int 
Part::get_gui_intvar(const string &base, const string &name, int &value )
{
  return gm->get_gui_intvar( base, name, value );
}

void
Part::set_gui_var(const string &base, const string &name, const string &value )
{
  return gm->set_guivar( base, name, value );
}

} // namespace SCIRun


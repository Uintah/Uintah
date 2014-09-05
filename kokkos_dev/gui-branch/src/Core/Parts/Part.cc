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
#include <Core/Parts/GuiVar.h>
#include <Core/Parts/PartPort.h>
#include <Core/Framework/CoreFramework.h>
#include <Core/GuiInterface/GuiManager.h>

namespace SCIRun {

using namespace std;
  
Part::Part( Part *parent, const string &name, const string &type,
	    bool initialize)
  : name_(name), type_(type), parent_(parent)
{
  if ( initialize ) init();
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
  cerr << "Part add child " << child->name() << endl;
  children_.push_back(child);
  has_child ( child->get_port() );
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
  map<string,GuiVar *>::iterator i;
  for ( i=vars_.begin(); i!=vars_.end(); i++)
    i->second->emit(out,midx);
}

void 
Part::add_gui_var( GuiVar *v )
{
  vars_[ v->name() ] = v;
}

void 
Part::rem_gui_var( GuiVar *v )
{
  map<string,GuiVar *>::iterator i = vars_.find(v->name());
  if ( i != vars_.end() )
    vars_.erase( i );
}


// tcl compatibity

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
  gm->set_guivar( base, name, value );
}

void
Part::var_set( GuiVar *var )
{
  port_->var_set_signal( var );
}

void 
Part::command( const string &cmd )
{
  port_->command_signal( cmd );
}

void
Part::eval( const string &cmd, string &results )
{
  port_->eval_signal( cmd, results );
}

GuiVar *
Part::find_var( const string &name ) 
{
  VarList::iterator i = vars_.find(name);
  if ( i == vars_.end() )
    return 0;
  else
    return i->second;
}

// 
// PartPort
//

const string &
PartPort::name()
{
  return part_->name();
} 

const string &
PartPort::type()
{
  return part_->type();
} 

void
PartPort::command( TCLArgs &args ) 
{
  part_->tcl_command( args, 0 );
}

template<> void
PartPort::set_var( const string &name, const string &value )
{
  GuiVar *var = part_->find_var<GuiVar>(name);
  if ( !var ) {
    cerr << "PartPort::set_var : Error- gui var " << name <<" does not exit\n";
  } else {
    var->string_set( value );
  }
}

} // namespace SCIRun


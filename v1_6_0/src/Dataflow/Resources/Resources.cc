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

// Resources.cc - Interface to module-finding and loading mechanisms

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <ctype.h>
using std::cerr;
using std::ostream;
using std::endl;
using std::cout;
#include <sys/stat.h>
#include <set>
using std::set;

#include <Dataflow/Resources/Resources.h>
#include <Dataflow/Resources/ResourcesParser.h>

namespace SCIRun {


Resources resources;

void post_message(const string &msg )
{
  cerr << msg << endl;
}

Resources::Resources() 
{
}

Resources::~Resources()
{
}


void
Resources::read( const string &file )
{
  ResourcesParser parser( this );
  parser.parse( file );
}

/*
 * General Info
 */

PackageInfo *
Resources::get_package_info( const string &type )
{
  PackagesList::iterator i = packages_.find( type );
  return i == packages_.end() ? 0 : i->second;
}

ModuleInfo *
Resources::get_module_info( const string &type )
{
  ModulesList::iterator i = modules_.find( type );
  return i == modules_.end() ? 0 : i->second;
} 


PortInfo *
Resources::get_port_info( const string &type )
{
  PortsList::iterator i = ports_.find( type );
  return i == ports_.end() ? 0 : i->second;
} 

/*
 * for compatability
 */

vector<string> 
Resources::get_packages_names()
{
  vector<string> names;

  PackagesList::iterator i;
  for (i=packages_.begin(); i != packages_.end(); i++)
    names.push_back( i->second->name_ );
  return names;
}

vector<string>
Resources::get_modules_names( const string &package, const string &cat )
{
  vector<string> names;

  ModulesList::iterator i;
  for (i=modules_.begin(); i!=modules_.end(); i++)
    if (i->second->package_ == package && i->second->categories_[0] == cat )
      names.push_back( i->second->name_ );
  return names;
}

vector<string>
Resources::get_categories_names( const string &package )
{
  set<string> a_set;

  ModulesList::iterator i;
  for (i=modules_.begin(); i!=modules_.end(); i++)
    if (i->second->package_ == package )
      a_set.insert( i->second->categories_[0] );
  
  vector<string> names;
  for (set<string>::iterator i=a_set.begin(); i!=a_set.end(); i++)
    names.push_back(*i);

  return names;
}

/*
 * Package
 */

string 
Resources::get_package_ui_path( const string &name )
{
  PackageInfo *info = get_package_info( name );
  return info ? info->ui_path_ : "";
}

/*
 * Module
 */


const vector<string> &
Resources::get_module_libs( const string &type )
{
  static vector<string> none;

  ModuleInfo *info = get_module_info( type );
  if ( !info ) {
    cerr << "get_module_libs: can not find info on module " << type << endl;
    return none;
  }
  else
    return info->libs_;
}

string 
Resources::get_module_ui( const string &type )
{
  ModuleInfo *info = get_module_info( type );

  return info ? info->ui_ : "";
}  

string
Resources::get_module_maker( const string &type )
{
  ModuleInfo *info = get_module_info( type );
  return info ? info->maker_ : "";
}


/*
 * Port
 */

const vector<string> &
Resources::get_port_libs( const string &type )
{
  vector<string> none;
  PortInfo *info = get_port_info( type );
  if (info) return info->libs_;
  
  cerr << "get_port_info for " << type << " not found\n";
  return none;
}

string
Resources::get_iport_maker( const string &type )
{
  PortInfo *info = get_port_info( type );
  return info ? info->imaker_ : "";
}

string
Resources::get_oport_maker( const string &type )
{
  PortInfo *info = get_port_info( type );
  return info ? info->omaker_ : "";
}

#if 0
    // libraries names
    string package("");
    if ( module_info->package != "SCIRun" )
      package = "Packages_"+module_info->package +"_";
    package += "Dataflow";

    module_info->large_lib = string("lib/lib")+package+".so";
    module_info->small_lib = string("lib/lib")+package
      +"_Modules_" + module_info->category +".so";

    // maker name
    module_info->maker = string("make_") + module_info->name;
  }

*******************

    // make iport & oport;
    PortInfo *port = new PortInfo;
    port->datatype = datatype;
    port->name = package+"::"+name;
    port->type = type;
    port->colorname = color.latin1();
    qApp->lock();
    port->color.setNamedColor( color );
    qApp->unlock();
 
    // library name
 
    string lib;
    if (package != "SCIRun" && package != "Dataflow" )
      lib = "Packages_"+package +"_Dataflow";
    else
      lib = "Dataflow";
 
    port->large_lib = string("lib/lib")+lib+".so";
    port->small_lib = string("lib/lib")+lib+"_Ports.so";
 
#endif

} // End namespace SCIRun




#include <map>
#include <iostream>
#include <dlfcn.h>   // dlopen & dlsym#include <Core/Malloc/Allocator.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Resources/Resources.h>
#include <Dataflow/Network/Services.h>
#include <Dataflow/Network/Module.h>

namespace SCIRun {

using std::map;

typedef Module* (*ModuleMaker)(const string& id);
typedef IPort* (*IPortMaker)(Module*, const string&);
typedef OPort* (*OPortMaker)(Module*, const string&);
typedef void* LibHandle;


typedef map<string, ModuleMaker> ModuleMakerMap;
typedef map<string, IPortMaker> IPortMakerMap;
typedef map<string, OPortMaker> OPortMakerMap;
typedef map<string, LibHandle> LibMap;

class ServicesDB {
public:
  ServicesDB() {}

public:  
  ModuleMakerMap module_maker_;
  IPortMakerMap iport_maker_;
  OPortMakerMap oport_maker_;
  LibMap libs_;

  ModuleMaker find_module_maker( const string &);
  IPortMaker  find_iport_maker( const string &);
  OPortMaker  find_oport_maker( const string &);
  LibHandle   get_lib( const vector<string> &);
};

ModuleMaker
ServicesDB::find_module_maker( const string &name )
{
  ModuleMakerMap::iterator i = module_maker_.find(name);
  if ( i != module_maker_.end() )
    return i->second;
  return 0;
}

IPortMaker
ServicesDB::find_iport_maker( const string &name )
{
  IPortMakerMap::iterator i = iport_maker_.find(name);
  if ( i != iport_maker_.end() )
    return i->second;
  return 0;
}

OPortMaker
ServicesDB::find_oport_maker( const string &name )
{
  OPortMakerMap::iterator i = oport_maker_.find(name);
  if ( i != oport_maker_.end() )
    return i->second;
  return 0;
}

LibHandle
ServicesDB::get_lib( const vector<string> &names )
{
  LibHandle lib =0;
  
  // do we have any of these libs ?
  for (unsigned i=0; i<names.size(); i++ ) {
    LibMap::iterator l = libs_.find(names[i]);
    if ( l != libs_.end() && l->second )
      return l->second;  // found a lib
  }

  // none of these libs was loaded. 
  for (unsigned i=0; i<names.size(); i++ ) {
    lib =  dlopen( names[i].c_str(), RTLD_LAZY );
    libs_[names[i]] = lib; 
    if ( lib ) 
      return lib;
    else
      cerr << "Loading lib error: " << dlerror() << endl;
  }
  
  return 0; // lib not found 
} 

/*
 * Services
 */

Services::Services()
{
  db_ = scinew ServicesDB;
}

Module *
Services::make_module( const string &type, const string &name ) 
{
  static int counter=0;

  // was a lib with this module loaded already ?
  ModuleMaker maker = db_->find_module_maker( type );

  if ( !maker ) {
    // need to load a library.
    const vector<string> &libs_names = resources.get_module_libs( type );
    cerr << "looking for libs ";
    for (unsigned i=0; i<libs_names.size(); i++)
      cerr << "["<<libs_names[i] << "] ";
    cerr << endl;
    
    LibHandle lib = db_->get_lib( libs_names );
    
    if ( !lib ) {
      // can load object
      cerr << "Services[make_module]: can not create module " << type << endl;
      return 0;
    }

    // look for the maker function
    string maker_name = resources.get_module_maker( type );
    maker= (ModuleMaker)dlsym( lib, maker_name.c_str() );

    // register maker
    db_->module_maker_[type] = maker;
  }

  if ( !maker ) {
    // nope. can't get maker
    cerr << "can not find a maker for " << type << endl;;
    return 0;
  }

  string instant = (name=="") ? type+"_"+to_string(counter++) : name ;
    
  // maker found. make a module
  Module *module = (maker)( instant );
  module->type = type;

  return module;  
}

IPort *
Services::make_iport( const string &type, const string &name, Module *module )
{
  IPortMaker maker = db_->find_iport_maker( type );

  if ( !maker ) {
    // need to load a library.
    LibHandle lib = db_->get_lib( resources.get_port_libs( type ) );
    
    if ( !lib ) {
      // can load object
      cerr << "Services[make_module]: can not create port " << type << endl;
      return 0;
    }

    // look for the maker function
    string maker_name = resources.get_iport_maker( type );
    maker= (IPortMaker)dlsym( lib, maker_name.c_str() );
    // register maker
    db_->iport_maker_[type] = maker;
  }

  if ( !maker ) {
    // nope. can't get maker
    cerr << "can not find a maker for " << type << endl;;
    return 0;
  }

  
  // maker found. make a port
  return (maker)( module, name );
}

OPort *
Services::make_oport( const string &type, const string &name, Module *module )
{
  OPortMaker maker = db_->find_oport_maker( type );

  if ( !maker ) {
    // need to load a library.
    LibHandle lib = db_->get_lib( resources.get_port_libs( type ) );
    
    if ( !lib ) {
      // can load object
      cerr << "Services[make_module]: can not create port " << type << endl;
      return 0;
    }

    // look for the maker function
    string maker_name = resources.get_oport_maker( type );
    maker= (OPortMaker)dlsym( lib, maker_name.c_str() );
    // register maker
    db_->oport_maker_[type] = maker;
  }

  if ( !maker ) {
    // nope. can't get maker
    cerr << "can not find a maker for " << type << endl;;
    return 0;
  }

  
  // maker found. make a port
  return (maker)( module, name );
}


Services services;
 
} // namespace SCIRun


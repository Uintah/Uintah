#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <Core/Exceptions/InvalidValue.h>
#include <sstream>
#include <iostream>
#include <stdexcept>

//===========================================================================

using namespace Uintah;

SourceTermFactory::SourceTermFactory()
{}

SourceTermFactory::~SourceTermFactory()
{
  // delete the builders
  for( BuildMap::iterator i=_builders.begin(); i!=_builders.end(); ++i ){
      delete i->second;
  }

  // delete all constructed solvers
  for( SourceMap::iterator i=_sources.begin(); i!=_sources.end(); ++i ){
      delete i->second;
  }
}

//---------------------------------------------------------------------------
// Method: Return a reference to itself. 
//---------------------------------------------------------------------------
SourceTermFactory&
SourceTermFactory::self()
{
  static SourceTermFactory s;
  return s;
}
//---------------------------------------------------------------------------
// Method: Register a source term
//---------------------------------------------------------------------------
void
SourceTermFactory::register_source_term( const std::string name,
                                         SourceTermBase::Builder* builder )
{

  ASSERT( builder != NULL );

  BuildMap::iterator i = _builders.find( name );
  if( i == _builders.end() ){
    i = _builders.insert( std::make_pair(name,builder) ).first;
  }
  else{
    string errmsg = "ERROR: Arches: SourceTermBuilder: A duplicate SourceTermBuilder object was loaded on equation\n";
    errmsg += "\t\t " + name + ". This is forbidden.\n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }
}
//---------------------------------------------------------------------------
// Method: Retrieve a source term from the map. 
//---------------------------------------------------------------------------
SourceTermBase&
SourceTermFactory::retrieve_source_term( const std::string name )
{
  const SourceMap::iterator isource= _sources.find( name );

  if( isource != _sources.end() ) return *(isource->second);

  const BuildMap::iterator ibuilder = _builders.find( name );

  if( ibuilder == _builders.end() ){
    string errmsg = "ERROR: Arches: SourceTermBuilder: No source term registered for " + name + "\n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }

  SourceTermBase::Builder* builder = ibuilder->second;
  SourceTermBase* prop = builder->build();
  _sources[name] = prop;

  return *prop;
}

//---------------------------------------------------------------------------
// Method: Find if a property model is included in the map. 
//---------------------------------------------------------------------------
bool
SourceTermFactory::source_term_exists( const std::string name )
{
  bool return_value;

  const SourceMap::iterator isource= _sources.find( name );

  if( isource != _sources.end() ) {
    return_value = true;
  } else {
    return_value = false;
  }

  return return_value;
}


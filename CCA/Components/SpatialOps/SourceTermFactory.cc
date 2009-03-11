#include <CCA/Components/SpatialOps/SourceTermFactory.h>
#include <CCA/Components/SpatialOps/SourceTermBase.h> 
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
  for( BuildMap::iterator i=builders_.begin(); i!=builders_.end(); ++i ){
    //delete *i;
    }

  // delete all constructed solvers
  for( SourceMap::iterator i=sources_.begin(); i!=sources_.end(); ++i ){
    //delete *i;
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
                                         SourceTermBuilder* builder )
{

  ASSERT( builder != NULL );

  BuildMap::iterator i = builders_.find( name );
  if( i == builders_.end() ){
    i = builders_.insert( std::make_pair(name,builder) ).first;
  }
  else{
    std::ostringstream errmsg;
    std::cout << "ERROR: A duplicate SourceTermBuilder object was loaded: " << std::endl
     << "       " << name << ".  This is forbidden." << std::endl;
    throw std::runtime_error( errmsg.str() );
  }
}
//---------------------------------------------------------------------------
// Method: Retrieve a source term from the map. 
//---------------------------------------------------------------------------
SourceTermBase&
SourceTermFactory::retrieve_source_term( const std::string name )
{
  const SourceMap::iterator isource= sources_.find( name );

  if( isource != sources_.end() ) return *(isource->second);

  const BuildMap::iterator ibuilder = builders_.find( name );

  if( ibuilder == builders_.end() ){
    std::ostringstream errmsg;
    errmsg << "ERROR: No source term registered for " << name << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  SourceTermBuilder* builder = ibuilder->second;
  SourceTermBase* src = builder->build();
  sources_[name] = src;

  return *src;
}

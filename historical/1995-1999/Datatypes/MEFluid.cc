#include <Datatypes/MEFluid.h>
#include <Malloc/Allocator.h>
#include <iostream.h>
#include <fstream.h>
#include <strstream.h>
#include <Classlib/NotFinished.h>
#include <string.h>



PersistentTypeID MEFluid::type_id("MEFluid",
				  "Datatype", 0);

MEFluid::~MEFluid()
{
}

MEFluid::MEFluid(){
ps = 0;
}

MEFluid* MEFluid::clone() const
{
  return scinew MEFluid();
}

void MEFluid::AddVectorField(const clString& name,
			     VectorFieldHandle vfh)
{
  vectorvars.add( name );
  vfs.add(vfh);
}

void MEFluid::AddScalarField(const clString& name,
			     ScalarFieldHandle sfh)
{
  scalarvars.add( name );
  sfs.add(sfh);
}

void MEFluid::AddParticleSet(ParticleSetHandle psh)
{
  ps = psh;
}

ParticleSetHandle MEFluid::getParticleSet()
{
  return ps;
}

void MEFluid::getVectorVars( Array1< clString>& vars)
{
  vars.setsize(0);
  for(int i = 0; i < vectorvars.size(); i++)
    vars.add( vectorvars[i] );
}

void MEFluid::getScalarVars( Array1< clString>& vars)
{
  vars.setsize(0);
  for(int i = 0; i < scalarvars.size(); i++)
    vars.add( scalarvars[i] );
}

VectorFieldHandle MEFluid::getVectorField( clString name )
{
  int i;
  for( i = 0; i < vectorvars.size(); i++)
    if( vectorvars[i] == name )
      return vfs[i];

  return 0;
}     

ScalarFieldHandle MEFluid::getScalarField( clString name )
{
  int i;
  for( i = 0; i < scalarvars.size(); i++)
    if( scalarvars[i] == name )
      return sfs[i];

  return 0;
}     


#define MEFLUID_VERSION 1

void MEFluid::io(Piostream& stream)
{
  stream.begin_class("MEFluid", MEFLUID_VERSION);

  stream.end_class();
}

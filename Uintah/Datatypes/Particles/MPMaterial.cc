//static char *id="@(#) $Id$";

/*
 *  MPMaterial.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <iostream.h>
#include <fstream.h>
#include <strstream.h>
#include <string.h>

#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Util/NotFinished.h>

#include <Uintah/Datatypes/Particles/MPMaterial.h>

namespace Uintah {
namespace Datatypes {

PersistentTypeID MPMaterial::type_id("MPMaterial",
				  "Datatype", 0);

MPMaterial::~MPMaterial()
{
}

MPMaterial::MPMaterial(){
ps = 0;
}

MPMaterial* MPMaterial::clone() const
{
  return scinew MPMaterial();
}

void MPMaterial::AddVectorField(const clString& name,
			     VectorFieldHandle vfh)
{
  vmap[ name ] = vfh;
}

void MPMaterial::AddScalarField(const clString& name,
			     ScalarFieldHandle sfh)
{
  smap[ name ] = sfh;
}

void MPMaterial::AddParticleSet(ParticleSetHandle psh)
{
  ps = psh;
}

ParticleSetHandle MPMaterial::getParticleSet()
{
  return ps;
}

void MPMaterial::getVectorNames( Array1< clString>& vars)
{
  vars.setsize(0);
  map<clString, VectorFieldHandle, ltstr>::iterator it = vmap.begin();

  while( it != vmap.end())
    {
      vars.add( (*it).first );
      it++;
    } 

}

void MPMaterial::getScalarNames( Array1< clString>& vars)
{
  vars.setsize(0);
  map<clString, ScalarFieldHandle, ltstr>::iterator it = smap.begin();
  
  while( it != smap.end())
    {
      vars.add( (*it).first );
      it++;
    }

}

VectorFieldHandle MPMaterial::getVectorField( clString name )
{
  map<clString, VectorFieldHandle, ltstr>::iterator it = vmap.find( name );
  if (it == vmap.end())
    return 0;
  else
    return vmap[ name ];
}     

ScalarFieldHandle MPMaterial::getScalarField( clString name )
{
  map<clString, ScalarFieldHandle, ltstr>::iterator it = smap.find( name );
  if (it == smap.end())
    return 0;
  else
    return smap[ name ];
}     


#define MPMATERIAL_VERSION 1

void MPMaterial::io(Piostream& stream)
{
  stream.begin_class("MPMaterial", MPMATERIAL_VERSION);

  stream.end_class();
}

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.2  1999/08/17 06:40:06  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:58  mcq
// Initial commit
//
// Revision 1.1  1999/06/09 23:21:32  kuzimmer
// reformed the material/particle classes and removed the particleSetExtensions.  Now MPVizParticleSet inherits from cfdlibParticleSet-->use the new stl routines to dynamically check the particleSet type
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//

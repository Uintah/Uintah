//static char *id="@(#) $Id$";

/*
 *  MPVizGrid.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include "MPVizGrid.h"

namespace Uintah {
namespace Datatypes {

PersistentTypeID MPVizGrid::type_id("MPVizGrid",
				  "Datatype", 0);

MPVizGrid::~MPVizGrid()
{
}

MPVizGrid::MPVizGrid() : name("")
{
}

MPVizGrid::MPVizGrid(clString n) :  name(n)
{
}

VizGrid* MPVizGrid::clone() const
{
  return new MPVizGrid();
}

void MPVizGrid::AddVectorField(const clString& name,
			     VectorFieldHandle vfh)
{
  vmap[ name ] = vfh;
}

void MPVizGrid::AddScalarField(const clString& name,
			     ScalarFieldHandle sfh)
{
  smap[ name ] = sfh;
}


void MPVizGrid::getVectorNames( Array1< clString>& vars)
{
  vars.setsize(0);
  map<clString, VectorFieldHandle, ltstr>::iterator it = vmap.begin();

  while( it != vmap.end())
    {
      vars.add( (*it).first );
      it++;
    } 

}

void MPVizGrid::getScalarNames( Array1< clString>& vars)
{
  vars.setsize(0);
  map<clString, ScalarFieldHandle, ltstr>::iterator it = smap.begin();
  
  while( it != smap.end())
    {
      vars.add( (*it).first );
      it++;
    }

}

VectorFieldHandle MPVizGrid::getVectorField( clString name )
{
  map<clString, VectorFieldHandle, ltstr>::iterator it = vmap.find( name );
  if (it == vmap.end())
    return 0;
  else
    return vmap[ name ];
}     

ScalarFieldHandle MPVizGrid::getScalarField( clString name )
{
  map<clString, ScalarFieldHandle, ltstr>::iterator it = smap.find( name );
  if (it == smap.end())
    return 0;
  else
    return smap[ name ];
}     


#define MPMATERIAL_VERSION 1

void MPVizGrid::io(Piostream& stream)
{
  stream.begin_class("MPVizGrid", MPMATERIAL_VERSION);

  stream.end_class();
}

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.1  1999/09/21 16:08:29  kuzimmer
// modifications for binary file format
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

//static char *id="@(#) $Id$";

/*
 *  MPVizParticleSet.cc
 *
 *  Written by:
 *   Author: Kurt Zimmerman
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Util/NotFinished.h>

#include <Uintah/Datatypes/Particles/MPVizParticleSet.h>

namespace Uintah {
namespace Datatypes {

PersistentTypeID MPVizParticleSet::type_id("MPVizParticleSet", "Datatype", 0);

MPVizParticleSet::MPVizParticleSet( clString scalarVar,
				      clString vectorVar,
				      void* cbClass) :
 name(clString("")), sVar( scalarVar ), vVar( vectorVar), cbClass( cbClass )
{
  // No body
}

 MPVizParticleSet::MPVizParticleSet( clString name,
				    clString scalarVar,
				      clString vectorVar,
				      void* cbClass) :
  name(name),  sVar( scalarVar ), vVar( vectorVar), cbClass( cbClass )
{
  // No body
}
 

MPVizParticleSet::MPVizParticleSet(const MPVizParticleSet& pset) :
  name(pset.name), sVar(pset.sVar),
  vVar(pset.vVar), cbClass(pset.cbClass)
{
  // No body
}

MPVizParticleSet::MPVizParticleSet() :
   name(""), sVar(""), vVar(""), cbClass(0)
{
  // No body
}

MPVizParticleSet::MPVizParticleSet(clString name) :
  name(name), sVar(""), vVar(""), cbClass(0)
{
  // No body
}

MPVizParticleSet::~MPVizParticleSet()
{
  // No body
}

void MPVizParticleSet::SetScalarId(const  clString& id)
{
  sVar = id;
}

void MPVizParticleSet::SetVectorId(const  clString& id)
{
  vVar = id;
}

void MPVizParticleSet::SetCallback( void* f )
{
  cbClass = f;
}

const clString& MPVizParticleSet::getScalarId()
{
  return sVar;
}
const clString& MPVizParticleSet::getVectorId()
{
  return vVar;
  }

void* MPVizParticleSet::getCallbackClass()
{
  return cbClass;
}

#define MPVISPARTICLESET_VERSION 1

void MPVizParticleSet::io(Piostream& stream)
{
  NOT_FINISHED( "MPVizParticleSet::io" );
  stream.begin_class("MPVizParticleSet", MPVISPARTICLESET_VERSION);
    stream.end_class();
}

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.4  1999/09/21 16:08:30  kuzimmer
// modifications for binary file format
//
// Revision 1.3  1999/09/08 02:27:07  sparker
// Various #include cleanups
//
// Revision 1.2  1999/08/17 06:40:07  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:59  mcq
// Initial commit
//
// Revision 1.1  1999/06/09 23:21:33  kuzimmer
// reformed the material/particle classes and removed the particleSetExtensions.  Now MPVizParticleSet inherits from cfdlibParticleSet-->use the new stl routines to dynamically check the particleSet type
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//

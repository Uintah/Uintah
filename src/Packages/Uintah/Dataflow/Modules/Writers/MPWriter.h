//static char *id="@(#) $Id$";

/****************************************
CLASS
    MPWriter

    A class for writing Material/Particle files.

OVERVIEW TEXT

KEYWORDS
    Material/Particle Method

AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    June 1999

    Copyright (C) 1999 SCI Group

LOG
    June 28, 1999
****************************************/
#ifndef PARTICLEGRID_H
#define PARTICLEGRID_H 1



#include <Uintah/Datatypes/Particles/ParticleGridReader.h>
#include <Uintah/Datatypes/Particles/ParticleGridReaderPort.h>
#include <Uintah/Datatypes/Particles/ParticleSetPort.h>
#include <Uintah/Datatypes/Particles/VizGrid.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <PSECore/Dataflow/Module.h> 
#include <SCICore/TclInterface/TCLvar.h> 
#include <SCICore/Containers/String.h>
  
namespace Uintah {
namespace Modules {

using namespace Uintah::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;


class MPWriter:public Module {
 public:
  

  // GROUP: Constructors
  //////////
  MPWriter(const clString& id); 

  // GROUP: Destructors
  //////////
  virtual ~MPWriter(); 

  // GROUP: cloning and execution 
  ////////// 
  virtual void execute(); 

  //////////
  // overides tcl_command in base class Module
  void tcl_command( TCLArgs&, void* );


protected:
  
private:

  TCLstring tcl_status;
  TCLstring sVar;
  TCLstring vVar;
  TCLstring psVar;
  TCLstring pvVar;
  TCLint sMaterial;
  TCLint vMaterial;
  TCLint pMaterial;

  ParticleGridReaderIPort *in;
 
  void SaveFile(clString fname, int isBin);
  ParticleGridReaderHandle pgrh;
  void setVars(ParticleGridReaderHandle reader);
  void checkVars(ParticleGridReaderHandle reader );
}; //class 

} // end namespace Modules
} // end namespace Uintah

//
// $Log$
// Revision 1.1  1999/09/21 16:12:27  kuzimmer
// changes made to support binary/ASCII file IO
//
// Revision 1.2  1999/06/09 23:23:44  kuzimmer
// Modified the modules to work with the new Material/Particle classes.  When a module needs to determine the type of particleSet that is incoming, the new stl dynamic type testing is used.  Works good so far.
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//

#endif

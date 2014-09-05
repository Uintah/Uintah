/****************************************
CLASS
    ParticleGridVisControl

    Visualization control for simulation data that contains
    information on both a regular grid in particle sets.

OVERVIEW TEXT
    This module receives a ParticleGridReader object.  The user
    interface is dynamically created based information provided by the
    ParticleGridReader.  The user can then select which variables he/she
    wishes to view in a visualization.



KEYWORDS
    ParticleGridReader, Material/Particle Method

AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 5, 1999
****************************************/
#ifndef PARTICLEGRIDVISCONTROL_H
#define PARTICLEGRIDVISCONTROL_H 1

#include <SCICore/TclInterface/TCLvar.h> 

#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <PSECore/Dataflow/Module.h> 

#include <Uintah/Datatypes/Particles/ParticleGridReader.h>
#include <Uintah/Datatypes/Particles/ParticleGridReaderPort.h>
#include <Uintah/Datatypes/Particles/ParticleSetPort.h>
  
namespace Uintah {
namespace Modules {

using namespace Uintah::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;

class ParticleGridVisControl : public Module { 
  
public: 

  // GROUP: Constructors
  //////////
  ParticleGridVisControl(const clString& id); 

  // GROUP: Destructors
  //////////
  virtual ~ParticleGridVisControl(); 

  // GROUP: execution 
  ////////// 
  virtual void execute(); 

  //////////
  // overides tcl_command in base class Module
  void tcl_command( TCLArgs&, void* );

  //////////
  // callback taking
  // [in] index--an index into the particle set.
  void callback( int index);


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
  ScalarFieldOPort *sfout;
  VectorFieldOPort *vfout;
  ParticleSetOPort *psout;


  ParticleGridReaderHandle pgrh;
  void setVars(ParticleGridReaderHandle reader);
  void checkVars(ParticleGridReaderHandle reader );
  void graph(clString, clString);
}; //class 

} // end namespace Modules
} // end namespace Uintah

//
// $Log$
// Revision 1.4  1999/08/25 03:49:03  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.3  1999/08/18 20:20:22  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:40:11  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 17:08:57  mcq
// Initial commit
//
// Revision 1.2  1999/06/09 23:23:44  kuzimmer
// Modified the modules to work with the new Material/Particle classes.  When a module needs to determine the type of particleSet that is incoming, the new stl dynamic type testing is used.  Works good so far.
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//

#endif

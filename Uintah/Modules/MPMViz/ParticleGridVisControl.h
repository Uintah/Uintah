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

#include <PSECore/CommonDatatypes/ScalarFieldPort.h>
#include <PSECore/CommonDatatypes/VectorFieldPort.h>
#include <PSECore/Dataflow/Module.h> 

#include <Uintah/Datatypes/Particles/ParticleGridReader.h>
#include <Uintah/Datatypes/Particles/ParticleGridReaderPort.h>
#include <Uintah/Datatypes/Particles/ParticleSetPort.h>
  
namespace Uintah {
namespace Modules {

using namespace Uintah::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;

class ParticleGridVisControl : public Module { 
  
public: 

  // GROUP: Constructors
  //////////
  ParticleGridVisControl(const clString& id); 
  ParticleGridVisControl(const ParticleGridVisControl&, int deep); 

  // GROUP: Destructors
  //////////
  virtual ~ParticleGridVisControl(); 

  // GROUP: cloning and execution 
  ////////// 
  virtual Module* clone(int deep); 
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

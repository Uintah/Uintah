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


#include <Datatypes/ParticleGridReader.h>
#include <Datatypes/ParticleGridReaderPort.h>
#include <Datatypes/ParticleSetPort.h>
#include <Datatypes/ParticleSetExtensionPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/VectorFieldPort.h>
#include <Dataflow/Module.h> 
#include <TCL/TCLvar.h> 

  
namespace SCI {
namespace CFD {


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
  TCLint sFluid;
  TCLint vFluid;
  TCLint pFluid;

  ParticleGridReaderIPort *in;
  ScalarFieldOPort *sfout;
  VectorFieldOPort *vfout;
  ParticleSetOPort *psout;
  ParticleSetExtensionOPort *pseout;


  ParticleGridReaderHandle pgrh;
  void setVars(ParticleGridReaderHandle reader);
  void checkVars(ParticleGridReaderHandle reader );
  void graph(clString, clString);
}; //class 

} // end namespace CFD
} // end namespace SCI

#endif

#ifndef PARTICLEGRIDVISCONTROL_H
#define PARTICLEGRIDVISCONTROL_H


/*----------------------------------------------------------------------
CLASS
    ParticleGridVisControl

    Select Tecplot files for use in visualization and animation.

OVERVIEW TEXT
    It simply allows to user to select a Tecplot file for use in
    visualization.  This class then creates a TecplotReader
    datatype (a subclass of ParticleGridReader) and sends it out
    the output port.



KEYWORDS
    ParticleGridSelector

AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 5, 1999
----------------------------------------------------------------------*/
    
#include <Classlib/NotFinished.h> 
#include <Datatypes/ParticleGridReaderPort.h>
#include <Datatypes/ParticleSetPort.h>
#include <Datatypes/ParticleSetExtensionPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/VectorFieldPort.h>
#include <Datatypes/ParticleGridReader.h>
#include <Dataflow/Module.h> 
#include <TCL/TCLvar.h> 
#include <iostream.h> 
  


class ParticleGridVisControl : public Module { 
  
public: 
  


  ////////// Constructors
  ParticleGridVisControl(const clString& id); 
  ParticleGridVisControl(const ParticleGridVisControl&, int deep); 
  virtual ~ParticleGridVisControl(); 
  virtual Module* clone(int deep); 
  void tcl_command( TCLArgs&, void* );
  virtual void execute(); 
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
  void setVars(ParticleGridReader *reader);
  void checkVars(ParticleGridReader *reader );
  void graph(clString, clString);
}; //class 

#endif

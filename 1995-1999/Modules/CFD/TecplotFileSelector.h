#ifndef TECPLOTFILESELECTOR_H
#define TECPLOTFILESELECTOR_H


/*----------------------------------------------------------------------
CLASS
    TecplotFileSelector

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
#include <Dataflow/Module.h> 
#include <TCL/TCLvar.h> 
  
class ParticleGridReaderPort;
class ParticleGridReader;


class TecplotFileSelector : public Module { 
  
public: 
  
  TCLstring tcl_status;
  TCLstring filebase; 
  TCLint animate;
  TCLint startFrame;
  TCLint endFrame;
  TCLint increment;
  ////////// Constructors
  TecplotFileSelector(const clString& id); 
  TecplotFileSelector(const TecplotFileSelector&, int deep); 
  virtual ~TecplotFileSelector(); 
  virtual Module* clone(int deep); 
  virtual void execute(); 

protected:
private:
  bool checkFile(const clString& fname);
  void doAnimation();
  ParticleGridReaderOPort *out;
  ParticleGridReader *reader;
  
}; //class 

#endif

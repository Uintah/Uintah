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
    
#include <Datatypes/Particles/ParticleGridReaderPort.h>

#include <Util/NotFinished.h> 
#include <Dataflow/Module.h> 
#include <TclInterface/TCLvar.h> 

namespace Uintah {
namespace Modules {

using namespace Uintah::Datatypes;
using namespace PSECommon::Dataflow;
using namespace PSECommon::CommonDatatypes;
using namespace SCICore::TclInterface;

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
  
}; //class TecplotFileSelector

} // End namespace Modules
} // End namespace Uintah

//
// $Log$
// Revision 1.1  1999/07/27 17:08:58  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//

#endif

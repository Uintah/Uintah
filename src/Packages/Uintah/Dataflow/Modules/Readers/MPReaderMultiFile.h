#ifndef MPREADERMULTIFILE_H
#define MPREADERMULTIFILE_H
/*----------------------------------------------------------------------
CLASS
    MPReader

    Select a set of files that represent one data set, output from a parallel
    run.

OVERVIEW TEXT
    It simply allows to user to select set of MaterialParticle files, 
    representing one dataset, for use in
    visualization.  This class then creates an MPParticleGridReader
    datatype (a subclass of ParticleGridReader) and sends it out
    the output port.



KEYWORDS
    

AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created November 30, 1999
----------------------------------------------------------------------*/
    
#include <Uintah/Datatypes/Particles/ParticleGridReaderPort.h>
#include <SCICore/Util/NotFinished.h> 
#include <PSECore/Dataflow/Module.h> 
#include <SCICore/TclInterface/TCLvar.h> 

namespace Uintah {
namespace Modules {

using namespace Uintah::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;

class MPReaderMultiFile : public Module { 
  
public: 
  
  TCLstring tcl_status;
  TCLstring filebase;
  TCLstring dirbase;
  TCLstring timestep;
  TCLint animate;
  TCLint startFrame;
  TCLint endFrame;
  TCLint increment;
  ////////// Constructors
  MPReaderMultiFile(const clString& id); 
  virtual ~MPReaderMultiFile(); 
  virtual void execute();

  //////////
  // overides tcl_command in base class Module
  void tcl_command( TCLArgs&, void* );


protected:
private:
  bool checkFile(const clString& fname);
  void doAnimation();
  ParticleGridReaderOPort *out;
  ParticleGridReader *reader;
  
}; //class MPReaderMultiFile

} // End namespace Modules
} // End namespace Uintah


#endif

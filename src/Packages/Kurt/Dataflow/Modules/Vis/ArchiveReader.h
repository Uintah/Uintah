#ifndef ARCHIVEREADER_H
#define ARCHIVEREADER_H

/*----------------------------------------------------------------------
CLASS
    ArchiveReader

    Select Tecplot files for use in visualization and animation.

OVERVIEW TEXT
    It simply allows to user to select a MaterialParticle file for use in
    visualization.  This class then creates an MPParticleGridReader
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
    
#include <Kurt/DataArchive/ArchivePort.h>
#include <SCICore/Util/NotFinished.h> 
#include <PSECore/Dataflow/Module.h> 
#include <SCICore/TclInterface/TCLvar.h> 

namespace Uintah {
  class DataArchive;
}

namespace Kurt {
namespace Modules {


using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;

using Uintah::DataArchive;
class ArchiveReader : public Module { 
  
public: 
  
  TCLstring tcl_status;
  TCLstring filebase; 
  ////////// Constructors
  ArchiveReader(const clString& id); 
  virtual ~ArchiveReader(); 
  virtual void execute(); 

protected:
private:
  ArchiveOPort *out;
  DataArchive *reader;
  
}; //class ParticleGrid

} // End namespace Modules
} // End namespace Uintah

#endif

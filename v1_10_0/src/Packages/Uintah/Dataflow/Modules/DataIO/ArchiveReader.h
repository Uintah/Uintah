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
    Packages/Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 5, 1999
----------------------------------------------------------------------*/
    
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Core/Util/NotFinished.h> 
#include <Dataflow/Network/Module.h> 
#include <Core/GuiInterface/GuiVar.h> 

namespace Uintah {

using namespace SCIRun;

class ArchiveReader : public Module { 
  
public: 
  
  GuiString tcl_status;
  GuiString filebase; 
  ////////// Constructors
  ArchiveReader(GuiContext* ctx);
  virtual ~ArchiveReader(); 
  virtual void execute(); 

protected:
private:
  ArchiveOPort *out;
  ArchiveHandle archiveH;
  DataArchive *reader;
  
}; //class ParticleGrid
} // End namespace Uintah


#endif

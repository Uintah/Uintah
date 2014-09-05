#ifndef MPREADER_H
#define MPREADER_H

/*----------------------------------------------------------------------
CLASS
    MPReader

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

class MPReader : public Module { 
  
public: 
  
  TCLstring tcl_status;
  TCLstring filebase; 
  TCLint animate;
  TCLint startFrame;
  TCLint endFrame;
  TCLint increment;
  ////////// Constructors
  MPReader(const clString& id); 
  virtual ~MPReader(); 
  virtual void execute(); 

protected:
private:
  bool checkFile(const clString& fname);
  void doAnimation();
  ParticleGridReaderOPort *out;
  ParticleGridReader *reader;
  
}; //class ParticleGrid

} // End namespace Modules
} // End namespace Uintah

//
// $Log$
// Revision 1.1  1999/09/21 16:12:26  kuzimmer
// changes made to support binary/ASCII file IO
//
// Revision 1.1  1999/07/27 17:08:58  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//

#endif

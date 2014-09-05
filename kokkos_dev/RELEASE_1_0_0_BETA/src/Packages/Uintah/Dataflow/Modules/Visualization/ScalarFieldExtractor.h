/****************************************
CLASS
    ScalarFieldExtractor

    

OVERVIEW TEXT
    This module receives a DataArchive object.  The user
    interface is dynamically created based information provided by the
    DataArchive.  The user can then select which variables he/she
    wishes to view in a visualization.



KEYWORDS
    ParticleGridReader, Material/Particle Method

AUTHOR
    Packages/Kurt Zimmerman
    Department of Computer Science
    University of Utah
    June, 2000

    Copyright (C) 2000 SCI Group

LOG
    Created June 27, 2000
****************************************/
#ifndef SCALARFIELDEXTRACTOR_H
#define SCALARFIELDEXTRACTOR_H 1


#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Packages/Uintah/Core/Datatypes/ArchivePort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Network/Module.h> 
#include <Core/GuiInterface/GuiVar.h> 
#include <string>
#include <vector>


namespace Uintah {
using namespace SCIRun;

class ScalarFieldExtractor : public Module { 
  
public: 

  // GROUP: Constructors
  //////////
  ScalarFieldExtractor(const clString& id); 

  // GROUP: Destructors
  //////////
  virtual ~ScalarFieldExtractor(); 

  // GROUP: cloning and execution 
  ////////// 
  virtual void execute(); 

  //////////
  // overides tcl_command in base class Module
  //  void tcl_command( TCLArgs&, void* );

protected:
  
private:

  GuiString tcl_status;

  GuiString sVar;
  GuiInt sMatNum;

  const TypeDescription *type;

  ArchiveIPort *in;
  ScalarFieldOPort *sfout;
  
  std::string positionName;

  ArchiveHandle  archiveH;
  void setVars();

}; //class 
} // End namespace Uintah



#endif

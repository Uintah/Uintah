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


#include <Core/Datatypes/Archive.h>
#include <Dataflow/Ports/ArchivePort.h>
#include <Dataflow/Modules/Selectors/FieldExtractor.h>
#include <SCIRun/Dataflow/Network/Ports/FieldPort.h>
#include <SCIRun/Dataflow/GuiInterface/GuiVar.h> 
#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>


namespace Uintah {
using namespace SCIRun;

class ScalarFieldExtractor : public FieldExtractor { 
  
public: 

  // GROUP: Constructors
  //////////
  ScalarFieldExtractor(GuiContext* ctx);

  // GROUP: Destructors
  //////////
  virtual ~ScalarFieldExtractor(); 

  //////////
  // overides tcl_command in base class Module
  //  void tcl_command( TCLArgs&, void* );

protected:
  virtual void get_vars(vector< string >&,
                        vector< const TypeDescription *>&);
  
private:

}; //class 

} // End namespace Uintah



#endif

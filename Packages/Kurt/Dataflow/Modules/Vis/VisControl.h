/****************************************
CLASS
    VisControl

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
#ifndef VISCONTROL_H
#define VISCONTROL_H 1


#include <Kurt/DataArchive/Archive.h>
#include <Kurt/DataArchive/ArchivePort.h>
#include <Kurt/DataArchive/VisParticleSetPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <PSECore/Dataflow/Module.h> 
#include <SCICore/TclInterface/TCLvar.h> 
#include <string>
#include <vector>


namespace SCICore {
  namespace Datatypes {
  class ScalarFieldRGdouble;
  class VectorFieldRG;
  }
}

namespace Kurt {
  namespace Datatypes {
  class VisParticleSet;
  }
namespace Modules {
using Kurt::Datatypes::VisParticleSet;
using SCICore::Datatypes::ScalarFieldRGdouble;
using SCICore::Datatypes::VectorFieldRG;
using Uintah::DataArchive;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;

class VisControl : public Module { 
  
public: 

  // GROUP: Constructors
  //////////
  VisControl(const clString& id); 

  // GROUP: Destructors
  //////////
  virtual ~VisControl(); 

  // GROUP: cloning and execution 
  ////////// 
  virtual void execute(); 

  //////////
  // overides tcl_command in base class Module
  //  void tcl_command( TCLArgs&, void* );

  //////////
  // callback taking
  // [in] index--an index into the particle set.
  void callback( int index);

protected:
  
private:

  TCLstring tcl_status;

  TCLstring gsVar;
  TCLstring gvVar;
  TCLstring gtVar;
  TCLstring psVar;
  TCLstring pvVar;
  TCLstring ptVar;

  TCLint gsMatNum;
  TCLint pNMaterials;
  TCLint gvMatNum;
  TCLint animate;
  



  TCLint time;
  TCLdouble timeval;

  ArchiveIPort *in;
  ScalarFieldOPort *sfout;
  VectorFieldOPort *vfout;
  VisParticleSetOPort *psout;

  
  std::string positionName;

  ArchiveHandle archive;
  void setVars(ArchiveHandle ar);
  void buildData(DataArchive& archive, std::vector< double >& times,
		 int idx, ScalarFieldRGdouble*& sf,
		 VectorFieldRG*& vf, VisParticleSet*& vps);


  //  void graph(clString, clString);
}; //class 

} // end namespace Modules
} // end namespace Kurt


#endif

/****************************************
CLASS
    ParticleFieldExtractor

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
#ifndef PARTICLEFIELDEXTRACTOR_H
#define PARTICLEFIELDEXTRACTOR_H 1


#include <Uintah/Datatypes/Archive.h>
#include <Uintah/Datatypes/ArchivePort.h>
#include <Uintah/Datatypes/ScalarParticlesPort.h>
#include <Uintah/Datatypes/VectorParticlesPort.h>
#include <Uintah/Datatypes/TensorParticlesPort.h>
#include <PSECore/Dataflow/Module.h> 
#include <SCICore/TclInterface/TCLvar.h> 
#include <string>
#include <vector>



namespace Uintah {
  namespace Datatypes {
  class VisParticleSet;
  }
namespace Modules {

using namespace Uintah::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;

class ParticleFieldExtractor : public Module { 
  
public: 

  // GROUP: Constructors
  //////////
  ParticleFieldExtractor(const clString& id); 

  // GROUP: Destructors
  //////////
  virtual ~ParticleFieldExtractor(); 

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

  TCLstring psVar;
  TCLstring pvVar;
  TCLstring ptVar;

  TCLint pNMaterials;


  ArchiveIPort *in;
  VectorParticlesOPort *pvout;
  ScalarParticlesOPort *psout;
  TensorParticlesOPort *ptout;

  
  std::string positionName;

  ArchiveHandle archive;
  void setVars(ArchiveHandle ar);
  void buildData(DataArchive& archive, double time,
		 ScalarParticles*& sp,
		 VectorParticles*& vp,
		 TensorParticles*& tp);


  //  void graph(clString, clString);
}; //class 

} // end namespace Modules
} // end namespace Kurt


#endif

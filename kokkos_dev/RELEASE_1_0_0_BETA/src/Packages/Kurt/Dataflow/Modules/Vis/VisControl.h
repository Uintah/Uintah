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
    Packages/Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 5, 1999
****************************************/
#ifndef VISCONTROL_H
#define VISCONTROL_H 1


#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Packages/Uintah/Core/Datatypes/ArchivePort.h>
#include <Packages/Kurt/DataArchive/VisParticleSetPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Dataflow/Network/Module.h> 
#include <Core/GuiInterface/GuiVar.h> 
#include <string>
#include <vector>


namespace SCIRun {
  class ScalarFieldRGdouble;
  class VectorFieldRG;
}

namespace Kurt {
using namespace SCIRun;
using Uintah::DataArchive;
using namespace Uintah::Datatypes;

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

  GuiString tcl_status;

  GuiString gsVar;
  GuiString gvVar;
  GuiString gtVar;
  GuiString psVar;
  GuiString pvVar;
  GuiString ptVar;

  GuiInt gsMatNum;
  GuiInt pNMaterials;
  GuiInt gvMatNum;
  GuiInt animate;
  



  GuiInt time;
  GuiDouble timeval;

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


} // End namespace Kurt
  //  void graph(clString, clString);



#endif

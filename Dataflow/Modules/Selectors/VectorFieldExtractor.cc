/****************************************
CLASS
    VectorFieldExtractor

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
#include "VectorFieldExtractor.h"

#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Util/Timer.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Datatypes/LevelMesh.h>
#include <Packages/Uintah/Core/Datatypes/LevelField.h>
#include <Packages/Uintah/CCA/Ports/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Containers/ConsecutiveRangeSet.h>

#include <iostream> 
#include <sstream>
#include <string>

namespace Uintah {

using std::cerr;
using std::endl;
using std::vector;
using std::string;
using std::ostringstream;

using namespace SCIRun;

extern "C" Module* make_VectorFieldExtractor( const string& id ) {
  return scinew VectorFieldExtractor( id ); 
}

//--------------------------------------------------------------- 
VectorFieldExtractor::VectorFieldExtractor(const string& id) 
  : FieldExtractor("VectorFieldExtractor", id, "Selectors", "Uintah"),
    tcl_status("tcl_status", id, this), sVar("sVar", id, this),
    sMatNum("sMatNum", id, this), type(0)
{ 
} 

//------------------------------------------------------------ 
VectorFieldExtractor::~VectorFieldExtractor(){} 

//------------------------------------------------------------- 

void VectorFieldExtractor::get_vars(vector< string >& names,
				   vector< const TypeDescription *>& types)
{
  string command;
  DataArchive& archive = *((*(archiveH.get_rep()))());
  // Set up data to build or rebuild GUI interface
  string sNames("");
  int index = -1;
  bool matches = false;
  // get all of the VectorField Variables
  for( int i = 0; i < (int)names.size(); i++ ){
    const TypeDescription *td = types[i];
    const TypeDescription *subtype = td->getSubType();
    //cerr << "\tVariable: " << names[i] << ", type " << td->getName() << "\n";
    if( td->getType() ==  TypeDescription::NCVariable ||
	td->getType() ==  TypeDescription::CCVariable ){

      if( subtype->getType() == TypeDescription::Vector){
	if( sNames.size() != 0 )
	  sNames += " ";
	sNames += names[i];
	if( sVar.get() == "" ){ sVar.set( names[i].c_str() ); }
	if( sVar.get() == names[i].c_str()){
	  type = td;
	  matches = true;
	} else {
	  if( index == -1) {index = i;}
	}
      }	
    }
  }

  if( !matches && index != -1 ) {
    sVar.get() = names[index].c_str();
    type = types[index];
  }

  // inherited from FieldExtractor
  update_GUI(sVar.get(), sNames);
}
//------------------------------------------------------------- 

void VectorFieldExtractor::execute() 
{ 
  tcl_status.set("Calling VectorFieldExtractor!"); 
  in = (ArchiveIPort *) get_iport("Data Archive");
  vfout = (FieldOPort *) get_oport("Vector Field");
  
  ArchiveHandle handle;
  if(!in->get(handle)){
    std::cerr<<"VectorFieldExtractor::execute() Didn't get a handle\n";
    grid = 0;
    return;
  }
  
  if (archiveH.get_rep()  == 0 ){
    // first time through a frame must be built
    build_GUI_frame();
  }
  
  archiveH = handle;
  DataArchive& archive = *((*(archiveH.get_rep()))());

  // get time, set timestep, set generation, update grid and update gui
  double time = update(); // yeah it does all that
  
  // set the index for the correct timestep.
  double dt = -1;
  if (timestep < (int)times.size() - 1)
    dt = times[timestep+1] - times[timestep];
  else if (times.size() > 1)
    dt = times[timestep] - times[timestep-1];
  
  LevelP level = grid->getLevel( 0 );
  const TypeDescription* subtype = type->getSubType();
  string var(sVar.get());
  int mat = sMatNum.get();
  if(var != ""){
    switch( type->getType() ) {
    case TypeDescription::NCVariable:
      switch ( subtype->getType() ) {
      case TypeDescription::Vector:
	{	
	  NCVariable<Vector> gridVar;
	  LevelMeshHandle mesh = scinew LevelMesh( grid, 0 );
	  LevelField<Vector> *vfd =
	    scinew LevelField<Vector>( mesh, Field::NODE );
	  // set the generation and timestep in the field
	  vfd->store("varname",string(var), true);
	  vfd->store("generation",generation, true);
	  vfd->store("timestep",timestep, true);
	  vfd->store("delta_t",dt, true);
	  build_field( archive, level, var, mat, time, gridVar, vfd);
	  // send the field out to the port
	  vfout->send(vfd);
	  return;
	}
	break;
      default:
	cerr<<"NCVariable<?>  Unknown vector type\n";
	return;
      }
      break;
    case TypeDescription::CCVariable:
      switch ( subtype->getType() ) {
      case TypeDescription::Vector:
	{	
	  CCVariable<Vector> gridVar;
	  LevelMeshHandle mesh = scinew LevelMesh( grid, 0 );
	  LevelField<Vector> *vfd =
	    scinew LevelField<Vector>( mesh, Field::CELL );
	  // set the generation and timestep in the field
	  vfd->store("varname",string(var), true);
	  vfd->store("generation",generation, true);
	  vfd->store("timestep",timestep, true);
	  vfd->store("delta_t",dt, true);
	  build_field(archive, level, var, mat, time, gridVar, vfd);
	  // send the field out to the port
	  vfout->send(vfd);
	  return;
	}
	break;
      default:
	cerr<<"CCVariable<?> Unknown vector type\n";
	return;
      }
      break;
    default:
      cerr<<"Not a VectorField\n";
      return;
    }
  }
}
} // End namespace Uintah

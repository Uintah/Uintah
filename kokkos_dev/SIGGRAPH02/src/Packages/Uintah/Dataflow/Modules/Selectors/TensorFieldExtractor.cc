/****************************************
CLASS
    TensorFieldExtractor

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
#include "TensorFieldExtractor.h"

#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Util/Timer.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
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

  DECLARE_MAKER(TensorFieldExtractor)

//--------------------------------------------------------------- 
TensorFieldExtractor::TensorFieldExtractor(GuiContext* ctx) 
  : FieldExtractor("TensorFieldExtractor", ctx, "Selectors", "Uintah"),
    tcl_status(ctx->subVar("tcl_status")), sVar(ctx->subVar("sVar")),
    sMatNum(ctx->subVar("sMatNum")), type(0)
{ 
} 

//------------------------------------------------------------ 
TensorFieldExtractor::~TensorFieldExtractor(){} 

//------------------------------------------------------------- 

void TensorFieldExtractor::get_vars(vector< string >& names,
				   vector< const TypeDescription *>& types)
{
  string command;
  //DataArchive& archive = *((*(archiveH.get_rep()))());
  // Set up data to build or rebuild GUI interface
  string sNames("");
  int index = -1;
  bool matches = false;
  // get all of the TensorField Variables
  for( int i = 0; i < (int)names.size(); i++ ){
    const TypeDescription *td = types[i];
    const TypeDescription *subtype = td->getSubType();
    if( td->getType() ==  TypeDescription::NCVariable ||
	td->getType() ==  TypeDescription::CCVariable ){

      if( subtype->getType() == TypeDescription::Matrix3){
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



void TensorFieldExtractor::execute() 
{ 
  tcl_status.set("Calling TensorFieldExtractor!"); 
  in = (ArchiveIPort *) get_iport("Data Archive");
  tfout = (FieldOPort *) get_oport("Tensor Field");

  
  ArchiveHandle handle;
   if(!in->get(handle)){
     std::cerr<<"Didn't get a handle\n";
     return;
   }
   
  if (archiveH.get_rep() == 0 ){
    // first time through a frame must be built
    build_GUI_frame();
  }

   archiveH = handle;
   DataArchive& archive = *((*(archiveH.get_rep()))());
  // get time, set timestep, set generation, update grid and update gui
  double time = update(); // yeah it does all that

  LevelP level = grid->getLevel( 0 );
  IntVector hi, low, range;
  level->findIndexRange(low, hi);
  range = hi - low;
  BBox box;
  level->getSpatialRange(box);

  if( mesh_handle_.get_rep() == 0 ){
    mesh_handle_ = scinew LatVolMesh(range.x(), range.y(),
				    range.z(), box.min(),
				    box.max());
  }

  const TypeDescription* subtype = type->getSubType();
  string var(sVar.get());
  int mat = sMatNum.get();
  if(var != ""){
    switch( type->getType() ) {
    case TypeDescription::NCVariable:
      switch ( subtype->getType() ) {
      case TypeDescription::Matrix3:
	{	
	  NCVariable<Matrix3> gridVar;
		  LatVolField<Matrix3> *tfd =
	    scinew LatVolField<Matrix3>( mesh_handle_, Field::NODE );
	  // set the generation and timestep in the field
	  build_field( archive, level, low, var, mat, time, gridVar, tfd );
	  // send the field out to the port
	  tfout->send(tfd);
	  // 	DumpAllocator(default_allocator, "TensorDump.allocator");
	  return;
	}
      default:
	cerr<<"NCVariable<?>  Unknown vector type\n";
	return;
      }
    case TypeDescription::CCVariable:
      switch ( subtype->getType() ) {
      case TypeDescription::Matrix3:
	{
	  CCVariable<Matrix3> gridVar;
	  LatVolField<Matrix3> *tfd =
	    scinew LatVolField<Matrix3>( mesh_handle_, Field::CELL );
	  // set the generation and timestep in the field
	  build_field( archive, level, low, var, mat, time, gridVar, tfd );
	  // send the field out to the port
	  tfout->send(tfd);
	  // 	DumpAllocator(default_allocator, "TensorDump.allocator");
	  return;
	}
      default:
	cerr<<"CCVariable<?> Unknown vector type\n";
	return;
      }
    default:
      cerr<<"Not a TensorField\n";
      return;
    }
  }
}
} // End namespace Uintah

  

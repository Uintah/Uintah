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

#include <SCICore/Util/NotFinished.h>
#include <Uintah/Grid/TypeDescription.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Geometry/IntVector.h>
#include <SCICore/Geometry/BBox.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Datatypes/NCVectorField.h>
#include <Uintah/Datatypes/CCVectorField.h>
#include <Uintah/Interface/DataArchive.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Patch.h>
 

#include <iostream> 
#include <sstream>
#include <string>

using std::cerr;
using std::endl;
using std::vector;
using std::string;

namespace Uintah {
namespace Modules {

using SCICore::Containers::to_string;
using namespace SCICore::TclInterface;
using SCICore::Geometry::BBox;
using namespace Uintah;
using namespace Uintah::Datatypes;
using SCICore::Datatypes::VectorFieldRG;

extern "C" Module* make_VectorFieldExtractor( const clString& id ) {
  return scinew VectorFieldExtractor( id ); 
}

//--------------------------------------------------------------- 
VectorFieldExtractor::VectorFieldExtractor(const clString& id) 
  : Module("VectorFieldExtractor", id, Filter),
    tcl_status("tcl_status", id, this), sVar("sVar", id, this),
    sMatNum("sMatNum", id, this), type(0)
{ 
  //////////// Initialization code goes here
  // Create Ports
  in=scinew ArchiveIPort(this, "Data Archive",
		      ArchiveIPort::Atomic);
  sfout=scinew VectorFieldOPort(this, "VectorField", VectorFieldIPort::Atomic);

  // Add them to the Module
  add_iport(in);
  add_oport(sfout);

} 

//------------------------------------------------------------ 
VectorFieldExtractor::~VectorFieldExtractor(){} 

//------------------------------------------------------------- 

void VectorFieldExtractor::setVars()
{
  string command;

  DataArchive& archive = *((*(archiveH.get_rep()))());

  vector< string > names;
  vector< const TypeDescription *> types;
  archive.queryVariables(names, types);


  vector< double > times;
  vector< int > indices;
  archive.queryTimesteps( indices, times );

  string sNames("");
  int index = -1;
  bool matches = false;


  // get all of the VectorField Variables
  for( int i = 0; i < names.size(); i++ ){
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

  // get the number of materials for the VectorField Variables
  GridP grid = archive.queryGrid(times[0]);
  LevelP level = grid->getLevel( 0 );
  Patch* r = *(level->patchesBegin());
  int nMatls = archive.queryNumMaterials(sVar.get()(), r, times[0]);

  clString visible;
  TCL::eval(id + " isVisible", visible);
  if( visible == "1"){
    TCL::execute(id + " destroyFrames");
    TCL::execute(id + " build");
    
    TCL::execute(id + " buildMaterials " 
		 + to_string(nMatls) );

    TCL::execute(id + " setVectors " + sNames.c_str());
    TCL::execute(id + " buildVarList");

    TCL::execute("update idletasks");
    reset_vars();
  }
}



void VectorFieldExtractor::execute() 
{ 
  tcl_status.set("Calling VectorFieldExtractor!"); 
  
  ArchiveHandle handle;
   if(!in->get(handle)){
     std::cerr<<"Didn't get a handle\n";
     return;
   }
   
   if (archiveH.get_rep()  == 0 ){
     clString visible;
     TCL::eval(id + " isVisible", visible);
     if( visible == "0" ){
       TCL::execute(id + " buildTopLevel");
     }
   }

   archiveH = handle;
   DataArchive& archive = *((*(archiveH.get_rep()))());
   cerr << "Calling setVars\n";
   setVars();
   cerr << "done with setVars\n";




   // what time is it?
   vector< double > times;
   vector< int > indices;
   archive.queryTimesteps( indices, times );
   
   // set the index for the correct timestep.
   int idx = handle->timestep();


  GridP grid = archive.queryGrid(times[idx]);
  LevelP level = grid->getLevel( 0 );
  const TypeDescription* subtype = type->getSubType();
  string var(sVar.get()());
  int mat = sMatNum.get();
  double time = times[idx];
  switch( type->getType() ) {
  case TypeDescription::NCVariable:
    switch ( subtype->getType() ) {
    case TypeDescription::Vector:
      {
	NCVectorField *vfd  = scinew NCVectorField();
	
	if(var != ""){
	  vfd->SetGrid( grid );
	  vfd->SetLevel( level );
	  vfd->SetName( var );
	  vfd->SetMaterial( mat );
	  // iterate over patches
	  for(Level::const_patchIterator r = level->patchesBegin();
	      r != level->patchesEnd(); r++ ){
	    NCVariable< Vector > vv;
	    archive.query(vv, var, mat, *r, time);
	    vfd->AddVar( vv );
	  }
	  sfout->send(vfd);
	  return;
	}
      } 
      break;
    default:
      cerr<<"NCVectorField<?>  Unknown vector type\n";
      return;
    }
    break;
  case TypeDescription::CCVariable:
    switch ( subtype->getType() ) {
    case TypeDescription::Vector:
      {
	CCVectorField *vfd  = scinew CCVectorField();
	
	if(var != ""){
	  vfd->SetGrid( grid );
	  vfd->SetLevel( level );
	  vfd->SetName( var );
	      
	  vfd->SetMaterial( mat );
	  // iterate over patches
	  for(Level::const_patchIterator r = level->patchesBegin();
	      r != level->patchesEnd(); r++ ){
	    CCVariable< Vector > vv;
	    archive.query(vv, var, mat, (*r), time);
	    vfd->AddVar( vv );

	    
	  }
	  sfout->send(vfd);
	  return;
	}
      } 
      break;
    default:
      cerr<<"CCVectorField<?> Unknown vector type\n";
      return;
    }
    break;
  default:
    cerr<<"Not a VectorField\n";
    return;
  }
  return;
}
//--------------------------------------------------------------- 
} // end namespace Modules
} // end namespace Kurt
  

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
#include <Packages/Uintah/Core/Grid/TypeDescription.h>
#include <Core/Containers/String.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/BBox.h>
#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>
#include <Packages/Uintah/Core/Datatypes/NCTensorField.h>
#include <Packages/Uintah/Core/Datatypes/CCTensorField.h>
#include <Packages/Uintah/CCA/Ports/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
 

#include <iostream> 
#include <sstream>
#include <string>

namespace Uintah {

using std::cerr;
using std::endl;
using std::vector;
using std::string;

using namespace SCIRun;

extern "C" Module* make_TensorFieldExtractor( const clString& id ) {
  return scinew TensorFieldExtractor( id ); 
}

//--------------------------------------------------------------- 
TensorFieldExtractor::TensorFieldExtractor(const clString& id) 
  : Module("TensorFieldExtractor", id, Filter),
    tcl_status("tcl_status", id, this), sVar("sVar", id, this),
    sMatNum("sMatNum", id, this), type(0)
{ 
  //////////// Initialization code goes here
  // Create Ports
  in=scinew ArchiveIPort(this, "Data Archive",
		      ArchiveIPort::Atomic);
  sfout=scinew TensorFieldOPort(this, "TensorField", TensorFieldIPort::Atomic);

  // Add them to the Module
  add_iport(in);
  add_oport(sfout);

} 

//------------------------------------------------------------ 
TensorFieldExtractor::~TensorFieldExtractor(){} 

//------------------------------------------------------------- 

void TensorFieldExtractor::setVars()
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


  // get all of the TensorField Variables
  for( int i = 0; i < (int)names.size(); i++ ){
    const TypeDescription *td = types[i];
    const TypeDescription *subtype = td->getSubType();
    //cerr << "\tVariable: " << names[i] << ", type " << td->getName() << "\n";
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

  // get the number of materials for the TensorField Variables
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

    TCL::execute(id + " setTensors " + sNames.c_str());
    TCL::execute(id + " buildVarList");

    TCL::execute("update idletasks");
    reset_vars();
  }
}



void TensorFieldExtractor::execute() 
{ 
  tcl_status.set("Calling TensorFieldExtractor!"); 
  
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
    case TypeDescription::Matrix3:
      {
	NCTensorField *vfd  = scinew NCTensorField();
	
	if(var != ""){
	  vfd->SetGrid( grid );
	  vfd->SetLevel( level );
	  vfd->SetName( var );
	  vfd->SetMaterial( mat );
	  // iterate over patches
	  for(Level::const_patchIterator r = level->patchesBegin();
	      r != level->patchesEnd(); r++ ){
	    NCVariable< Matrix3 > vv;
	    archive.query(vv, var, mat, *r, time);
	    vfd->AddVar( vv );
	  }
	  sfout->send(vfd);
	  return;
	}
      } 
      break;
    default:
      cerr<<"NCTensorField<?>  Unknown vector type\n";
      return;
    }
    break;
  case TypeDescription::CCVariable:
    switch ( subtype->getType() ) {
    case TypeDescription::Matrix3:
      {
	CCTensorField *vfd  = scinew CCTensorField();
	CCVariable< Matrix3 > vv;
	
	if(var != ""){
	  vfd->SetGrid( grid );
	  vfd->SetLevel( level );
	  vfd->SetName( var );
	      
	  vfd->SetMaterial( mat );
	  // iterate over patches
	  for(Level::const_patchIterator r = level->patchesBegin();
	      r != level->patchesEnd(); r++ ){
	    CCVariable< Matrix3 > vv;
	    archive.query(vv, var, mat, (*r), time);
	    vfd->AddVar( vv );
	  }
	  sfout->send(vfd);
	  return;
	}
      } 
      break;
    default:
      cerr<<"CCTensorField<?> Unknown vector type\n";
      return;
    }
    break;
  default:
    cerr<<"Not a TensorField\n";
    return;
  }
  return;
}
} // End namespace Uintah

  

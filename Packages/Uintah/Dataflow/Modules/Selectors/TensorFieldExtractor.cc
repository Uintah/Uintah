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

#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

#include <iostream> 
#include <vector>
#include <string>



using std::cerr;
using std::vector;
using std::string;
using namespace Uintah;
using namespace SCIRun;

  DECLARE_MAKER(TensorFieldExtractor)

//--------------------------------------------------------------- 
TensorFieldExtractor::TensorFieldExtractor(GuiContext* ctx) 
  : FieldExtractor("TensorFieldExtractor", ctx, "Selectors", "Uintah")
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


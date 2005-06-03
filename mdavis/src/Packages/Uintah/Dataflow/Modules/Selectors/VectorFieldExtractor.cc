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

#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

#include <iostream> 
#include <string>
#include <vector>

using namespace Uintah;
using namespace SCIRun;
using std::cerr;
using std::vector;
using std::string;

  DECLARE_MAKER(VectorFieldExtractor)

//--------------------------------------------------------------- 
VectorFieldExtractor::VectorFieldExtractor(GuiContext* ctx) 
  : FieldExtractor("VectorFieldExtractor", ctx, "Selectors", "Uintah")
{ 
} 

//------------------------------------------------------------ 
VectorFieldExtractor::~VectorFieldExtractor(){} 

//------------------------------------------------------------- 

void VectorFieldExtractor::get_vars(vector< string >& names,
				   vector< const TypeDescription *>& types)
{
  string command;
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
    sVar.set(names[index].c_str());
    type = types[index];
  }

  if( names.size() == 0 ) type = 0;
  // inherited from FieldExtractor
  update_GUI(sVar.get(), sNames);
}

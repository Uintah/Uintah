/****************************************
CLASS
    FieldExtractor

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
#include "FieldExtractor.h"

#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/CCA/Ports/DataArchive.h>
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
#include <Core/Geometry/Transform.h>
#include <Packages/Uintah/Core/Grid/ShareAssignArray3.h>
//#include <Packages/Uintah/Core/Grid/NodeIterator.h>
 
#include <iostream> 
#include <sstream>
#include <string>

using std::cerr;
using std::endl;
using std::vector;
using std::string;
using std::ostringstream;

namespace Uintah {

using namespace SCIRun;

  //using DumbField;



//--------------------------------------------------------------- 
FieldExtractor::FieldExtractor(const string& name,
			       const string& id,
			       const string& cat,
			       const string& pack)
  : Module(name, id, Filter, cat, pack),
    generation(-1),  timestep(-1), material(-1), grid(0), archiveH(0)
{ 

} 

//------------------------------------------------------------ 
FieldExtractor::~FieldExtractor(){} 

//------------------------------------------------------------- 
void FieldExtractor::build_GUI_frame()
{
  // create the variable extractor interface.
  string visible;
  TCL::eval(id + " isVisible", visible);
  if( visible == "0" ){
    TCL::execute(id + " buildTopLevel");
  }
}

//------------------------------------------------------------- 
// get time, set timestep, set generation, update grid and update gui
double FieldExtractor::update()
{
   DataArchive& archive = *((*(archiveH.get_rep()))());
   // set the index for the correct timestep.
   int new_timestep = archiveH->timestep();
   vector< const TypeDescription *> types;
   vector< string > names;
   vector< int > indices;
   double time;
   // check to see if we have a new Archive
   archive.queryVariables(names, types);
   int new_generation = archiveH->generation;
   bool archive_dirty =  new_generation != generation;
   if (archive_dirty) {
     generation = new_generation;
     timestep = -1; // make sure old timestep is different from current
     times.clear();
     archive.queryTimesteps( indices, times );
   }
   
   if (timestep != new_timestep) {
     time = times[new_timestep];
     grid = archive.queryGrid(time);
     timestep = new_timestep;
   } else {
     time = times[timestep];
   }
   get_vars( names, types );
   return time;
}

//------------------------------------------------------------- 
void FieldExtractor::update_GUI(const string& var,
			       const string& varnames)
  // update the variable list for the GUI
{
  DataArchive& archive = *((*(archiveH.get_rep()))());
  LevelP level = grid->getLevel( 0 );
  Patch* r = *(level->patchesBegin());
  ConsecutiveRangeSet matls = 
    archive.queryMaterials(var, r, times[timestep]);

  string visible;
  TCL::eval(id + " isVisible", visible);
  if( visible == "1"){
    TCL::execute(id + " destroyFrames");
    TCL::execute(id + " build");
      
    TCL::execute(id + " buildMaterials " + matls.expandedString().c_str());
      
    TCL::execute(id + " setVars " + varnames.c_str());
    TCL::execute(id + " buildVarList");
      
    TCL::execute("update idletasks");
    reset_vars();
  }
}

} // End namespace Uintah

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

#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Util/Timer.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Geometry/Transform.h>
#include <Packages/Uintah/Core/Grid/ShareAssignArray3.h>
//#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include "FieldExtractor.h"
 
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
				 GuiContext* ctx,
				 const string& cat,
				 const string& pack)
  : Module(name, ctx, Filter, cat, pack),
    generation(-1),  timestep(-1), material(-1), levelnum(0),
    level_(ctx->subVar("level")), grid(0), 
    archiveH(0), mesh_handle_(0)
{ 

} 

//------------------------------------------------------------ 
FieldExtractor::~FieldExtractor(){} 

//------------------------------------------------------------- 
void FieldExtractor::build_GUI_frame()
{
  // create the variable extractor interface.
  string visible;
  gui->eval(id + " isVisible", visible);
  if( visible == "0" ){
    gui->execute(id + " buildTopLevel");
  }
}

//------------------------------------------------------------- 
// get time, set timestep, set generation, update grid and update gui
double FieldExtractor::field_update()
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
     mesh_handle_ = 0;
     archive.queryTimesteps( indices, times );
   }

   if (timestep != new_timestep) {
     time = times[new_timestep];
     grid = archive.queryGrid(time);
//      BBox gbox; grid->getSpatialRange(gbox);
     //     cerr<<"box: min("<<gbox.min()<<"), max("<<gbox.max()<<")\n";
     timestep = new_timestep;
   } else {
     time = times[timestep];
   }

   // Deal with changed level information
   int n = grid->numLevels();
   if (level_.get() >= (n-1)){
     level_.set(n-1);
   }
   if (levelnum != level_.get() ){
     mesh_handle_ = 0;
     levelnum = level_.get();
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
  int levels = grid->numLevels();
  LevelP level = grid->getLevel( level_.get() );

  Patch* r = *(level->patchesBegin());
  ConsecutiveRangeSet matls = 
    archive.queryMaterials(var, r, times[timestep]);

  ostringstream os;
  os << levels;

  string visible;
  gui->eval(id + " isVisible", visible);
  if( visible == "1"){
    gui->execute(id + " destroyFrames");
    gui->execute(id + " build");
    gui->execute(id + " buildLevels "+ os.str());
    gui->execute(id + " buildMaterials " + matls.expandedString().c_str());
      
    gui->execute(id + " setVars " + varnames.c_str());
    gui->execute(id + " buildVarList");
      
    gui->execute("update idletasks");
    reset_vars();
  }
}

bool 
FieldExtractor::is_periodic_bcs(IntVector cellir, IntVector ir)
{
  if( cellir.x() == ir.x() ||
      cellir.y() == ir.y() ||
      cellir.z() == ir.z() )
    return true;
  else
    return false;
}

void 
FieldExtractor::get_periodic_bcs_range(IntVector cellmax, IntVector datamax,
				       IntVector range, IntVector& newrange)
{
  if( cellmax.x() == datamax.x())
    newrange.x( range.x() + 1 );
  else
    newrange.x( range.x() );
  if( cellmax.y() == datamax.y())
    newrange.y( range.y() + 1 );
  else
    newrange.y( range.y() );
  if( cellmax.z() == datamax.z())
    newrange.z( range.z() + 1 );
  else
    newrange.z( range.z() );
}



} // End namespace Uintah

/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "VisItControlInterface_V2.h"

#include "visit_libsim.h"
#include "visit_libsim_customUI.h"

#include <sci_defs/mpi_defs.h>
#include <sci_defs/visit_defs.h>

#include <CCA/Components/SimulationController/SimulationController.h>
#include <CCA/Components/OnTheFlyAnalysis/MinMax.h>

#include <Core/Grid/Material.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>

#define ALL_LEVELS 99

static SCIRun::DebugStream visitdbg( "VisItLibSim", true );

namespace Uintah {

//---------------------------------------------------------------------
// GetTimeVars
//    Get the output checkpoints so they can be displayed in the Custon UI
//---------------------------------------------------------------------
void visit_GetTimeVars( visit_simulation_data *sim )
{
  VisItUI_setValueI("TimeStep",
		    sim->cycle, 0);
  VisItUI_setValueI("MaxTimeStep",
		    sim->simController->getSimulationTime()->maxTimestep, 1);

  VisItUI_setValueD("Time",
		    sim->time, 0);
  VisItUI_setValueD("MaxTime",
		    sim->simController->getSimulationTime()->maxTime, 1);

  VisItUI_setValueD("DeltaT",
		    sim->delt, 0);
  VisItUI_setValueD("DeltaTNext",
		    sim->delt_next, 1);
  VisItUI_setValueD("DeltaTFactor",
		    sim->simController->getSimulationTime()->delt_factor, 1);
  VisItUI_setValueD("DeltaTMin",
		    sim->simController->getSimulationTime()->delt_min, 1);
  VisItUI_setValueD("DeltaTMax",
		    sim->simController->getSimulationTime()->delt_max, 1);
  VisItUI_setValueD("ElapsedTime",
		    sim->elapsedt, 0);
  VisItUI_setValueD("MaxWallTime",
		    sim->simController->getSimulationTime()->max_wall_time, 1);
}
  
//---------------------------------------------------------------------
// GetOutputIntervals
//    Get the output checkpoints so they can be displayed in the Custon UI
//---------------------------------------------------------------------
void visit_GetOutputIntervals( visit_simulation_data *sim )
{
  SimulationStateP simStateP  = sim->simController->getSimulationStateP();
  Output          *output     = sim->simController->getOutput();

  if( output )
  {
    VisItUI_setValueS( "OutputIntervalGroupBox", "SHOW_WIDGET", 1);
    
    // Add in the output and checkout intervals.
    std::string name;
    double val;
      
    // Output interval based on time.
    if( output->getOutputInterval() > 0 )
    {
      name = simStateP->get_outputInterval_label()->getName();
      val = output->getOutputInterval();
    }
    // Output interval based on timestep.
    else
    {
      name = simStateP->get_outputTimestepInterval_label()->getName();
      val = output->getOutputTimestepInterval();
    }

    // This var must be in row specified by CheckpointIntervalRow so
    // that the callback OutputIntervalVariableTableCallback can get it.
    VisItUI_setTableValueS("OutputIntervalVariableTable",
			   OutputIntervalRow, 0, name.c_str(),  0);
    VisItUI_setTableValueD("OutputIntervalVariableTable",
			   OutputIntervalRow, 1, val, 1);
    
    // Checkpoint interval based on times.
    if( output->getCheckpointInterval() > 0 )
    {
      name = simStateP->get_checkpointInterval_label()->getName();
      val = output->getCheckpointInterval();
    }
      // Checkpoint interval based on timestep.
    else
    {
      name = simStateP->get_checkpointTimestepInterval_label()->getName();
      val = output->getCheckpointTimestepInterval();
    }

    // This var must be in row specified by CheckpointIntervalRow so
    // that the callback OutputIntervalVariableTableCallback can get it.
    VisItUI_setTableValueS("OutputIntervalVariableTable",
			   CheckpointIntervalRow, 0, name.c_str(),  0);
    VisItUI_setTableValueD("OutputIntervalVariableTable",
			   CheckpointIntervalRow, 1, val, 1);
  }
  else
    VisItUI_setValueS( "OutputIntervalGroupBox", "HIDE_WIDGET", 0);
}


//---------------------------------------------------------------------
// GetAnalysisVars
//    Get the min/max analysis vars so they can be displayed in the Custon UI
//---------------------------------------------------------------------
void visit_GetAnalysisVars( visit_simulation_data *sim )
{
  SchedulerP      schedulerP = sim->simController->getSchedulerP();
  SimulationStateP simStateP = sim->simController->getSimulationStateP();
  GridP           gridP      = sim->gridP;
  DataWarehouse*  newDW      = sim->simController->getSchedulerP()->getLastDW();

  std::vector< SimulationState::analysisVar > minMaxVars =
    simStateP->d_analysisVars;
    
  if( minMaxVars.size() )
  {
    int numLevels = gridP->numLevels();
    
    VisItUI_setValueS( "MinMaxVariableGroupBox", "SHOW_WIDGET", 1);
    
    for( unsigned int i=0; i<minMaxVars.size(); ++i )
    {
      SimulationState::analysisVar minMaxVar = minMaxVars[i];
      
      double varMin = 0, varMax = 0;
      
      // Get level info
      for (int l=0; l<numLevels; ++l)
      {
	if( minMaxVar.level != ALL_LEVELS &&
	    minMaxVar.level != l )
	  continue;
	
	LevelP levelP = gridP->getLevel(l);
	Level *level = levelP.get_rep();
	
	// double
	if( minMaxVar.label->typeDescription()->getSubType()->getType() ==
	    TypeDescription::double_type )
	{
	  min_vartype var_min;
	  max_vartype var_max;
	    
	  newDW->get(var_min, minMaxVar.reductionMinLabel, level );
	  newDW->get(var_max, minMaxVar.reductionMaxLabel, level );
	  varMin = var_min;
	  varMax = var_max;
	}

	// Vector
	else if( minMaxVar.label->typeDescription()->getSubType()->getType() ==
		 TypeDescription::Vector )
	{
	  minvec_vartype var_min;
	  maxvec_vartype var_max;
	  
	  newDW->get(var_min, minMaxVar.reductionMinLabel, level );
	  newDW->get(var_max, minMaxVar.reductionMaxLabel, level );
	  
	  varMin = ((Vector) var_min).length();
	  varMax = ((Vector) var_max).length();
	}
	
	std::stringstream name;
	
	name << "L-" << l << "/"
	     << minMaxVar.label->getName()
	     << "/" << minMaxVar.matl;
	
	VisItUI_setTableValueS("MinMaxVariableTable", i, 0, name.str().c_str(), 0);
	// VisItUI_setTableValueI("MinMaxVariableTable", i, 1, matl, 0);
	// VisItUI_setTableValueI("MinMaxVariableTable", i, 2, level, 0);
	VisItUI_setTableValueD("MinMaxVariableTable", i, 1, varMin,  0);
	VisItUI_setTableValueD("MinMaxVariableTable", i, 2, varMax,  0);
      }
    }
  }
  else
    VisItUI_setValueS( "MinMaxVariableGroupBox", "HIDE_WIDGET", 0);

}

//---------------------------------------------------------------------
// GetUPSVars
//    Get the UPS vars that the user can modify in the Custon UI
//---------------------------------------------------------------------
void visit_GetUPSVars( visit_simulation_data *sim )
{
  SimulationStateP simStateP = sim->simController->getSimulationStateP();

  if( simStateP->d_VisIt_modifiableVars.size() )
  {
    VisItUI_setValueS( "UPSVariableGroupBox", "SHOW_WIDGET", 1);

    std::vector< SimulationState::modifiableVar > &vars =
      simStateP->d_VisIt_modifiableVars;
      
    for( unsigned int i=0; i<vars.size(); ++i )
    {
      SimulationState::modifiableVar &var = vars[i];
      
      var.modified = false;
      
      VisItUI_setTableValueS("UPSVariableTable", i, 0, var.name.c_str(), 0);
      
      switch( var.type )
      {
      case Uintah::TypeDescription::int_type:
	VisItUI_setTableValueI("UPSVariableTable", i, 1, *(var.Ivalue), 1);
	break;
	    
      case Uintah::TypeDescription::double_type:
	VisItUI_setTableValueD("UPSVariableTable", i, 1, *(var.Dvalue), 1);
	break;
	    
      case Uintah::TypeDescription::Vector:
	VisItUI_setTableValueV("UPSVariableTable", i, 1,
			       var.Vvalue->x(), var.Vvalue->y(), var.Vvalue->z(), 1);
	break;
	    
      default:
	throw InternalError(" invalid data type", __FILE__, __LINE__); 
      }
    }
  }
  else
    VisItUI_setValueS( "UPSVariableGroupBox", "HIDE_WIDGET", 0);
}

} // End namespace Uintah

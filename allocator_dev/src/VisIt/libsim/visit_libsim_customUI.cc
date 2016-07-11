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

#include <sci_defs/visit_defs.h>

#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <CCA/Components/SimulationController/SimulationController.h>

#include <Core/Grid/Material.h>
#include <Core/OS/ProcessInfo.h>
#include <Core/Util/DebugStream.h>

#include <CCA/Ports/Output.h>

#define ALL_LEVELS 99
#define FINEST_LEVEL -1 

static Uintah::DebugStream visitdbg( "VisItLibSim", true );

namespace Uintah {

//---------------------------------------------------------------------
// SetTimeVars
//    Set the output checkpoints so they can be displayed in the Custon UI
//---------------------------------------------------------------------
void visit_SetTimeVars( visit_simulation_data *sim )
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

  visit_SetStripChartValue( sim, "DeltaT", sim->delt );
  visit_SetStripChartValue( sim, "DeltaTNext", sim->delt_next );
  visit_SetStripChartValue( sim, "TimeStep", (double) sim->cycle );
}
  
//---------------------------------------------------------------------
// SetOutputIntervals
//    Set the output checkpoints so they can be displayed in the Custon UI
//---------------------------------------------------------------------
void visit_SetOutputIntervals( visit_simulation_data *sim )
{
  SimulationStateP simStateP  = sim->simController->getSimulationStateP();
  Output          *output     = sim->simController->getOutput();

  VisItUI_setTableValueS("OutputIntervalVariableTable",
                         -1, -1, "CLEAR_TABLE", 0);

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
// SetAnalysisVars
//    Set the min/max analysis vars so they can be displayed in the Custon UI
//---------------------------------------------------------------------
void visit_SetAnalysisVars( visit_simulation_data *sim )
{
  GridP           gridP      = sim->gridP;
  SimulationStateP simStateP = sim->simController->getSimulationStateP();
  SchedulerP      schedulerP = sim->simController->getSchedulerP();
  DataWarehouse  *dw         = sim->simController->getSchedulerP()->getLastDW();

  std::vector< SimulationState::analysisVar > analysisVars =
    simStateP->d_analysisVars;
    
  VisItUI_setTableValueS("AnalysisVariableTable", -1, -1, "CLEAR_TABLE", 0);

  if( analysisVars.size() )
  {
    const char table[] = "AnalysisVariableTable";
    
    int numLevels = gridP->numLevels();
    
    VisItUI_setValueS( "AnalysisVariableGroupBox", "SHOW_WIDGET", 1);

    unsigned int row = 0;

    for( unsigned int i=0; i<analysisVars.size(); ++i )
    {
      SimulationState::analysisVar analysisVar = analysisVars[i];

      // Set level info
      for (int l=0; l<numLevels; ++l)
      {
	// Get the correct level.
	if( (analysisVar.level == ALL_LEVELS) ||
	    (analysisVar.level == FINEST_LEVEL &&
	     analysisVar.level == numLevels - 1) ||
	    (analysisVar.level == l) )
	{
          LevelP levelP = gridP->getLevel(l);
          Level *level = levelP.get_rep();

	  // Set the variable name, material, and level.
          VisItUI_setTableValueS(table, row, 0, analysisVar.name.c_str(), 0);
          VisItUI_setTableValueI(table, row, 1, analysisVar.matl, 0);
	  VisItUI_setTableValueI(table, row, 2, l, 0);

	  // Loop through all of the variables.
	  for( unsigned int j=0; j<analysisVar.labels.size(); ++j )
	  {
	    const VarLabel* label = analysisVar.labels[j];

	    // Work on reduction variables only (for now).
	    if( label->typeDescription()->isReductionVariable() )
	    {
	      // Get the reduction type.

	      // ARS - FIXME the material is -1 which is a flag for
	      // all materials but the variable was for a particular
	      // materiai.
	      ReductionVariableBase* var =
		dw->getReductionVariable( label, -1, level );

	      int sendcount;
	      MPI_Datatype senddatatype = MPI_DATATYPE_NULL;
	      MPI_Op sendop = MPI_OP_NULL;

	      if( var )
		var->getMPIInfo( sendcount, senddatatype, sendop );

	      // Minimum values
	      if( sendop == MPI_MIN )
	      {
		double varMin = 0;
        
		VisItUI_setTableValueS(table, row, 3+j*2, "Min", 0);
		
		// Doubles
	  	if( label->typeDescription()->getSubType()->getType() ==
	  	    TypeDescription::double_type )
	  	{
	  	  min_vartype var_min;            
	  	  dw->get(var_min, label, level );
	  	  varMin = var_min;
            
	  	  VisItUI_setTableValueD(table, row, 4+j*2, varMin, 0);
	  	}
		// Vectors
	  	else if( label->typeDescription()->getSubType()->getType() ==
                   TypeDescription::Vector )
	  	{
	  	  minvec_vartype var_min;		  
	  	  dw->get(var_min, label, level );
	  	  varMin = ((Vector) var_min).length();
		  
	  	  VisItUI_setTableValueV(table, row, 4+j*2,
	  				 ((Vector) var_min).x(),
	  				 ((Vector) var_min).y(),
	  				 ((Vector) var_min).z(), 0);    
	  	}

	  	visit_SetStripChartValue( sim, analysisVar.name+"_Min", varMin );
	      }
	      // Maximum values
	      else if( sendop == MPI_MAX )
	      {
		double varMax = 0;
        
		VisItUI_setTableValueS(table, row, 3+j*2, "Max", 0);
		
		// Doubles
	  	if( label->typeDescription()->getSubType()->getType() ==
	  	    TypeDescription::double_type )
	  	{
	  	  max_vartype var_max;
	  	  dw->get(var_max, label, level );
	  	  varMax = var_max;
            
	  	  VisItUI_setTableValueD(table, row, 4+j*2, varMax, 0);
	  	}
		// Vectors
	  	else if( label->typeDescription()->getSubType()->getType() ==
                   TypeDescription::Vector )
	  	{
	  	  maxvec_vartype var_max;
	  	  dw->get(var_max, label, level );
	  	  varMax = ((Vector) var_max).length();
		  
	  	  VisItUI_setTableValueV(table, row, 4+j*2,
	  				 ((Vector) var_max).x(),
	  				 ((Vector) var_max).y(),
	  				 ((Vector) var_max).z(), 0);    
	  	}

	  	visit_SetStripChartValue( sim, analysisVar.name+"_Max", varMax );
	      }
	      // Summ values
	      else if( sendop == MPI_SUM )
	      {
		double varSum = 0;
        
		VisItUI_setTableValueS(table, row, 3+j*2, "Sum", 0);

		// Doubles
	  	if( label->typeDescription()->getSubType()->getType() ==
	  	    TypeDescription::double_type )
	  	{
	  	  sum_vartype var_sum;
	  	  dw->get(var_sum, label, level );
	  	  varSum = var_sum;
            
	  	  VisItUI_setTableValueD(table, row, 4+j*2, varSum, 0);
	  	}
		// Vectors
	  	else if( label->typeDescription()->getSubType()->getType() ==
                   TypeDescription::Vector )
	  	{
	  	  sumvec_vartype var_sum;
	  	  dw->get(var_sum, label, level );
	  	  varSum = ((Vector) var_sum).length();
		  
	  	  VisItUI_setTableValueV(table, row, 4+j*2,
	  				 ((Vector) var_sum).x(),
	  				 ((Vector) var_sum).y(),
	  				 ((Vector) var_sum).z(), 0);    
	  	}

	  	visit_SetStripChartValue( sim, analysisVar.name+"_Sum", varSum );
	      }
	    }
	  }
        }

	++row;
      }
    }
  }
  else
    VisItUI_setValueS( "MinMaxVariableGroupBox", "HIDE_WIDGET", 0);

}


//---------------------------------------------------------------------
// SetUPSVars
//    Set the UPS vars that the user can interact with in the Custon UI
//---------------------------------------------------------------------
void visit_SetUPSVars( visit_simulation_data *sim )
{
  SimulationStateP simStateP = sim->simController->getSimulationStateP();

  if( simStateP->d_interactiveVars.size() )
  {
    VisItUI_setValueS( "UPSVariableGroupBox", "SHOW_WIDGET", 1);

    std::vector< SimulationState::interactiveVar > &vars =
      simStateP->d_interactiveVars;
      
    for( unsigned int i=0; i<vars.size(); ++i )
    {
      SimulationState::interactiveVar &var = vars[i];
      
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


//---------------------------------------------------------------------
// SetGridStats
//    Set the Grid stats
//---------------------------------------------------------------------
void visit_SetGridInfo( visit_simulation_data *sim )
{
  SimulationStateP simStateP = sim->simController->getSimulationStateP();
  GridP           gridP      = sim->gridP;

  VisItUI_setValueS( "GridInfoGroupBox", "SHOW_WIDGET", 1);
  VisItUI_setTableValueS("GridInfoTable", -1, -1, "CLEAR_TABLE", 0);

  for (int l = 0; l < gridP->numLevels(); l++) 
  {
    LevelP level = gridP->getLevel(l);
    unsigned int num_patches = level->numPatches();

    if(num_patches == 0)
      break;

    double total_cells = 0;
    double sum_of_cells_squared = 0;
      
    //calculate total cells and cells squared
    for(unsigned int p=0; p<num_patches; ++p)
    {
      const Patch* patch = level->getPatch(p);
      int num_cells = patch->getNumCells();
      
      total_cells += num_cells;
      sum_of_cells_squared += num_cells * num_cells;
    }
    
    //calculate conversion factor into simulation coordinates
    double factor = 1.0;
    
    for(int d=0; d<3; d++)
      factor *= gridP->getLevel(l)->dCell()[d];
      
    //calculate mean
    double mean = total_cells /(double) num_patches;
    // double stdv = sqrt((sum_of_cells_squared-total_cells*total_cells /
    //                     (double) num_patches) / (double) num_patches);
    IntVector refineRatio = level->getRefinementRatio();

    std::stringstream ratio;
    ratio << refineRatio.x() << ","
          << refineRatio.y() << "," << refineRatio.z();
        
    VisItUI_setTableValueI("GridInfoTable", l, 0, l+1, 0);
    VisItUI_setTableValueS("GridInfoTable", l, 1, ratio.str().c_str(), 0);
    VisItUI_setTableValueI("GridInfoTable", l, 2, num_patches, 0);
    VisItUI_setTableValueI("GridInfoTable", l, 3, total_cells, 0);
    VisItUI_setTableValueD("GridInfoTable", l, 4, mean, 0);
    VisItUI_setTableValueD("GridInfoTable", l, 5, total_cells*factor, 0);
  }
}

//---------------------------------------------------------------------
// SetRuntimeStats
//    Set the Runtime stats
//---------------------------------------------------------------------
void visit_SetRuntimeStats( visit_simulation_data *sim )
{
  SimulationStateP simStateP = sim->simController->getSimulationStateP();

  ReductionInfoMapper< SimulationState::RunTimeStat, double > &runTimeStats =
    simStateP->d_runTimeStats;

  VisItUI_setValueS( "RunTimeStatsGroupBox", "SHOW_WIDGET", 1);
  VisItUI_setTableValueS("RuntimeStatsTable", -1, -1, "CLEAR_TABLE", 0);

  int cc = 0;

  for (unsigned int i=0; i<runTimeStats.size(); ++i)
  {
    SimulationState::RunTimeStat e = (SimulationState::RunTimeStat) i;
    
    std::string name  = runTimeStats.getName(e);
    std::string units = runTimeStats.getUnits(e);

    double  average = runTimeStats.getAverage(e);
    double  maximum = runTimeStats.getMaximum(e);
    int     rank    = runTimeStats.getRank(e);

    if( average > 0 && units == std::string("MBytes"))
    {
      VisItUI_setTableValueS("RuntimeStatsTable", cc, 0, name.c_str(), 0);
      VisItUI_setTableValueS("RuntimeStatsTable", cc, 1, units.c_str(), 0);
      VisItUI_setTableValueS("RuntimeStatsTable", cc, 2,
                             ProcessInfo::toHumanUnits(average).c_str(), 0);
      VisItUI_setTableValueS("RuntimeStatsTable", cc, 3,
                             ProcessInfo::toHumanUnits(maximum).c_str(), 0);
      VisItUI_setTableValueI("RuntimeStatsTable", cc, 4, rank, 0);
      VisItUI_setTableValueD("RuntimeStatsTable", cc, 5,
                             100*(1-(average/maximum)), 0);
    
      ++cc;
    }
  
    else if( average > 0 )
    {
      VisItUI_setTableValueS("RuntimeStatsTable", cc, 0, name.c_str(), 0);
      VisItUI_setTableValueS("RuntimeStatsTable", cc, 1, units.c_str(), 0);
      VisItUI_setTableValueD("RuntimeStatsTable", cc, 2, average, 0);
      VisItUI_setTableValueD("RuntimeStatsTable", cc, 3, maximum, 0);
      VisItUI_setTableValueI("RuntimeStatsTable", cc, 4, rank, 0);
      VisItUI_setTableValueD("RuntimeStatsTable", cc, 5,
                             100*(1-(average/maximum)), 0);

      ++cc;
    }

    visit_SetStripChartValue( sim, name, average );
    visit_SetStripChartValue( sim, name+"_Ave", average );
    visit_SetStripChartValue( sim, name+"_Max", maximum );
  }
}

//---------------------------------------------------------------------
// SetMPIStats
//    Set the MPI stats
//---------------------------------------------------------------------
void visit_SetMPIStats( visit_simulation_data *sim )
{
  SimulationStateP simStateP = sim->simController->getSimulationStateP();

  MPIScheduler *mpiScheduler = dynamic_cast<MPIScheduler*>
    (sim->simController->getSchedulerP().get_rep());
  
  // Add in the mpi run time stats.
  if( mpiScheduler )
  {
    ReductionInfoMapper< MPIScheduler::TimingStat, double > &mpiStats =
      mpiScheduler->mpi_info_;

    VisItUI_setValueS( "MPIStatsGroupBox", "SHOW_WIDGET", 1);
    VisItUI_setTableValueS("MPIStatsTable", -1, -1, "CLEAR_TABLE", 0);

    for (unsigned int i=0; i<mpiStats.size(); ++i)
    {
      MPIScheduler::TimingStat e = (MPIScheduler::TimingStat) i;
      
      std::string name  = mpiStats.getName(e);
      std::string units = mpiStats.getUnits(e);
      
      double  average = mpiStats.getAverage(e);
      double  maximum = mpiStats.getMaximum(e);
      int     rank    = mpiStats.getRank(e);
      
      VisItUI_setTableValueS("MPIStatsTable", i, 0, name.c_str(), 0);
      VisItUI_setTableValueS("MPIStatsTable", i, 1, units.c_str(), 0);
      VisItUI_setTableValueD("MPIStatsTable", i, 2, average, 0);
      VisItUI_setTableValueD("MPIStatsTable", i, 3, maximum, 0);
      VisItUI_setTableValueI("MPIStatsTable", i, 4, rank, 0);
      VisItUI_setTableValueD("MPIStatsTable", i, 5,
                             100*(1-(average/maximum)), 0);
      visit_SetStripChartValue( sim, name, average );
      visit_SetStripChartValue( sim, name+"_Ave", average );
      visit_SetStripChartValue( sim, name+"_Max", maximum );
    }
  }
  else
  {
    VisItUI_setValueS( "MPIStatsGroupBox", "HIDE_WIDGET", 0);
    VisItUI_setTableValueS("MPIStatsTable", -1, -1, "CLEAR_TABLE", 0);
  }
}

//---------------------------------------------------------------------
// SetImageVars
//    
//---------------------------------------------------------------------
void visit_SetImageVars( visit_simulation_data *sim )
{
  VisItUI_setValueI("ImageGroupBox", sim->imageGenerate, 1);
  VisItUI_setValueS("ImageFilename", sim->imageFilename.c_str(), 1);
  VisItUI_setValueI("ImageHeight",   sim->imageHeight, 1);
  VisItUI_setValueI("ImageWidth",    sim->imageWidth,  1);
  VisItUI_setValueI("ImageFormat",   sim->imageFormat, 1);
}

//---------------------------------------------------------------------
// SetStripChartValue
//    
//---------------------------------------------------------------------
void visit_SetStripChartNames( visit_simulation_data *sim )
{
  for( unsigned int chart=0; chart<5; ++chart ) // Five Charts
  {
    for( unsigned int i=0; i<5; ++i ) // Chart name plus four curves
    {
//      if( sim->stripChartNames[chart][i].size() )
      {
        char cmd[128];
      
        sprintf( cmd, "%d | %d | %s",
                 chart, i, sim->stripChartNames[chart][i].c_str() );
        
        VisItUI_setValueS("STRIP_CHART_SET_NAME", cmd, 1);

        // Restore the table values.
        VisItUI_setTableValueS("StripChartTable", i, chart,
                               sim->stripChartNames[chart][i].c_str(), 1);
      }
    }
  }
}

//---------------------------------------------------------------------
// SetStripChartValue
//    
//---------------------------------------------------------------------
void visit_SetStripChartValue( visit_simulation_data *sim,
                               std::string name,
                               double value )
{
  for( unsigned int chart=0; chart<5; ++chart ) // Five Charts
  {
    for( unsigned int i=1, curve=0; i<5; ++i, ++curve) // Four curves
    {
      if( name == sim->stripChartNames[chart][i] )
      {
        char cmd[128];
      
        sprintf( cmd, "%d | %d | %lf | %lf",
                 chart, curve, (double) sim->cycle, value );
        
        VisItUI_setValueS("STRIP_CHART_ADD_POINT", cmd, 1);
      }
    }
  }
}
  
} // End namespace Uintah

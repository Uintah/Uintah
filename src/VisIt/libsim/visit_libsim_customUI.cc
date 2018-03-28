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

#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/Output.h>

#include <CCA/Components/Schedulers/MPIScheduler.h>
#include <CCA/Components/SimulationController/SimulationController.h>

#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationTime.h>
#include <Core/OS/ProcessInfo.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/DOUT.hpp>


#include <sci_defs/visit_defs.h>

#define ALL_LEVELS 99
#define FINEST_LEVEL -1
#define IGNORE_LEVEL -2

namespace Uintah {

//---------------------------------------------------------------------
// SetTimeVars
//    Set the times values so they can be displayed in the Custon UI
//---------------------------------------------------------------------
void visit_SetTimeValues( visit_simulation_data *sim )
{
  SimulationTime* simTime =
    sim->simController->getApplicationInterface()->getSimulationTime();

  VisItUI_setValueI("TimeStep",      sim->cycle, 0);
  VisItUI_setValueI("MaxTimeStep",   simTime->m_max_time_steps, 1);

  VisItUI_setValueD("Time",          sim->time, 0);
  VisItUI_setValueD("MaxTime",       simTime->m_max_time, 1);

  VisItUI_setValueI("EndAtMaxTime",      simTime->m_end_at_max_time, 1);
  VisItUI_setValueI("ClampTimeToOutput", simTime->m_clamp_time_to_output, 1);

  VisItUI_setValueI("StopAtTimeStep",     sim->stopAtTimeStep,     1);
  VisItUI_setValueI("StopAtLastTimeStep", sim->stopAtLastTimeStep, 1);

  // visit_SetStripChartValue( sim, "TimeStep", (double) sim->cycle );
}

//---------------------------------------------------------------------
// SetWallTimes
//    Set the wall times so they can be displayed in the Custon UI
//---------------------------------------------------------------------
void visit_SetDeltaTValues( visit_simulation_data *sim )
{
  SimulationTime* simTime =
    sim->simController->getApplicationInterface()->getSimulationTime();

  int row = 0;

  VisItUI_setTableValueS("DeltaTVariableTable", -1, -1, "CLEAR_TABLE", 0);

  VisItUI_setTableValueS("DeltaTVariableTable", row, 0, "DeltaT", 0);
  VisItUI_setTableValueD("DeltaTVariableTable", row, 1, sim->delt, 0);
  ++row;
  
  VisItUI_setTableValueS("DeltaTVariableTable", row, 0, "DeltaTNext", 0);
  VisItUI_setTableValueD("DeltaTVariableTable", row, 1, sim->delt_next, 1);
  ++row;

  VisItUI_setTableValueS("DeltaTVariableTable", row, 0, "DeltaTFactor", 0);
  VisItUI_setTableValueD("DeltaTVariableTable", row, 1, simTime->m_delt_factor, 1);
  ++row;

  VisItUI_setTableValueS("DeltaTVariableTable", row, 0, "MaxDeltaTIncrease", 0);
  VisItUI_setTableValueD("DeltaTVariableTable", row, 1, simTime->m_max_delt_increase, 1);
  ++row;

  VisItUI_setTableValueS("DeltaTVariableTable", row, 0, "DeltaTMin", 0);
  VisItUI_setTableValueD("DeltaTVariableTable", row, 1, simTime->m_delt_min, 1);
  ++row;

  VisItUI_setTableValueS("DeltaTVariableTable", row, 0, "DeltaTMax", 0);
  VisItUI_setTableValueD("DeltaTVariableTable", row, 1, simTime->m_delt_max, 1);
  ++row;

  VisItUI_setTableValueS("DeltaTVariableTable", row, 0, "MaxInitialDeltaT", 0);
  VisItUI_setTableValueD("DeltaTVariableTable", row, 1, simTime->m_max_initial_delt, 1);
  ++row;

  VisItUI_setTableValueS("DeltaTVariableTable", row, 0, "InitialDeltaTRange", 0);
  VisItUI_setTableValueD("DeltaTVariableTable", row, 1, simTime->m_initial_delt_range, 1);
  ++row;

  VisItUI_setTableValueS("DeltaTVariableTable", row, 0, "OverrideRestartDeltaT", 0);
  VisItUI_setTableValueD("DeltaTVariableTable", row, 1, simTime->m_override_restart_delt, 1);
  ++row;

  visit_SetStripChartValue( sim, "DeltaT/Current", sim->delt );
  visit_SetStripChartValue( sim, "DeltaT/Next", sim->delt_next );
}

//---------------------------------------------------------------------
// SetWallTimes
//    Set the wall times so they can be displayed in the Custon UI
//---------------------------------------------------------------------
void visit_SetWallTimes( visit_simulation_data *sim )
{
  WallTimers* walltimers  = sim->simController->getWallTimers();
  SimulationTime* simTime =
    sim->simController->getApplicationInterface()->getSimulationTime();

  double time = walltimers->GetWallTime();

  int row = 0;

  VisItUI_setTableValueS("WallTimesVariableTable", -1, -1, "CLEAR_TABLE", 0);

  VisItUI_setTableValueS("WallTimesVariableTable", row, 0, "ExpMovingAve",  0);
  VisItUI_setTableValueD("WallTimesVariableTable", row, 1,
                         walltimers->ExpMovingAverage().seconds(), 0);
  ++row;
  VisItUI_setTableValueS("WallTimesVariableTable", row, 0, "TimeStep",  0);
  VisItUI_setTableValueD("WallTimesVariableTable", row, 1,
                         walltimers->TimeStep().seconds(), 0);
  ++row;
  VisItUI_setTableValueS("WallTimesVariableTable", row, 0, "InSitu",  0);
  VisItUI_setTableValueD("WallTimesVariableTable", row, 1,
                         walltimers->InSitu().seconds(), 0);
  ++row;
  VisItUI_setTableValueS("WallTimesVariableTable", row, 0, "Total",  0);
  VisItUI_setTableValueD("WallTimesVariableTable", row, 1,
                         time, 0);
  ++row;
  VisItUI_setTableValueS("WallTimesVariableTable", row, 0, "Maximum",  0);
  VisItUI_setTableValueD("WallTimesVariableTable", row, 1,
                         simTime->m_max_wall_time, 1);
  ++row;

  visit_SetStripChartValue( sim, "WallTimes/TimeStep",     walltimers->TimeStep().seconds() );
  visit_SetStripChartValue( sim, "WallTimes/ExpMovingAve", walltimers->ExpMovingAverage().seconds() );
  visit_SetStripChartValue( sim, "WallTimes/InSitu",       walltimers->InSitu().seconds() );
  visit_SetStripChartValue( sim, "WallTimes/Total",        time );
}


//---------------------------------------------------------------------
// SetOutputIntervals
//    Set the output checkpoints so they can be displayed in the Custon UI
//---------------------------------------------------------------------
void visit_SetOutputIntervals( visit_simulation_data *sim )
{
  // ApplicationInterface* appInterface =
  //   sim->simController->getApplicationInterface();

  // SimulationTime* simTime = appInterface->getSimulationTime();

  Output          *output       = sim->simController->getOutput();

  VisItUI_setTableValueS("OutputIntervalVariableTable",
                         -1, -1, "CLEAR_TABLE", 0);

  if( output )
  {
    VisItUI_setValueS( "OutputIntervalGroupBox", "SHOW_WIDGET", 1);
    
    // Add in the output and checkout intervals.
    std::string name;
    double val;

    std::string nextName;
    double nextVal;

    // Output interval based on time.
    if( output->getOutputInterval() > 0 )
    {
      name = "OutputInterval";
      val = output->getOutputInterval();
      nextName = "NextOutput";
      nextVal = output->getNextOutputTime();
    } 
    // Output interval based on time step.
    else // if( output->getOutputTimeStepInterval() > 0 )
    {
      name = "OutputTimeStepInterval";
      val = output->getOutputTimeStepInterval();
      nextName = "OutputTimeStep";
      nextVal = output->getNextOutputTimeStep();
    }

    // This var must be in the row specified by OutputIntervalRow so
    // that the callback OutputIntervalVariableTableCallback can get it.
    VisItUI_setTableValueS("OutputIntervalVariableTable",
                           OutputIntervalRow, 0, name.c_str(),  0);
    VisItUI_setTableValueD("OutputIntervalVariableTable",
                           OutputIntervalRow, 1, val, 1);
    
    // This var must be in the row specified by OutputIntervalRow
    VisItUI_setTableValueS("OutputIntervalVariableTable",
                           OutputIntervalRow+1, 0, nextName.c_str(),  0);
    VisItUI_setTableValueD("OutputIntervalVariableTable",
                           OutputIntervalRow+1, 1, nextVal, 0);
    
    // This var must be in the row specified by OutputIntervalRow
    VisItUI_setTableValueS("OutputIntervalVariableTable",
                           OutputIntervalRow+2, 0, "isOutputTimeStep",  0);
    VisItUI_setTableValueS("OutputIntervalVariableTable",
                           OutputIntervalRow+2, 1,
                           (output->isOutputTimeStep() ? "Yes" : "No"), 0);
    
    // Checkpoint interval based on sim time.
    if( output->getCheckpointInterval() > 0 )
    {
      name = "CheckpointInterval";
      nextName = "NextCheckpoint";
      val = output->getCheckpointInterval();
      nextVal = output->getNextCheckpointTime();
    }
    // Checkpoint interval based on the wall time.
    else if( output->getCheckpointWallTimeInterval() > 0 )
    {
      name = "CheckpointWallTimeInterval";
      val = output->getCheckpointWallTimeInterval();
      nextName = "NextCheckpointWallTime";
      nextVal = output->getNextCheckpointWallTime();
    }
    // Checkpoint interval based on the time step.
    else // if( output->getCheckpointTimeStepInterval() > 0 )
    {
      name = "CheckpointTimeStepInterval";
      val = output->getCheckpointTimeStepInterval();
      nextName = "NextCheckpointTimeStep";
      nextVal = output->getNextCheckpointTimeStep();
    }
    
    // This var must be in the row specified by CheckpointIntervalRow so
    // that the callback OutputIntervalVariableTableCallback can get it.
    VisItUI_setTableValueS("OutputIntervalVariableTable",
                           CheckpointIntervalRow, 0, name.c_str(),  0);
    VisItUI_setTableValueD("OutputIntervalVariableTable",
                           CheckpointIntervalRow, 1, val, 1);

    // This var must be in the row specified by CheckpointIntervalRow
    VisItUI_setTableValueS("OutputIntervalVariableTable",
                           CheckpointIntervalRow+1, 0, nextName.c_str(), 0);
    VisItUI_setTableValueD("OutputIntervalVariableTable",
                           CheckpointIntervalRow+1, 1, nextVal, 0);

    // This var must be in the row specified by CheckpointIntervalRow
    VisItUI_setTableValueS("OutputIntervalVariableTable",
                           CheckpointIntervalRow+2, 0, "isCheckpointTimeStep", 0);
    VisItUI_setTableValueS("OutputIntervalVariableTable",
                           CheckpointIntervalRow+2, 1,
                           (output->isCheckpointTimeStep() ? "Yes" : "No"), 0);


    // This var must be in row specified by CheckpointCyclelRow
    VisItUI_setTableValueS("OutputIntervalVariableTable",
                           CheckpointCycleRow, 0, "CheckpointCycle",  0);
    VisItUI_setTableValueI("OutputIntervalVariableTable",
                           CheckpointCycleRow, 1, output->getCheckpointCycle(), 1);
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
  // No data for the inital setup or restart.
  if( sim->first )
    return;
  
  const char table[] = "AnalysisVariableTable";
    
  GridP           gridP      = sim->gridP;
  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();
  SchedulerP      schedulerP = sim->simController->getSchedulerP();
  DataWarehouse  *dw         = sim->simController->getSchedulerP()->getLastDW();

  std::vector< ApplicationInterface::analysisVar > analysisVars =
    appInterface->getAnalysisVars();
    
  VisItUI_setTableValueS("AnalysisVariableTable", -1, -1, "CLEAR_TABLE", 0);

  if( analysisVars.size() )
  {
    unsigned int numLevels = gridP->numLevels();
    
    VisItUI_setValueS( "AnalysisVariableGroupBox", "SHOW_WIDGET", 1);

    unsigned int row = 0;

    for( unsigned int i=0; i<analysisVars.size(); ++i )
    {
      ApplicationInterface::analysisVar analysisVar = analysisVars[i];

      // Set level info
      for (unsigned int l=0; l<numLevels; ++l)
      {
        // Get the correct level.
        if( (analysisVar.level == IGNORE_LEVEL && l == 0) ||
            (analysisVar.level == ALL_LEVELS) ||
            (analysisVar.level == FINEST_LEVEL && l == numLevels - 1) ||
            (analysisVar.level == (int) l) )
        {
          LevelP levelP = gridP->getLevel(l);
          Level *level = levelP.get_rep();

          std::stringstream stripChartName;
          stripChartName << "Analysis/"
                         << analysisVar.component << "/"  << analysisVar.name;

          // Set the variable name, material, and level.
          VisItUI_setTableValueS(table, row, 0, analysisVar.component.c_str(), 0);
          VisItUI_setTableValueS(table, row, 1, analysisVar.name.c_str(), 0);

          // Add the material to the analysis var
          if( analysisVar.matl < 0 )
          {
            VisItUI_setTableValueS(table, row, 2, "NA", 0);
          }
          else
          {
            VisItUI_setTableValueI(table, row, 2, analysisVar.matl, 0);
            stripChartName << "/" << analysisVar.matl;
          }

          // Add the level to the analysis var
          if( analysisVar.level < 0)
          {
            VisItUI_setTableValueS(table, row, 3, "NA", 0);
          }
          else
          {
            VisItUI_setTableValueI(table, row, 3, l, 0);
            stripChartName << "/l" << l;
          }
          
          // Loop through all of the variables.
          for( unsigned int j=0; j<analysisVar.labels.size(); ++j )
          {
            const VarLabel* label = analysisVar.labels[j];

            // Work on reduction variables only (for now) and make
            // sure they exist.
            if( label->typeDescription()->isReductionVariable() &&
                ( (analysisVar.level == IGNORE_LEVEL && dw->exists( label )) ||
                  (dw->exists( label, analysisVar.matl, level )) ) )
            {
              // Get the reduction type.
              if( label->typeDescription() == min_vartype::getTypeDescription() )
              {
                VisItUI_setTableValueS(table, row, 4+j*2, "Min", 0);

                min_vartype var_min;

                if( analysisVar.level == IGNORE_LEVEL )
                  dw->get( var_min, label );
                else
                  dw->get( var_min, label, level, analysisVar.matl );
                
                double varMin = var_min;

                VisItUI_setTableValueD(table, row, 5+j*2, varMin, 0);

                visit_SetStripChartValue( sim, stripChartName.str() +
                                          "/Minimum", varMin );
              }
              else if( label->typeDescription() == max_vartype::getTypeDescription() )
              {
                VisItUI_setTableValueS(table, row, 4+j*2, "Max", 0);

                max_vartype var_max;
                if( analysisVar.level == IGNORE_LEVEL )
                  dw->get( var_max, label );
                else
                  dw->get( var_max, label, level, analysisVar.matl );

                double varMax = var_max;

                VisItUI_setTableValueD(table, row, 5+j*2, varMax, 0);

                visit_SetStripChartValue( sim, stripChartName.str() +
                                          "/Maximum", varMax );
              }
              
              else if( label->typeDescription() == minvec_vartype::getTypeDescription() )
              {
                VisItUI_setTableValueS(table, row, 4+j*2, "Min", 0);

                minvec_vartype var_min;

                if( analysisVar.level == IGNORE_LEVEL )
                  dw->get( var_min, label );
                else
                  dw->get( var_min, label, level, analysisVar.matl );

                double varMin = ((Vector) var_min).length();

                VisItUI_setTableValueV(table, row, 5+j*2,
                                       ((Vector) var_min).x(),
                                       ((Vector) var_min).y(),
                                       ((Vector) var_min).z(), 0);    

                visit_SetStripChartValue( sim, stripChartName.str() +
                                          "/Minimum", varMin );
              }
              else if( label->typeDescription() == maxvec_vartype::getTypeDescription() )
              {
                VisItUI_setTableValueS(table, row, 4+j*2, "Max", 0);

                maxvec_vartype var_max;
                if( analysisVar.level < 0 )
                  dw->get( var_max, label );
                else
                  dw->get( var_max, label, level, analysisVar.matl );

                double varMax = ((Vector) var_max).length();

                VisItUI_setTableValueV(table, row, 5+j*2,
                                       ((Vector) var_max).x(),
                                       ((Vector) var_max).y(),
                                       ((Vector) var_max).z(), 0);

                visit_SetStripChartValue( sim, stripChartName.str() +
                                          "/Maximum",
                                          varMax );
              }
              else if( label->typeDescription() == sum_vartype::getTypeDescription() )
              {
                VisItUI_setTableValueS(table, row, 4+j*2, "Sum", 0);

                sum_vartype var_sum;
                if( analysisVar.level == IGNORE_LEVEL )
                  dw->get( var_sum, label );
                else
                  dw->get( var_sum, label, level, analysisVar.matl );

                double varSum = var_sum;

                VisItUI_setTableValueD(table, row, 5+j*2, varSum, 0);

                visit_SetStripChartValue( sim, stripChartName.str() +
                                          "/Sum",
                                          varSum );
              }
              else if( label->typeDescription() == sumvec_vartype::getTypeDescription() )
              {
                VisItUI_setTableValueS(table, row, 4+j*2, "Sum", 0);

                sumvec_vartype var_sum;
                if( analysisVar.level == IGNORE_LEVEL )
                  dw->get( var_sum, label );
                else
                  dw->get( var_sum, label, level, analysisVar.matl );

                double varSum = ((Vector) var_sum).length();

                VisItUI_setTableValueV(table, row, 5+j*2,
                                       ((Vector) var_sum).x(),
                                       ((Vector) var_sum).y(),
                                       ((Vector) var_sum).z(), 0);

                visit_SetStripChartValue( sim, stripChartName.str() +
                                          "/Sum",
                                          varSum );
              }
              else
              {
                std::stringstream msg;
                msg << stripChartName.str() << "  "
                    << label->getName() << "  "
                    << label->typeDescription()->getName() << "  "
                    << "unknown_vartype";

                VisItUI_setValueS("SIMULATION_MESSAGE_WARNING", msg.str().c_str(), 1);
                VisItUI_setTableValueS(table, row, 4+j*2, "Unknown", 0);
                VisItUI_setTableValueS(table, row, 5+j*2, "NA", 0);
              }
            }
          }

          ++row;
        }
      }
    }
  }
  else
    VisItUI_setValueS( "AnalysisVariableGroupBox", "HIDE_WIDGET", 0);
}


//---------------------------------------------------------------------
// SetUPSVars
//    Set the UPS vars that the user can interact with in the Custon UI
//---------------------------------------------------------------------
void visit_SetUPSVars( visit_simulation_data *sim )
{
  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();

  if( appInterface->getUPSVars().size() )
  {
    VisItUI_setValueS( "UPSVariableGroupBox", "SHOW_WIDGET", 1);

    std::vector< ApplicationInterface::interactiveVar > &vars =
      appInterface->getUPSVars();
      
    for( unsigned int i=0; i<vars.size(); ++i )
    {
      ApplicationInterface:: interactiveVar &var = vars[i];
      
      var.modified = false;
      
      VisItUI_setTableValueS("UPSVariableTable", i, 0, var.component.c_str(), 0);
      VisItUI_setTableValueS("UPSVariableTable", i, 1, var.name.c_str(), 0);
      
      switch( var.type )
      {
        case Uintah::TypeDescription::bool_type:
        {
          bool *val = (bool*) var.value;
          VisItUI_setTableValueI("UPSVariableTable", i, 2, *val, 1);
        }
        break;
            
        case Uintah::TypeDescription::int_type:
        {
          int *val = (int*) var.value;
          VisItUI_setTableValueI("UPSVariableTable", i, 2, *val, 1);
        }
        break;
            
        case Uintah::TypeDescription::double_type:
        {
          double *val = (double*) var.value;
          VisItUI_setTableValueD("UPSVariableTable", i, 2, *val, 1);
        }
        break;
            
        case Uintah::TypeDescription::Point:
        {
          Point *val = (Point*) var.value;
          VisItUI_setTableValueV("UPSVariableTable", i, 2,
                                 val->x(), val->y(), val->z(), 1);
        }
        break;
            
        case Uintah::TypeDescription::Vector:
        {
          Vector *val = (Vector*) var.value;
          VisItUI_setTableValueV("UPSVariableTable", i, 2,
                                 val->x(), val->y(), val->z(), 1);
        }
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
  // ApplicationInterface* appInterface =
  //   sim->simController->getApplicationInterface();

  GridP                gridP        = sim->gridP;

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
  ReductionInfoMapper< RuntimeStatsEnum, double > runtimeStats =
    sim->simController->getRuntimeStats();

  VisItUI_setValueS( "RuntimeStatsGroupBox", "SHOW_WIDGET", 1);
  VisItUI_setTableValueS("RuntimeStatsTable", -1, -1, "CLEAR_TABLE", 0);

  int cc = 0;

  for (unsigned int i=0; i<runtimeStats.size(); ++i)
  {
    std::string name  = runtimeStats.getName(i);
    std::string units = runtimeStats.getUnits(i);

    double  average = runtimeStats.getAverage(i);
    double  maximum = runtimeStats.getMaximum(i);
    int     rank    = runtimeStats.getRank(i);

    if( average > 0 && units == std::string("MBytes"))
    {
      VisItUI_setTableValueS("RuntimeStatsTable", cc, 0, name.c_str(), 0);
      VisItUI_setTableValueS("RuntimeStatsTable", cc, 1, units.c_str(), 0);
      VisItUI_setTableValueS("RuntimeStatsTable", cc, 2,
                             ProcessInfo::toHumanUnits(average).c_str(), 0);
      VisItUI_setTableValueS("RuntimeStatsTable", cc, 3,
                             ProcessInfo::toHumanUnits(maximum).c_str(), 0);
      VisItUI_setTableValueI("RuntimeStatsTable", cc, 4, rank, 0);
      if( maximum != 0 )
        VisItUI_setTableValueD("RuntimeStatsTable", cc, 5,
                               100.0*(1.0-(average/maximum)), 0);
      else
        VisItUI_setTableValueD("RuntimeStatsTable", cc, 5, 0.0, 0);
      
      ++cc;
    }
  
    else if( average > 0 )
    {
      VisItUI_setTableValueS("RuntimeStatsTable", cc, 0, name.c_str(), 0);
      VisItUI_setTableValueS("RuntimeStatsTable", cc, 1, units.c_str(), 0);
      VisItUI_setTableValueD("RuntimeStatsTable", cc, 2, average, 0);
      VisItUI_setTableValueD("RuntimeStatsTable", cc, 3, maximum, 0);
      VisItUI_setTableValueI("RuntimeStatsTable", cc, 4, rank, 0);
      if( maximum != 0 )
        VisItUI_setTableValueD("RuntimeStatsTable", cc, 5,
                               100.0*(1.0-(average/maximum)), 0);
      else
        VisItUI_setTableValueD("RuntimeStatsTable", cc, 5, 0.0, 0);

      ++cc;
    }

    visit_SetStripChartValue( sim, "RuntimeStats/"+name+"/Average", average );
    visit_SetStripChartValue( sim, "RuntimeStats/"+name+"/Maximum", maximum );
  }
}

//---------------------------------------------------------------------
// SetMPIStats
//    Set the MPI stats
//---------------------------------------------------------------------
void visit_SetMPIStats( visit_simulation_data *sim )
{
  MPIScheduler *mpiScheduler = dynamic_cast<MPIScheduler*>
    (sim->simController->getSchedulerP().get_rep());
  
  // Add in the mpi run time stats.
  if( mpiScheduler && mpiScheduler->mpi_info_.size() )
  {
    ReductionInfoMapper< MPIScheduler::TimingStatEnum, double > &mpiStats =
      mpiScheduler->mpi_info_;

    VisItUI_setValueS( "MPIStatsGroupBox", "SHOW_WIDGET", 1);
    VisItUI_setTableValueS("MPIStatsTable", -1, -1, "CLEAR_TABLE", 0);

    for (unsigned int i=0; i<mpiStats.size(); ++i)
    {
      std::string name  = mpiStats.getName(i);
      std::string units = mpiStats.getUnits(i);
      
      double  average = mpiStats.getAverage(i);
      double  maximum = mpiStats.getMaximum(i);
      int     rank    = mpiStats.getRank(i);
      
      VisItUI_setTableValueS("MPIStatsTable", i, 0, name.c_str(), 0);
      VisItUI_setTableValueS("MPIStatsTable", i, 1, units.c_str(), 0);
      VisItUI_setTableValueD("MPIStatsTable", i, 2, average, 0);
      VisItUI_setTableValueD("MPIStatsTable", i, 3, maximum, 0);
      VisItUI_setTableValueI("MPIStatsTable", i, 4, rank, 0);
      if( maximum != 0 )
        VisItUI_setTableValueD("MPIStatsTable", i, 5,
                               100.0*(1.0-(average/maximum)), 0);
      else
        VisItUI_setTableValueD("MPIStatsTable", i, 5, 0.0, 0);

      visit_SetStripChartValue( sim, "MPIStats/"+name+"/Average", average );
      visit_SetStripChartValue( sim, "MPIStats/"+name+"/Maximum", maximum );
    }
  }
  else
  {
    VisItUI_setValueS( "MPIStatsGroupBox", "HIDE_WIDGET", 0);
    VisItUI_setTableValueS("MPIStatsTable", -1, -1, "CLEAR_TABLE", 0);
  }
}

//---------------------------------------------------------------------
// SetApplicationStats
//    Set the application stats
//---------------------------------------------------------------------
void visit_SetApplicationStats( visit_simulation_data *sim )
{
  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();

  unsigned int nStats = appInterface->getApplicationStats().size();

  if( nStats )
  {
    VisItUI_setValueS( "ApplicationStatsGroupBox", "SHOW_WIDGET", 1);
    VisItUI_setTableValueS("ApplicationStatsTable", -1, -1, "CLEAR_TABLE", 0);

    for (unsigned int i=0; i<nStats; ++i)
    {
      std::string name  = appInterface->getApplicationStats().getName(i);
      std::string units = appInterface->getApplicationStats().getUnits(i);
      
      double  average = appInterface->getApplicationStats().getAverage(i);
      double  maximum = appInterface->getApplicationStats().getMaximum(i);
      int     rank    = appInterface->getApplicationStats().getRank(i);
      
      VisItUI_setTableValueS("ApplicationStatsTable", i, 0, name.c_str(), 0);
      VisItUI_setTableValueS("ApplicationStatsTable", i, 1, units.c_str(), 0);
      VisItUI_setTableValueD("ApplicationStatsTable", i, 2, average, 0);
      VisItUI_setTableValueD("ApplicationStatsTable", i, 3, maximum, 0);
      VisItUI_setTableValueI("ApplicationStatsTable", i, 4, rank, 0);
      // if( maximum != 0 )
      //         VisItUI_setTableValueD("ApplicationStatsTable", i, 5,
      //                                100.0*(1.0-(average/maximum)), 0);
      // else
      //         VisItUI_setTableValueD("ApplicationStatsTable", i, 5, 0.0, 0);
      
      visit_SetStripChartValue( sim, "ApplicationStats/"+name+"/Average", average );
      visit_SetStripChartValue( sim, "ApplicationStats/"+name+"/Maximum", maximum );
    }
  }
  else
  {
    VisItUI_setValueS( "ApplicationStatsGroupBox", "HIDE_WIDGET", 0);
    VisItUI_setTableValueS("ApplicationStatsTable", -1, -1, "CLEAR_TABLE", 0);
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
void visit_SetStripChartValue( visit_simulation_data *sim,
                               std::string name,
                               double value )
{
  VisItUI_setValueS("STRIP_CHART_ADD_MENU_ITEM", name.c_str(), 1);
  
  for( unsigned int chart=0; chart<5; ++chart ) // Five Charts
  {
    for( unsigned int curve=0; curve<5; ++curve) // Five curves
    {
      if( name == sim->stripChartNames[chart][curve] )
      {
        char cmd[128];
      
        sprintf( cmd, "%d | %d | %lf | %lf",
                 chart, curve, (double) sim->cycle, value );
        
        VisItUI_setValueS("STRIP_CHART_ADD_POINT", cmd, 1);
      }
    }
  }
}
  
//---------------------------------------------------------------------
// SetStateVars
//    Set the state vars that the user can interact with in the Custon UI
//---------------------------------------------------------------------
void visit_SetStateVars( visit_simulation_data *sim )
{
  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();

  if( appInterface->getStateVars().size() )
  {
    VisItUI_setValueS( "StateVariableGroupBox", "SHOW_WIDGET", 1);

    std::vector< ApplicationInterface::interactiveVar > &vars =
      appInterface->getStateVars();
      
    for( unsigned int i=0; i<vars.size(); ++i )
    {
      ApplicationInterface::interactiveVar &var = vars[i];
      
      var.modified = false;
      
      VisItUI_setTableValueS("StateVariableTable", i, 0, var.component.c_str(), 0);
      VisItUI_setTableValueS("StateVariableTable", i, 1, var.name.c_str(), 0);
      
      switch( var.type )
      {
        case Uintah::TypeDescription::bool_type:
        {
          bool *val = (bool*) var.value;
          VisItUI_setTableValueI("StateVariableTable", i, 2, *val, 1);
        }
        break;
            
        case Uintah::TypeDescription::int_type:
        {
          int *val = (int*) var.value;
          VisItUI_setTableValueI("StateVariableTable", i, 2, *val, 1);
        }
        break;

        case Uintah::TypeDescription::double_type:
        {
          double *val = (double*) var.value;
          VisItUI_setTableValueD("StateVariableTable", i, 2, *val, 1);
        }
        break;

        case Uintah::TypeDescription::Point:
        {
          Point *val = (Point*) var.value;
          VisItUI_setTableValueV("StateVariableTable", i, 2,
                                 val->x(), val->y(), val->z(), 1);
        }
        break;
        
        case Uintah::TypeDescription::Vector:
        {
          Vector *val = (Vector*) var.value;
          VisItUI_setTableValueV("StateVariableTable", i, 2,
                                 val->x(), val->y(), val->z(), 1);
        }
        break;
            
      default:
        throw InternalError(" invalid data type", __FILE__, __LINE__); 
      }
    }
  }
  else
    VisItUI_setValueS( "StateVariableGroupBox", "HIDE_WIDGET", 0);
}


//---------------------------------------------------------------------
// SetDebugStreams
//    Set the debug streams so they can be displayed in the Custon UI
//---------------------------------------------------------------------
void visit_SetDebugStreams( visit_simulation_data *sim )
{
  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();

  VisItUI_setTableValueS("DebugStreamTable",
                         -1, -1, "CLEAR_TABLE", 0);

  if( DebugStream::m_all_debug_streams.size() )
  {
    VisItUI_setValueS( "DebugStreamGroupBox", "SHOW_WIDGET", 1);

    int i = 0;
    
    for (auto iter = DebugStream::m_all_debug_streams.begin();
         iter != DebugStream::m_all_debug_streams.end();
         ++iter, ++i)
    {
      // Add in the stream and state.
      std::string name        = (*iter).second->getName();
      std::string component   = (*iter).second->getComponent();
      // Can not show the description because of spaces.
      // std::string description = (*iter).second->getDescription();
      std::string filename    = (*iter).second->getFilename();
      bool        active      = (*iter).second->active();

      unsigned int col = 0;
      VisItUI_setTableValueS("DebugStreamTable",
                             i, col++, (active ? "true":"false"), 1);
      VisItUI_setTableValueS("DebugStreamTable",
                             i, col++, name.c_str(), 0);
      // VisItUI_setTableValueS("DebugStreamTable",
      //                        i, col++, description.c_str(), 0);
      VisItUI_setTableValueS("DebugStreamTable",
                             i, col++, component.c_str(), 0);
      VisItUI_setTableValueS("DebugStreamTable",
                             i, col++, filename.c_str(), 1);
    }
  }
  else
    VisItUI_setValueS( "DebugStreamGroupBox", "HIDE_WIDGET", 0);
}

//---------------------------------------------------------------------
// SetDouts
//    Set the douts so they can be displayed in the Custon UI
//---------------------------------------------------------------------
void visit_SetDouts( visit_simulation_data *sim )
{
  ApplicationInterface* appInterface =
    sim->simController->getApplicationInterface();

  VisItUI_setTableValueS("DoutTable",
                         -1, -1, "CLEAR_TABLE", 0);

  if( Dout::m_all_douts.size() )
  {
    VisItUI_setValueS( "DoutGroupBox", "SHOW_WIDGET", 1);

    int i = 0;
    
    for (auto iter = Dout::m_all_douts.begin();
         iter != Dout::m_all_douts.end();
         ++iter, ++i)
    {
      // Add in the stream and state.
      std::string name        = (*iter).second->name();
      std::string component   = (*iter).second->component();
      // Can not show the description because of spaces.
      // std::string description = (*iter).second->description();
      std::string filename    = "cout"; // (*iter).second->getFilename();
      bool        active      = (*iter).second->active();

      unsigned int col = 0;
      VisItUI_setTableValueS("DoutTable",
                             i, col++, (active ? "true":"false"), 1);
      VisItUI_setTableValueS("DoutTable",
                             i, col++, name.c_str(), 0);
      // VisItUI_setTableValueS("DoutTable",
      //                        i, col++, description.c_str(), 0);
      VisItUI_setTableValueS("DoutTable",
                             i, col++, component.c_str(), 0);
      VisItUI_setTableValueS("DoutTable",
                             i, col++, filename.c_str(), 0);
    }
  }
  else
    VisItUI_setValueS( "DoutGroupBox", "HIDE_WIDGET", 0);
}

//---------------------------------------------------------------------
// SetDatabase
//    Set the database behavior so they can be displayed in the Custon UI
//---------------------------------------------------------------------
void visit_SetDatabase( visit_simulation_data *sim )
{
  VisItUI_setValueI("LoadExtraElements", sim->loadExtraElements, 1);
}

} // End namespace Uintah

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

#ifndef UINTAH_VISIT_LIBSIM_H
#define UINTAH_VISIT_LIBSIM_H

/**************************************
        
CLASS
   visit_init
        
   Short description...
        
GENERAL INFORMATION
        
   visit_init
        
   Allen R. Sanderson
   Scientific Computing and Imaging Institute
   University of Utah
        
KEYWORDS
   VisIt, libsim, in-situ
        
DESCRIPTION
   Long description...
        
WARNING
        

****************************************/

#include "Core/Grid/Grid.h"
#include "Core/Parallel/ProcessorGroup.h"

#include <VisIt/interfaces/datatypes.h>

#include <map>
#include <string>
#include <utility>

class TimeStepInfo;

namespace Uintah {

class SimulationController;

// Simulation Mode
#define VISIT_SIMMODE_UNKNOWN  0
#define VISIT_SIMMODE_RUNNING  1
#define VISIT_SIMMODE_STOPPED  2

#define VISIT_SIMMODE_STEP       3
#define VISIT_SIMMODE_FINISHED   4
#define VISIT_SIMMODE_TERMINATED 5


typedef struct visit_simulation_data
{
  const ProcessorGroup* myworld;

  // Uintah data members
  SimulationController *simController {nullptr};
  GridP gridP {nullptr};
  
  TimeStepInfo* stepInfo {nullptr};

  int    cycle {0};
  double time  {0};

  // UDA archive variables.
  LoadExtraGeometry loadExtraGeometry {NO_EXTRA_GEOMETRY};
  LoadVariables     loadVariables     {LOAD_ALL_VARIABLES};
  bool forceMeshReload {true};
  std::string mesh_for_patch_data;
  
  int blocking {0};

  // Simulation control members
  int  runMode {VISIT_SIMMODE_RUNNING};  // What the libsim is doing.
  int  simMode {VISIT_SIMMODE_RUNNING};  // What the simulation is doing.

  bool isProc0 {false};
  bool first {false};
  
  bool timeRange {false};
  int timeStart  {0};
  int timeStep   {1};
  int timeStop   {0};
  
  bool imageGenerate {false};
  std::string imageFilename {"NoFileName"};
  int imageHeight {640};
  int imageWidth  {480};
  int imageFormat {2};

  int  stopAtTimeStep     {0};
  bool stopAtLastTimeStep {false};

  // The first row is the strip chart name.
  std::string stripChartNames[5][5];
  
  // Container for storing modiied variables - gets passed to the
  // DataArchiver so they are stored in the index.xml file.  

//std::map< VarName      std::pair<oldValue,    newValue   > >
  std::map< std::string, std::pair<std::string, std::string> > modifiedVars;

  // In-situ machine layout.

  // The root name of the host.
  std::string hostName;
  std::string hostNode;

  // A list of nodes on each switch.
  std::vector< std::vector< unsigned int > > switchNodeList;

  // A table of nodes and the number of cores and memory.
  std::vector< unsigned int > nodeStart  {0};
  std::vector< unsigned int > nodeStop   {0};
  std::vector< unsigned int > nodeCores  {0};

  unsigned int maxNodes, maxCores, xNode, yNode;

  // The index of the switch and node for this core.
  unsigned int switchIndex {static_cast<unsigned int>(-1)};
  unsigned int   nodeIndex {static_cast<unsigned int>(-1)};
  
} visit_simulation_data;

void visit_LibSimArguments(int argc, char **argv);
void visit_InitLibSim(visit_simulation_data *sim);
void visit_EndLibSim(visit_simulation_data *sim);
bool visit_CheckState(visit_simulation_data *sim);

void visit_UpdateSimData( visit_simulation_data *sim, 
                          GridP currentGrid,
                          bool first, bool last );

void visit_Initialize( visit_simulation_data *sim );
  
} // End namespace Uintah

#endif

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

#include <map>
#include <string>
#include <utility>

class TimeStepInfo;

namespace Uintah {

class SimulationController;

/* Simulation Mode */
#define VISIT_SIMMODE_UNKNOWN  0
#define VISIT_SIMMODE_RUNNING  1
#define VISIT_SIMMODE_STOPPED  2

#define VISIT_SIMMODE_STEP       3
#define VISIT_SIMMODE_FINISHED   4
#define VISIT_SIMMODE_TERMINATED 5


typedef struct
{
  // Uintah data members
  SimulationController *simController;
  GridP gridP;
  
  TimeStepInfo* stepInfo;

  int cycle;

  double time;
  double delt;
  double delt_next;

  // UDA archive variables.
  bool useExtraCells;
  bool forceMeshReload;
  std::string mesh_for_patch_data;
  
  int blocking;

  // Simulation control members
  int  runMode;  // What the libsim is doing.
  int  simMode;  // What the simulation is doing.

  int  rank;
  bool isProc0;

  bool first;
  
  bool timeRange;
  int timeStart;
  int timeStep;
  int timeStop;
  
  bool imageGenerate;
  std::string imageFilename;
  int imageHeight;
  int imageWidth;
  int imageFormat;

  int  stopAtTimeStep;
  bool stopAtLastTimeStep;

  // The first row is the strip chart name.
  std::string stripChartNames[5][5];
  
  // Container for storing modiied variables - gets passed to the
  // DataArchiver so they are stored in the index.xml file.  

  //   map< VarName           pair<oldValue,    newValue> >
  std::map< std::string, std::pair<std::string, std::string> > modifiedVars;

} visit_simulation_data;

void visit_LibSimArguments(int argc, char **argv);
void visit_InitLibSim(visit_simulation_data *sim);
void visit_EndLibSim(visit_simulation_data *sim);
bool visit_CheckState(visit_simulation_data *sim);

void visit_UpdateSimData( visit_simulation_data *sim, 
                          GridP currentGrid,
                          double time, double delt, double delt_next,
                          bool first, bool last );

void visit_Initialize( visit_simulation_data *sim );
  
} // End namespace Uintah

#endif

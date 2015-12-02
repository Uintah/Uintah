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

#include "VisItControlInterface_V2.h"
#include "VisItDataInterface_V2.h"

#include <Core/Grid/Grid.h>

#include <sci_defs/mpi_defs.h>

#include <string>

class TimeStepInfo;

namespace Uintah {

class AMRSimulationController;

/* Simulation Mode */
//#define VISIT_SIMMODE_UNKNOWN  0
//#define VISIT_SIMMODE_RUNNING  1
//#define VISIT_SIMMODE_STOPPED  2

#define VISIT_SIMMODE_STEP       3
#define VISIT_SIMMODE_FINISHED   4
#define VISIT_SIMMODE_TERMINATED 5

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
typedef struct
{
  // Uintah data members
  AMRSimulationController *AMRSimController;
  GridP gridP;
  
  TimeStepInfo* stepInfo;

  int cycle;
  double time;
  double delt;

  std::string message;

  int blocking;

  bool useExtraCells;
  bool nodeCentered;
  bool forceMeshReload;

  // Simulation control members
  int  runMode;
  int  simMode;

  bool isProc0;

} visit_simulation_data;


static int visit_BroadcastStringCallback(char *str, int len, int sender);
static int visit_BroadcastIntCallback(int *value, int sender);
static void visit_BroadcastSlaveCommand(int *command);
void visit_SlaveProcessCallback();
int visit_ProcessVisItCommand( visit_simulation_data *sim );

void
visit_ControlCommandCallback(const char *cmd, const char *args, void *cbdata);

void visit_LibSimArguments(int argc, char **argv);
void visit_InitLibSim(visit_simulation_data *sim);
void visit_EndLibSim(visit_simulation_data *sim);
void visit_CheckState(visit_simulation_data *sim);


void visit_CalculateDomainNesting(TimeStepInfo* stepInfo,
                                  bool &forceMeshReload,
                                  int timestate, const std::string &meshname);

visit_handle visit_ReadMetaData(void *cbdata);

visit_handle visit_SimGetMetaData(void *cbdata);
visit_handle visit_SimGetMesh(int domain, const char *name, void *cbdata);
visit_handle visit_SimGetVariable(int domain, const char *name, void *cbdata);

visit_handle visit_SimGetDomainList(const char *name, void *cbdata);
} // End namespace Uintah

#endif

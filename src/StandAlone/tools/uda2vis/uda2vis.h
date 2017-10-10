/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

/*
 *  uda2vis.h: Provides an interface between VisIt's libsim and Uintah.
 *
 *  Written by:
 *   Department of Computer Science
 *   University of Utah
 *   March 2015
 *
 */

#ifndef UINTAH_UDA2VIS_H
#define UINTAH_UDA2VIS_H

#include "StandAlone/tools/uda2vis/udaData.h"

#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Ports/SchedulerP.h>
#include <Core/Grid/GridP.h>

// Define these for the in-situ usage.

namespace Uintah {

TimeStepInfo* getTimeStepInfo(SchedulerP schedulerP,
			      GridP grid,
			      bool useExtraCells);

GridDataRaw* getGridData(SchedulerP schedulerP,
			 GridP gridP,
			 int level_i,
			 int patch_i,
			 std::string variable_name,
			 int material,
			 int low[3],
			 int high[3]);

ParticleDataRaw* getParticleData(SchedulerP schedulerP,
				 GridP gridP,
				 int level_i,
				 int patch_i,
				 std::string variable_name,
				 int material);

void GetLevelAndLocalPatchNumber(TimeStepInfo* stepInfo,
				 int global_patch, 
				 int &level, int &local_patch);
  
int GetGlobalDomainNumber(TimeStepInfo* stepInfo,
			  int level, int local_patch);

void CheckNaNs(double *data, const int num,
	       const char* varname, const int level, const int patch);
}

#endif //UINTAH_UDA2VIS_H

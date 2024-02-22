/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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

#ifndef __DISSOLUTION_H__
#define __DISSOLUTION_H__

//#include <CCA/Components/MPM/Materials/Dissolution/DissolutionMaterialSpec.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SchedulerP.h>
#include <cmath>

namespace Uintah {

  class DataWarehouse;
  class MPMLabel;
  class ProcessorGroup;
  class Patch;
  class VarLabel;
  class Task;

/**************************************

CLASS
   Dissolution
   
   Short description...

GENERAL INFORMATION

   Dissolution.h

   James E. Guilkey
   Laird Avenue Consulting/University of Utah

KEYWORDS
   Dissolution_Model

DESCRIPTION
   Long description...
  
WARNING

****************************************/

  class Dissolution {
    public:
     // Constructor
     Dissolution(const ProcessorGroup* myworld, MPMLabel* Mlb, ProblemSpecP ps);
     virtual ~Dissolution();

     virtual void outputProblemSpec(ProblemSpecP& ps) = 0;

     // Basic dissolution methods
     virtual void computeMassBurnFraction(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset* matls,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw) = 0;
     
     virtual void addComputesAndRequiresMassBurnFrac(SchedulerP & sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls) = 0;

     virtual void setTemperature(double BHTemp);

     virtual void setPhase(std::string LCPhase);

     virtual void setTimeConversionFactor(const double tcf);

     virtual void setGrowthFractionRate(const double QGVF);

    protected:
     MPMLabel* lb;
     double d_temperature;
     double d_timeConversionFactor;
     double d_growthFractionRate;
     std::string d_phase;
    };

//    inline bool compare(double num1, double num2) {
//      //double EPSILON=1.e-20;
//      double EPSILON=1.e-14;
//      return (fabs(num1-num2) <= EPSILON);
//    }

} // End namespace Uintah

#endif // __DISSOLUTION_H__

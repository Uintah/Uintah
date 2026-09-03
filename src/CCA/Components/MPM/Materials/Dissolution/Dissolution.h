/*
 * Copyright © 2026 by Geocosm LLC                                   
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
  class MPMFlags;

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
     Dissolution(const ProcessorGroup* myworld, MPMLabel* Mlb, ProblemSpecP ps,
                                                                MPMFlags* flag);
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
     MPMFlags* flag;
    };

//    inline bool compare(double num1, double num2) {
//      //double EPSILON=1.e-20;
//      double EPSILON=1.e-14;
//      return (fabs(num1-num2) <= EPSILON);
//    }

} // End namespace Uintah

#endif // __DISSOLUTION_H__

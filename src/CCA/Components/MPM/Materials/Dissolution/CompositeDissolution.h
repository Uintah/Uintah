/*
 * Copyright © 2026 by Geocosm LLC                                   
 */

#ifndef __COMPOSITE_DISSOLUTION_H__
#define __COMPOSITE_DISSOLUTION_H__

#include <CCA/Components/MPM/Materials/Dissolution/Dissolution.h>
#include <list>

namespace Uintah {

/**************************************

CLASS
   CompositeDissolution
   
GENERAL INFORMATION

   CompositeDissolution.h

   James Guilkey
   Laird Avenue Consulting
 

KEYWORDS
   Dissolution_Model Composite

DESCRIPTION
   Long description...
  
WARNING

****************************************/

    class CompositeDissolution :public Dissolution {
      public:
         // Constructor
         CompositeDissolution(const ProcessorGroup* myworld, MPMLabel* Mlb);
         virtual ~CompositeDissolution();

         virtual void outputProblemSpec(ProblemSpecP& ps);
         
         // memory deleted on destruction of composite
         void add(Dissolution * m);
         
         // how many 
         size_t size() const { return d_m.size(); }
         
         // Basic dissolution methods
         void computeMassBurnFraction(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset* matls,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw);
         
         void addComputesAndRequiresMassBurnFrac(SchedulerP & sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls);

         void setTemperature(double BHTemp);
         void setPhase(std::string LCPhase);
         void setTimeConversionFactor(double tcf);
         void setGrowthFractionRate(double QGFR);

      private: // hide
         CompositeDissolution(const CompositeDissolution &);
         CompositeDissolution& operator=(const CompositeDissolution &);

      protected: // data
         std::list< Dissolution * > d_m;
      };
      
} // End namespace Uintah

#endif // __COMPOSITE_DISSOLUTION_H__

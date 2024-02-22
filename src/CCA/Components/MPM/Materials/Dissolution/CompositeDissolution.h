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

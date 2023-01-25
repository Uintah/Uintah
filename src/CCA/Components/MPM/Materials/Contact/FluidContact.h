/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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

// Fluid.h

#ifndef __FLUID_H__
#define __FLUID_H__

#include <CCA/Components/MPM/Materials/Contact/Contact.h>
#include<CCA/Components/MPM/Materials/Contact/ContactMaterialSpec.h> 
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>


namespace Uintah {

    class DataWarehouse;
    class MPMLabel;
    class HydroMPMLabel;
    class MPMFlags;
    class ProcessorGroup;
    class Patch;
    class VarLabel;
    class Task;

/**************************************

CLASS
   FluidContact
   
   Short description...

GENERAL INFORMATION

   FluidContact.h

   Hilde Aas Nøst
   Department of Civil Engineering
   Norwegian University of Science and Technology


KEYWORDS
   Contact_Model_Fluid

DESCRIPTION
  One of the derived Contact classes.  This particular
  version is used to apply contact with pore fluid.
  
WARNING
  
****************************************/

      class FluidContact  {
      private:
         
         // Prevent copying of this class
         // copy constructor
         FluidContact(const FluidContact &con);
         FluidContact& operator=(const FluidContact &con);
         
         MaterialManagerP    d_sharedState;
        
         int d_rigid_material;

         double d_vol_const;
         double d_sepFac;
         bool d_compColinearNorms;

         int NGP;
         int NGN;

      public:
         // Constructor
         FluidContact(const ProcessorGroup* myworld,
                         MaterialManagerP& d_sS,MPMLabel* lb, HydroMPMLabel* Hlb,
                         MPMFlags* MFlag);
         
         // Destructor
         virtual ~FluidContact();

         virtual void outputProblemSpec(ProblemSpecP& ps);

         // Basic contact methods
         virtual void exMomInterpolated(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* matls,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw);
         
         virtual void exMomIntegrated(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset* matls,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw);
         
         virtual void addComputesAndRequiresInterpolated(SchedulerP & sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls);

         virtual void addComputesAndRequiresIntegrated(SchedulerP & sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls);
      
      protected:
          MPMLabel* lb;
          MPMFlags* flag;
          HydroMPMLabel* Hlb;
          int    d_oneOrTwoStep;

          ContactMaterialSpec d_matls;
      };
} // End namespace Uintah
      

#endif /* __FLUID_H__ */


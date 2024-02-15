/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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

// Penalty.h

#ifndef __PENALTY_H__
#define __PENALTY_H__

#include <CCA/Components/MPM/Materials/Contact/Contact.h>
#include <CCA/Components/MPM/Materials/Contact/ContactMaterialSpec.h> 
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/MaterialManagerP.h>


namespace Uintah {
/**************************************

CLASS
   PenaltyContact
   
   Short description...

GENERAL INFORMATION

   PenaltyContact.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   Contact_Model_Penalty

DESCRIPTION
  One of the derived Contact classes.  This particular
  version is used to apply Coulombic frictional contact.
  
WARNING
  
****************************************/

      class PenaltyContact : public Contact {
      private:
         
         // Prevent copying of this class
         // copy constructor
         PenaltyContact(const PenaltyContact &con);
         PenaltyContact& operator=(const PenaltyContact &con);
         
         MaterialManagerP    d_materialManager;
         
         // Coefficient of friction
         double d_mu;
         // Nodal volume fraction that must occur before contact is applied
         double d_vol_const;
         double d_sepFac;
         int NGP;
         int NGN;

      public:
         // Constructor
         PenaltyContact(const ProcessorGroup* myworld,
                         ProblemSpecP& ps, MaterialManagerP& d_sS,MPMLabel* lb,
                         MPMFlags* MFlag);
         
         // Destructor
         virtual ~PenaltyContact();

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
      };
} // End namespace Uintah

#endif /* __PENALTY_H__ */

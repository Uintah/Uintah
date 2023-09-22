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

// FrictionContactLRVar.h

#ifndef __FRICTIONLRVar
#define __FRICTIONLRVar

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
   FrictionContactLRVar
   
   This version of contact is based on John Nairn and Chad
   Hammerquist's 2019 manuscript that describes the use of logistic
   regression to find a common normal between objects, and uses
   particle geometry to find the most prominent portion of a particle
   at each node, and applies contact when contacting materials'
   prominences overlap.

GENERAL INFORMATION

   FrictionContactLRVar.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   Contact_Model_Friction

DESCRIPTION
  One of the derived Contact classes.  This particular
  version is used to apply Coulombic frictional contact.
  
WARNING
  
****************************************/

      class FrictionContactLRVar : public Contact {
      private:
         
         // Prevent copying of this class
         // copy constructor
         FrictionContactLRVar(const FrictionContactLRVar &con);
         FrictionContactLRVar& operator=(const FrictionContactLRVar &con);
         
         MaterialManagerP    d_materialManager;
         
         int NGP;
         int NGN;

         // Private Data
         // Coefficient of friction vs Color data
         std::vector<double> d_Color;
         std::vector<double> d_mu;
         int d_material;

      public:
         // Constructor
         FrictionContactLRVar(const ProcessorGroup* myworld,
                         ProblemSpecP& ps, MaterialManagerP& d_sS,MPMLabel* lb,
                         MPMFlags* MFlag);
         
         // Destructor
         virtual ~FrictionContactLRVar();

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

         inline double findMuFromColor(double color){
           int n_entries = static_cast<int>(d_Color.size());
           if (color >= d_Color[n_entries-1]){
              return  d_mu[n_entries-1];
           }

           for (int ii = 1; ii < n_entries; ++ii) {
            if (color <= d_Color[ii]) {
              double s = (d_Color[ii]-color)/(d_Color[ii]-d_Color[ii-1]);
              return  d_mu[ii-1]*s + d_mu[ii]*(1.0-s);
             }
           }

           return d_mu[0];
         }
      };
} // End namespace Uintah

#endif /* __FRICTIONLRVar */

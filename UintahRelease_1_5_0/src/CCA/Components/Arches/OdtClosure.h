/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

//----- OdtClosure.h --------------------------------------------------

#ifndef Uintah_Component_Arches_OdtClosure_h
#define Uintah_Component_Arches_OdtClosure_h

/**************************************
CLASS
   OdtClosure
   
   Class OdtClosure is an LES model for
   computing sub-grid scale turbulent stress.


GENERAL INFORMATION
   OdtClosure.h - declaration of the class
   
   Author: Zhaosheng Gao (zgao@crsim.utah.edu)
      
   Creation Date:   November 12, 2004
   
   C-SAFE 
   

KEYWORDS


DESCRIPTION
   Class OdtClosure is an LES model for
   computing sub-grid scale turbulent stress.


WARNING
   none
****************************************/

#include <CCA/Components/Arches/SmagorinskyModel.h>


namespace Uintah {
class PhysicalConstants;
class BoundaryCondition;

class OdtClosure: public SmagorinskyModel {

public:

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  // Blank constructor for OdtClosure.
  OdtClosure(const ArchesLabel* label, 
             const MPMArchesLabel* MAlb,
             PhysicalConstants* phyConsts,
             BoundaryCondition* bndryCondition);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual destructor for OdtClosure.
  virtual ~OdtClosure();

  // GROUP: Problem Setup :
  ///////////////////////////////////////////////////////////////////////
  // Set up the problem specification database
  virtual void problemSetup(const ProblemSpecP& db);

  // GROUP: Schedule Action :
  ///////////////////////////////////////////////////////////////////////
  // Schedule the initialization of the Smagorinsky Coefficient
  //    [in] 
  //        data User data needed for solve 
  virtual void sched_initializeSmagCoeff(SchedulerP&,
                                         const PatchSet* patches,
                                         const MaterialSet* matls,
                                         const TimeIntegratorLabel* timelabels);

  ///////////////////////////////////////////////////////////////////////
  // Schedule the recomputation of Turbulence Model data
  //    [in] 
  //        data User data needed for solve 
  void sched_initializeOdtvariable(SchedulerP&,
                                   const PatchSet* patches,
                                   const MaterialSet* matls,
                                   const TimeIntegratorLabel* timelabels);
                                   
  virtual void sched_reComputeTurbSubmodel(SchedulerP&,
                             const PatchSet* patches,
                             const MaterialSet* matls,
                             const TimeIntegratorLabel* timelabels);

  // GROUP: Schedule Action :
  ///////////////////////////////////////////////////////////////////////
  // Schedule the computation of Turbulence Model data
  //    [in] 
  //        data User data needed for solve 
  virtual void sched_computeScalarVariance(SchedulerP&,
                                           const PatchSet* patches,
                                           const MaterialSet* matls,
                                           const TimeIntegratorLabel* timelabels);

  virtual void sched_computeScalarDissipation(SchedulerP&,
                                              const PatchSet* patches,
                                              const MaterialSet* matls,
                                              const TimeIntegratorLabel* timelabels);

protected:
      int d_odtPoints; // number of odt Points at each LES cell
private:
  
  // GROUP: Constructors (not instantiated):
  ////////////////////////////////////////////////////////////////////////
  // Blank constructor for OdtClosure.
  OdtClosure();

  // GROUP: Action Methods (private)  :
  ///////////////////////////////////////////////////////////////////////

  void initializeSmagCoeff( const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw,
                            const TimeIntegratorLabel* timelabels);

  // Actually reCalculate the Turbulence sub model
  //    [in] 
  //        documentation here
  void initializeOdtvariable( const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw,
                              const TimeIntegratorLabel* timelabels);
                            
  void reComputeTurbSubmodel(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             const TimeIntegratorLabel* timelabels);

  ///////////////////////////////////////////////////////////////////////
  // Actually Calculate the subgrid scale variance
  //    [in] 
  //        documentation here
  void computeScalarVariance(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             const TimeIntegratorLabel* timelabels );

  void computeScalarDissipation(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                const TimeIntegratorLabel* timelabels );

private:
      double d_CF; //model constant
      double d_viscosity; // moleculor viscosity 

}; // End class OdtClosure

/*______________________________________________________________________
 *   different data types 
 *______________________________________________________________________*/ 
  
}

#endif

// $Log : $


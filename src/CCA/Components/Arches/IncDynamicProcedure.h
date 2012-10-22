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

//----- IncDynamicProcedure.h --------------------------------------------------

#ifndef Uintah_Component_Arches_IncDynamicProcedure_h
#define Uintah_Component_Arches_IncDynamicProcedure_h

/**************************************
CLASS
   IncDynamicProcedure
   
   Class IncDynamicProcedure is an LES model for
   computing sub-grid scale turbulent viscosity.


GENERAL INFORMATION
   IncDynamicProcedure.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
      
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   

KEYWORDS


DESCRIPTION
   Class IncDynamicProcedure is an LES model for
   computing sub-grid scale turbulent viscosity.


WARNING
   none
****************************************/

#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/TurbulenceModel.h>
#include <iostream>
using namespace std;

namespace Uintah {
class PhysicalConstants;
class BoundaryCondition;


class IncDynamicProcedure: public TurbulenceModel {

public:

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  // Blank constructor for IncDynamicProcedure.
  IncDynamicProcedure(const ArchesLabel* label, 
                      const MPMArchesLabel* MAlb,
                      PhysicalConstants* phyConsts,
                      BoundaryCondition* bndryCondition);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual destructor for IncDynamicProcedure.
  virtual ~IncDynamicProcedure();

  // GROUP: Problem Setup :
  ///////////////////////////////////////////////////////////////////////
  // Set up the problem specification database
  virtual void problemSetup(const ProblemSpecP& db);

  // GROUP: Schedule Action :
  ///////////////////////////////////////////////////////////////////////
  // Schedule the recomputation of Turbulence Model data
  //    [in] 
  //        data User data needed for solve 
  virtual void sched_reComputeTurbSubmodel(SchedulerP&, 
                                           const PatchSet* patches,
                                           const MaterialSet* matls,
                                           const TimeIntegratorLabel* timelabels);

  virtual void sched_computeScalarVariance(SchedulerP&, 
                                           const PatchSet* patches,
                                           const MaterialSet* matls,
                                           const TimeIntegratorLabel* timelabels);

  virtual void sched_computeScalarDissipation(SchedulerP&,
                                              const PatchSet* patches,
                                              const MaterialSet* matls,
                                              const TimeIntegratorLabel* timelabels);

  // GROUP: Access Methods :
  ///////////////////////////////////////////////////////////////////////
  // Get the molecular viscosity
  double getMolecularViscosity() const; 

  ////////////////////////////////////////////////////////////////////////
  // Get the Smagorinsky model constant
  double getSmagorinskyConst() const {
    cerr << "There is no Smagorinsky constant in IncDynamic Procedure" << endl;
    exit(0);
    return 0;
  }
  inline void set3dPeriodic(bool periodic) {
    d_3d_periodic = periodic;
  }
  inline double getTurbulentPrandtlNumber() const {
    return d_turbPrNo;
  }
  inline void setTurbulentPrandtlNumber(double turbPrNo) {
    d_turbPrNo = turbPrNo;
  }
  inline bool getDynScalarModel() const {
    return false;
  }

protected:
      PhysicalConstants* d_physicalConsts;
      BoundaryCondition* d_boundaryCondition;

private:
  
  // GROUP: Constructors (not instantiated):
  ////////////////////////////////////////////////////////////////////////
  // Blank constructor for IncDynamicProcedure.
  IncDynamicProcedure();

  // GROUP: Action Methods (private)  :


  ///////////////////////////////////////////////////////////////////////
  // Actually reCalculate the Turbulence sub model
  //    [in] 
  //        documentation here
  void reComputeTurbSubmodel(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             const TimeIntegratorLabel* timelabels);

  void reComputeFilterValues(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             const TimeIntegratorLabel* timelabels);

  void reComputeSmagCoeff(const ProcessorGroup*,
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
                             const TimeIntegratorLabel* timelabels);
                             
  void computeScalarDissipation(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                const TimeIntegratorLabel* timelabels);

 protected:
  double d_CFVar; // model constant for mixture fraction variance
  double d_turbPrNo; // turbulent prandtl number
  bool d_filter_cs_squared; //option for filtering Cs^2 in IncDynamic Procedure
  bool d_3d_periodic;
  bool d_filter_var_limit_scalar;


 private:

      // const VarLabel* variables 

 }; // End class IncDynamicProcedure
} // End namespace Uintah
  
  

#endif

// $Log : $


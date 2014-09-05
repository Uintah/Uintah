/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


//----- CompLocalDynamicProcedure.h --------------------------------------------------

#ifndef Uintah_Component_Arches_CompLocalDynamicProcedure_h
#define Uintah_Component_Arches_CompLocalDynamicProcedure_h

#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/TurbulenceModel.h>
#include <iostream>
using namespace std;

namespace Uintah {

/** 
  @class  CompLocalDynamicProcedure
  @author Rajesh Rawat, Zhaosheng Gao
  @date   March 2000, May 2005
  
  @brief  An LES model for computing the subgrid scale stress term \f$ \tau_{SGS} \f$.
  
  */

class PhysicalConstants;
class BoundaryCondition;


class CompLocalDynamicProcedure: public TurbulenceModel {

public:
  
  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  // Blank constructor for CompLocalDynamicProcedure.
  CompLocalDynamicProcedure(const ArchesLabel* label, 
                            const MPMArchesLabel* MAlb,
                            PhysicalConstants* phyConsts,
                            BoundaryCondition* bndryCondition);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual destructor for CompLocalDynamicProcedure.
  virtual ~CompLocalDynamicProcedure();

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
  void sched_reComputeSmagCoeff( SchedulerP& sched, 
                                 const PatchSet* patches,
                                 const MaterialSet* matls,
                                 const TimeIntegratorLabel* timelabels );

  virtual void sched_reComputeTurbSubmodel(SchedulerP&, 
                                           const PatchSet* patches,
                                           const MaterialSet* matls,
                                           const TimeIntegratorLabel* timelabels);

  virtual void sched_computeScalarVariance(SchedulerP&, 
                                           const PatchSet* patches,
                                           const MaterialSet* matls,
                                           const TimeIntegratorLabel* timelabels,
                                           bool d_EKTCorrection,
                                           bool doing_EKT_now);

  virtual void sched_computeScalarDissipation(SchedulerP&,
                                              const PatchSet* patches,
                                              const MaterialSet* matls,
                                              const TimeIntegratorLabel* timelabels,
                                              bool d_EKTCorrection,
                                              bool doing_EKT_now);

  // GROUP: Access Methods :
  ///////////////////////////////////////////////////////////////////////
  // Get the molecular viscosity
  double getMolecularViscosity() const; 

  ////////////////////////////////////////////////////////////////////////
  // Get the Smagorinsky model constant
  double getSmagorinskyConst() const {
    cerr << "There is no Smagorinsky constant in CompDynamic Procedure" << endl;
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
    return d_dynScalarModel;
  }
  Arches* d_arches;
protected:
  PhysicalConstants* d_physicalConsts;
  BoundaryCondition* d_boundaryCondition;

private:

  // GROUP: Constructors (not instantiated):
  ////////////////////////////////////////////////////////////////////////
  // Blank constructor for CompLocalDynamicProcedure.
  CompLocalDynamicProcedure();

  // GROUP: Action Methods (private)  :


  void initializeSmagCoeff( const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw,
                            const TimeIntegratorLabel* timelabels);

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

  void reComputeStrainRateTensors(const ProcessorGroup*,
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
                             const TimeIntegratorLabel* timelabels,
                             bool d_EKTCorrection,
                             bool doing_EKT_now);
                             
  void computeScalarDissipation(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                const TimeIntegratorLabel* timelabels,
                                bool d_EKTCorrection,
                                bool doing_EKT_now);

protected:
  double d_CF;
  double d_factorMesh;      // lengthscale = fac_mesh*meshsize
  double d_filterl;         // prescribed filter length scale
  double d_CFVar;           // model constant for mixture fraction variance
  double d_turbPrNo;        // turbulent prandtl number
  bool d_filter_cs_squared; //option for filtering Cs^2 in CompDynamic Procedure
  bool d_3d_periodic;
  bool d_dynScalarModel;
  double d_lower_limit;

 private:

      // const VarLabel* variables 

 }; // End class CompLocalDynamicProcedure
} // End namespace Uintah
  
  

#endif

// $Log : $


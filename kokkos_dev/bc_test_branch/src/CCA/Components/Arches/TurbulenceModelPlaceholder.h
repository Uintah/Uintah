/*
 
 The MIT License
 
 Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and
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

//----- TurbulenceModelPlaceholder.h --------------------------------------------------

#ifndef Uintah_Components_Arches_TurbulenceModelPlaceholder_h
#define Uintah_Components_Arches_TurbulenceModelPlaceholder_h


#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/TurbulenceModel.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/CCVariable.h>

#ifdef WASATCH_IN_ARCHES
  #include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulentViscosity.h>
#endif

namespace Uintah {
  class PhysicalConstants;
  class BoundaryCondition;
  
  /**
   *  \class  TurbulenceModelPlaceholder
   *  \author Tony Saad
   *  \date   October, 2012
   *
   *  \brief Allow Arches to use Wasatch's turbulence models.
   *
   */
  
  class TurbulenceModelPlaceholder: public TurbulenceModel {
    
  public:
    
    TurbulenceModelPlaceholder(const ArchesLabel* label,
                     const MPMArchesLabel* MAlb,
                     PhysicalConstants* phyConsts,
                     BoundaryCondition* bndryCondition);
    
    ~TurbulenceModelPlaceholder();
    
    void problemSetup(const ProblemSpecP& db);
    
    void sched_reComputeTurbSubmodel(SchedulerP&,
                                             const PatchSet* patches,
                                             const MaterialSet* matls,
                                             const TimeIntegratorLabel* timelabels);
    
        
    void sched_computeScalarVariance(SchedulerP&,
                                             const PatchSet* patches,
                                             const MaterialSet* matls,
                                             const TimeIntegratorLabel* timelabels)
    {;}
    
    void sched_computeScalarDissipation(SchedulerP&,
                                                const PatchSet* patches,
                                                const MaterialSet* matls,
                                                const TimeIntegratorLabel* timelabels)
    {;}
    
    double getMolecularViscosity() const;
    
    double getSmagorinskyConst() const {
      return d_CF;
    }
    inline void set3dPeriodic(bool periodic) {}
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
    
    TurbulenceModelPlaceholder();

    void reComputeTurbSubmodel(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw,
                               const TimeIntegratorLabel* timelabels);        
  protected:
    double d_CF; //model constant
    double d_CFVar; // model constant for mixture fraction variance
    double d_turbPrNo; // turbulent prandtl number    
  }; // End class TurbulenceModelPlaceholder
} // End namespace Uintah



#endif // Uintah_Components_Arches_TurbulenceModelPlaceholder_h


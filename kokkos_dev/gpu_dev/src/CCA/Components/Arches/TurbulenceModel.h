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


//----- TurbulenceModel.h --------------------------------------------------

#ifndef Uintah_Component_Arches_TurbulenceModel_h
#define Uintah_Component_Arches_TurbulenceModel_h

/**************************************
CLASS
   TurbulenceModel
   
   Class TurbulenceModel is an abstract base class
   which defines the operations needed to compute
   unresolved turbulence submodels

GENERAL INFORMATION
   TurbulenceModel.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
      
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class TurbulenceModel is an abstract base class
   which defines the operations needed to compute
   unresolved turbulence submodels

WARNING
   none
****************************************/

#include <CCA/Components/Arches/Arches.h>
#ifdef PetscFilter
#include <CCA/Components/Arches/Filter.h>
#endif
namespace Uintah {
class TimeIntegratorLabel;
class TurbulenceModel
{
public:

      // GROUP: Constructors:
      ////////////////////////////////////////////////////////////////////////
      // Blank constructor for TurbulenceModel.
      TurbulenceModel(const ArchesLabel* label, 
                      const MPMArchesLabel* MAlb);

      // GROUP: Destructors:
      ////////////////////////////////////////////////////////////////////////
      // Virtual destructor for TurbulenceModel.
      virtual ~TurbulenceModel();
#ifdef PetscFilter
      inline void setFilter(Filter* filter) {
        d_filter = filter;
      }
#endif
      // GROUP: Access Methods :
      ///////////////////////////////////////////////////////////////////////
      // Get the molecular viscisity
      virtual double getMolecularViscosity() const = 0;

      ////////////////////////////////////////////////////////////////////////
      // Get the Smagorinsky model constant
      virtual double getSmagorinskyConst() const = 0;

       // GROUP: Problem Setup :
      ///////////////////////////////////////////////////////////////////////
      // Interface for Set up the problem specification database
      virtual void problemSetup(const ProblemSpecP& db) = 0;


      // access function
#ifdef PetscFilter
      Filter* getFilter() const{
        return d_filter;
      }

      void sched_initFilterMatrix(const LevelP&, 
                                  SchedulerP&, 
                                  const PatchSet* patches,
                                  const MaterialSet* matls);

#endif
      virtual void set3dPeriodic(bool periodic) = 0;
      virtual double getTurbulentPrandtlNumber() const = 0;
      virtual void setTurbulentPrandtlNumber(double turbPrNo) = 0;
      virtual bool getDynScalarModel() const = 0;

      inline void setCombustionSpecifics(bool calcScalar,
                                         bool calcEnthalpy,
                                         bool calcReactingScalar) {
        d_calcScalar = calcScalar;
        d_calcEnthalpy = calcEnthalpy;
        d_calcReactingScalar = calcReactingScalar;
      }

      inline void modelVariance(bool calcVariance) {
        d_calcVariance = calcVariance;
      }
      inline void setMixedModel(bool mixedModel) {
        d_mixedModel = mixedModel;
      }
      inline bool getMixedModel() const {
        return d_mixedModel;
      }

      // GROUP: Schedule Action :
      ///////////////////////////////////////////////////////////////////////
      // Interface for Schedule the recomputation of Turbulence Model data
      //    [in] 
      //        data User data needed for solve 
      virtual void sched_reComputeTurbSubmodel(SchedulerP&,
                                               const PatchSet* patches,
                                               const MaterialSet* matls,
                                               const TimeIntegratorLabel* timelabels) = 0;


      virtual void sched_computeScalarVariance(SchedulerP&,
                                               const PatchSet* patches,
                                               const MaterialSet* matls,
                                               const TimeIntegratorLabel* timelabels) = 0;
                                               
      virtual void sched_computeScalarDissipation(SchedulerP&,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls,
                                                  const TimeIntegratorLabel* timelabels) = 0;
 protected:

      const ArchesLabel* d_lab;
      const MPMArchesLabel* d_MAlab;

#ifdef PetscFilter
      Filter* d_filter;
#endif
      bool d_calcScalar, d_calcEnthalpy, d_calcReactingScalar;
      bool d_calcVariance;
      std::string d_mix_frac_label_name; 
      const VarLabel* d_mf_label;

private:
bool d_mixedModel;
#ifdef PetscFilter
      void initFilterMatrix(const ProcessorGroup* pg,
                            const PatchSubset* patches,
                            const MaterialSubset*,
                            DataWarehouse*,
                            DataWarehouse* new_dw);
#endif


}; // End class TurbulenceModel
} // End namespace Uintah
  
  

#endif

// $Log :$




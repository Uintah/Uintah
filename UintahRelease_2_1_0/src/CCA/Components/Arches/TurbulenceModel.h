/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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


KEYWORDS


DESCRIPTION
   Class TurbulenceModel is an abstract base class
   which defines the operations needed to compute
   unresolved turbulence submodels

WARNING
   none
****************************************/

#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/Filter.h>
#include <Core/Grid/Variables/VarLabel.h>

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
      inline void setFilter(Filter* filter) {
        d_filter = filter;
      }
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
      Filter* getFilter() const{
        return d_filter;
      }

      virtual void set3dPeriodic(bool periodic) = 0;

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
                                               const LevelP& level,
                                               const MaterialSet* matls,
                                               const TimeIntegratorLabel* timelabels) = 0;

      void sched_computeFilterVol( SchedulerP& sched,
                                   const LevelP& level,
                                   const MaterialSet* matls );

      void sched_carryForwardFilterVol( SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls );


 protected:

      const ArchesLabel* d_lab;
      const MPMArchesLabel* d_MAlab;
      const VarLabel* d_dissipationRateLabel;


      Filter* d_filter;
      bool d_calcVariance;
      std::string d_mix_frac_label_name;
      const VarLabel* d_mf_label;

      void problemSetupCommon( const ProblemSpecP& params );

  private:

    bool d_mixedModel;

    bool d_use_old_filter;
    int d_filter_width;
    std::string d_filter_type;

    void computeFilterVol( const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset*,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw );

    void carryForwardFilterVol( const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw );

}; // End class TurbulenceModel
} // End namespace Uintah



#endif

// $Log :$

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

//----- CompDynamicProcedure.h --------------------------------------------------

#ifndef Uintah_Component_Arches_CompDynamicProcedure_h
#define Uintah_Component_Arches_CompDynamicProcedure_h

/**************************************
CLASS
   CompDynamicProcedure

   Class CompDynamicProcedure is an LES model for
   computing sub-grid scale turbulent viscosity.


GENERAL INFORMATION
   CompDynamicProcedure.h - declaration of the class

   Author: Stanislav Borodai (borodai@crsim.utah.edu), developed based on
   IncDynamicProcedure

   Creation Date:   Mar 1, 2000

   C-SAFE


KEYWORDS


DESCRIPTION
   Class CompDynamicProcedure is an LES model for
   computing sub-grid scale turbulent viscosity.


WARNING
   none
****************************************/

#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/TurbulenceModel.h>
#include <iostream>

namespace Uintah {

class PhysicalConstants;
class BoundaryCondition;

class CompDynamicProcedure : public TurbulenceModel {

public:

  CompDynamicProcedure(const ArchesLabel* label,
                       const MPMArchesLabel* MAlb,
                       PhysicalConstants* phyConsts,
                       BoundaryCondition* bndryCondition);

  virtual ~CompDynamicProcedure();

  virtual void problemSetup(const ProblemSpecP& db);

  virtual void sched_reComputeTurbSubmodel(SchedulerP&,
                                           const LevelP& level,
                                           const MaterialSet* matls,
                                           const TimeIntegratorLabel* timelabels);


  double getMolecularViscosity() const;

  double getSmagorinskyConst() const {
    std::cerr << "There is no Smagorinsky constant in CompDynamic Procedure" << std::endl;
    exit(0);
    return 0;
  }
  inline void set3dPeriodic(bool periodic) {
    d_3d_periodic = periodic;
  }

protected:
      PhysicalConstants* d_physicalConsts;
      BoundaryCondition* d_boundaryCondition;

private:

  CompDynamicProcedure();

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


  bool d_filter_cs_squared; //option for filtering Cs^2 in CompDynamic Procedure
  bool d_3d_periodic;

  const VarLabel* d_denRefArrayLabel;

  void apply_zero_neumann( const Patch* patch, CCVariable<double>& var,
                           CCVariable<double>& var2, constCCVariable<double> vol_fraction ){

    std::vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);
    Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;

    for( std::vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){

      Patch::FaceType face = *itr;
      IntVector f_dir = patch->getFaceDirection(face);

      for( CellIterator iter=patch->getFaceIterator(face, MEC); !iter.done(); iter++) {
        IntVector c = *iter;

        if ( vol_fraction[c] > 1e-10 ){
          var[c] = var[c-f_dir];
          var2[c] = var2[c-f_dir];
        }
      }
    }
  }

 private:

 }; // End class CompDynamicProcedure
} // End namespace Uintah



#endif

// $Log : $

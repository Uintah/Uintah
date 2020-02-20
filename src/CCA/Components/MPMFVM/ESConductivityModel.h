/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef UINTAH_CCA_COMPONENTS_MPMFVM_ESCONDUCTIVITYMODEL_H
#define UINTAH_CCA_COMPONENTS_MPMFVM_ESCONDUCTIVITYMODEL_H

#include <CCA/Ports/SchedulerP.h>

#include <Core/Geometry/Point.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/MaterialManager.h>

#include <string>

namespace Uintah{
/*************************************************
 *
 * CLASS
 *   ESConductivityModel
 *
 *   This model will compute the face center
 *   conductivity values that is used in the
 *   ElectrostaticSolve component.
 *
 *   The is a base class model that can be
 *   extended to implement new conductivity
 *   models.
 *
 *************************************************/

  class FVMLabel;
  class MPMFlags;
  class MPMLabel;

  class ConductivityEquation;
  
  class DataWarehouse;
  
  class ESConductivityModel{
    public:
      ESConductivityModel(MaterialManagerP& materialManager,
                          MPMFlags* mpm_flags,
                          MPMLabel* mpm_lb,
			  FVMLabel* fvm_lb,
                          std::string& model_type);

      virtual ~ESConductivityModel();

      virtual void scheduleComputeConductivity(SchedulerP& sched,
                                               const PatchSet* patches,
                                               const MaterialSet* all_matls,
                                               const MaterialSubset* one_matls);

    protected:
      virtual void computeConductivity(const ProcessorGroup* pg,
                                       const PatchSubset* patches,
                                       const MaterialSubset*,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw);

      virtual double distanceFunc(Point p1, Point p2);

    private:
      std::string d_model_type;

      Ghost::GhostType d_gac;
      double d_TINY_RHO;
      MaterialManagerP d_materialManager;
      MPMLabel* d_mpm_lb;
      FVMLabel* d_fvm_lb;
      MPMFlags* d_mpm_flags;
      ConductivityEquation* d_conductivity_equation;
  };
}
#endif // End of UINTAH_CCA_COMPONENTS_MPMFVM_ESCONDUCTIVITYMODEL_H

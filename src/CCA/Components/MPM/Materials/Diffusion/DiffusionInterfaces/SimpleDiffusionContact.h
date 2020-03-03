/*
 * DiscreteInterface.h
 *
 *  Created on: Feb 18, 2017
 *      Author: jbhooper
 *
 *
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

#ifndef SRC_CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONINTERFACES_SIMPLEINTERFACE_H_
#define SRC_CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONINTERFACES_SIMPLEINTERFACE_H_

#include <CCA/Components/MPM/Materials/Diffusion/DiffusionInterfaces/SDInterfaceModel.h>
#include <CCA/Components/MPM/Core/MPMDiffusionLabel.h>

#include <CCA/Ports/DataWarehouse.h>

#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/ParticleVariable.h>

namespace Uintah
{
  class SimpleSDInterface : public SDInterfaceModel
  {
    public:
       SimpleSDInterface(
                           ProblemSpecP     & probSpec  ,
                           MaterialManagerP & simState  ,
                           MPMFlags         * mFlags    ,
                           MPMLabel         * mpmLabel
                          );
      ~SimpleSDInterface();

      virtual void addComputesAndRequiresInterpolated(
                                                            SchedulerP  & sched   ,
                                                      const PatchSet    * patches ,
                                                      const MaterialSet * matls
                                                     );

      virtual void sdInterfaceInterpolated(
                                           const ProcessorGroup *         ,
                                           const PatchSubset    * patches ,
                                           const MaterialSubset * matls   ,
                                                 DataWarehouse  * old_dw  ,
                                                 DataWarehouse  * new_dw
                                          );

      virtual void addComputesAndRequiresDivergence(
                                                          SchedulerP  & sched   ,
                                                    const PatchSet    * patches ,
                                                    const MaterialSet * matls
                                                   );

      virtual void  sdInterfaceDivergence(
                                          const ProcessorGroup  *         ,
                                          const PatchSubset     * patches ,
                                          const MaterialSubset  * matl    ,
                                                DataWarehouse   * old_dw  ,
                                                DataWarehouse   * new_dw
                                         );

      virtual void  outputProblemSpec(
                                      ProblemSpecP  & ps
                                     );
    private:
      inline bool compare(double num1, double num2)
      {
            //double EPSILON=1.e-20;
            double EPSILON=1.e-15;
            double diff = fabs(num1-num2);
            bool compareTrue = (diff <= EPSILON);
            return (compareTrue);
      }

      SimpleSDInterface(const SimpleSDInterface &);
      SimpleSDInterface&  operator=(const SimpleSDInterface&);

  };
}



#endif /* SRC_CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONINTERFACES_SIMPLEINTERFACE_H_ */

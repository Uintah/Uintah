/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <CCA/Components/ElectroChem/Diffusion.h>

#include <iostream>

using namespace Uintah;

Diffusion::Diffusion(const ProcessorGroup* myworld,
                     const MaterialManagerP materialManager)
    : ApplicationCommon(myworld, materialManager) {

  d_delt       = 0.0;
  d_diff_coeff = 0.0;
}
    
Diffusion::~Diffusion(){
}

void Diffusion::problemSetup(const ProblemSpecP& ps,
                             const ProblemSpecP& restart_ps,
                                   GridP&        grid){
  std::cout << "!!!!!!!!!! In  problemSetup" << std::endl;
  ProblemSpecP root_ps = 0;
  if (restart_ps){
    root_ps = restart_ps;
  } else{
    root_ps = ps;
  }

  ProblemSpecP ec_ps = root_ps->findBlock("ElectroChem");

  ec_ps->require("delt",       d_delt);
  ec_ps->require("diff_coeff", d_diff_coeff);

  std::cout << "!!!!!!!!!! Out problemSetup" << std::endl;
}

void Diffusion::scheduleInitialize(const LevelP&     level,
                                         SchedulerP& sched){
}

void Diffusion::initialize(const ProcessorGroup* pg,
                           const PatchSubset*    patches,
                           const MaterialSubset* matls,
                                 DataWarehouse*  old_dw,
                                 DataWarehouse*  new_dw){
}

void Diffusion::scheduleRestartInitialize(const LevelP&     level,
                                                SchedulerP& sched){
}

void Diffusion::scheduleComputeStableTimeStep(const LevelP&     level,
                                                    SchedulerP& sched){
}

void Diffusion::computeStableTimeStep(const ProcessorGroup* pg,
                                      const PatchSubset*    patches,
                                      const MaterialSubset* matls,
                                            DataWarehouse*  old_dw,
                                            DataWarehouse*  new_dw){
}

void Diffusion::scheduleTimeAdvance(const LevelP&     level,
                                          SchedulerP& sched){
}

void Diffusion::timeAdvance(const ProcessorGroup* pg,
                            const PatchSubset*    patches,
                            const MaterialSubset* matls,
                                  DataWarehouse*  old_dw,
                                  DataWarehouse*  new_dw){
}

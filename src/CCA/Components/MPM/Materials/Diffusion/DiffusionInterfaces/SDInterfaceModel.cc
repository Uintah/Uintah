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

#include <CCA/Components/MPM/Materials/Diffusion/DiffusionInterfaces/SDInterfaceModel.h>

#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Materials/Diffusion/DiffusionModels/ScalarDiffusionModel.h>

using namespace Uintah;

SDInterfaceModel::SDInterfaceModel(ProblemSpecP         & ps      ,
                                   MaterialManagerP     & sS      ,
                                   MPMFlags             * Mflag   ,
                                   MPMLabel             * mpm_lb  )
                                  : d_materials_list(ps)
{
  d_mpm_lb = mpm_lb;
  d_materialManager = sS;
  d_mpm_flags = Mflag;

  sdInterfaceRate = VarLabel::create("g.dCdt_interface",
                                     NCVariable<double>::getTypeDescription());
  sdInterfaceFlag = VarLabel::create("g.interfaceFlag",
                                     NCVariable<int>::getTypeDescription());
}

SDInterfaceModel::~SDInterfaceModel() {
  VarLabel::destroy(sdInterfaceRate);
  VarLabel::destroy(sdInterfaceFlag);
}

void SDInterfaceModel::addComputesAndRequiresInterpolated(      SchedulerP  & sched   ,
                                                          const PatchSet    * patches ,
                                                          const MaterialSet * matls
                                                         )
{
}

void SDInterfaceModel::sdInterfaceInterpolated(
                                               const ProcessorGroup *         ,
                                               const PatchSubset    * patches ,
                                               const MaterialSubset * matls   ,
                                                     DataWarehouse  * old_dw  ,
                                                     DataWarehouse  * new_dw
                                              )
{
}

void SDInterfaceModel::addComputesAndRequiresDivergence(      SchedulerP  & sched   ,
                                                        const PatchSet    * patches ,
                                                        const MaterialSet * matls   )
{
  Task* task = scinew Task("SDInterfaceModel::sdInterfaceDivergence", this,
                           &SDInterfaceModel::sdInterfaceDivergence);
  // By default for the null model, set the interface concentration rate to zero.
  setBaseComputesAndRequiresDivergence(task, matls->getUnion());
  sched->addTask(task, patches, matls);

}

void SDInterfaceModel::setBaseComputesAndRequiresDivergence(        Task            * task
                                                           ,  const MaterialSubset  * matls )
{
  task->computes(sdInterfaceRate, matls);
  task->computes(sdInterfaceRate, d_materialManager->getAllInOneMatls(),
                 Task::OutOfDomain);

  task->computes(sdInterfaceFlag, matls);
  task->computes(sdInterfaceFlag, d_materialManager->getAllInOneMatls(),
                 Task::OutOfDomain);
}

void SDInterfaceModel::sdInterfaceDivergence( const ProcessorGroup  *
                                            , const PatchSubset     * patches
                                            , const MaterialSubset  * matls
                                            ,       DataWarehouse   * old_dw
                                            ,       DataWarehouse   * new_dw  )
{
  // Set the interfacial flux rate to zero for the (default) null model.
  for (int patchIdx = 0; patchIdx < patches->size(); ++patchIdx) {
    const Patch*  patch     = patches->get(patchIdx);
    int           numMatls  = matls->size();

    NCVariable<double>  gdCdt_interface_Total;
    NCVariable<int>     gInterfaceFlag_Total;
    // Initialize global references to interface flux and interface presence
    new_dw->allocateAndPut(gdCdt_interface_Total, sdInterfaceRate,
                           d_materialManager->getAllInOneMatls()->get(0), patch);
    gdCdt_interface_Total.initialize(0.0);
    new_dw->allocateAndPut(gInterfaceFlag_Total, sdInterfaceFlag,
                           d_materialManager->getAllInOneMatls()->get(0), patch);
    gInterfaceFlag_Total.initialize(false);

    std::vector<NCVariable<double> >  gdCdt_interface(numMatls);
    std::vector<NCVariable<int>    >  gInterfaceFlag(numMatls);
    for (int matlIdx = 0; matlIdx < numMatls; ++matlIdx) {
      int dwi = matls->get(matlIdx);
      new_dw->allocateAndPut(gdCdt_interface[matlIdx],  sdInterfaceRate,  dwi,
                             patch);
      gdCdt_interface[matlIdx].initialize(0.0);
      new_dw->allocateAndPut(gInterfaceFlag[matlIdx],   sdInterfaceFlag,  dwi,
                             patch);
      gInterfaceFlag[matlIdx].initialize(false);
    }
  }
}

void SDInterfaceModel::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP sdim_ps = ps;
  sdim_ps = ps->appendChild("diffusion_interface");
  sdim_ps->appendElement("type","null");
}

const VarLabel* SDInterfaceModel::getInterfaceFluxLabel() const {
  return sdInterfaceRate;
}

const VarLabel* SDInterfaceModel::getInterfaceFlagLabel() const {
  return sdInterfaceFlag;
}

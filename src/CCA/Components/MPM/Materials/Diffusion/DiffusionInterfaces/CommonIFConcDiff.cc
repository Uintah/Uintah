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

#include <CCA/Components/MPM/Materials/Diffusion/DiffusionInterfaces/CommonIFConcDiff.h>
#include <CCA/Components/MPM/Core/MPMDiffusionLabel.h>

#include <Core/Grid/Task.h>

using namespace Uintah;
using namespace std;

CommonIFConcDiff::CommonIFConcDiff(ProblemSpecP& ps, MaterialManagerP& sS,
                                   MPMFlags* mpm_flags, MPMLabel* mpm_lb)
                 : SDInterfaceModel(ps, sS, mpm_flags, mpm_lb)
{
//  ps->get("materials", d_materials_list);
}

CommonIFConcDiff::~CommonIFConcDiff()
{
}

void CommonIFConcDiff::addComputesAndRequiresInterpolated(SchedulerP & sched,
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls)
{
  Task* task = scinew Task("CommonIFConcDiff::sdInterfaceInterpolated", this,
                           &CommonIFConcDiff::sdInterfaceInterpolated);

  const MaterialSubset* mss = matls->getUnion();
  task->requires( Task::NewDW, d_mpm_lb->gMassLabel, Ghost::AroundNodes, 1);

  task->modifies(d_mpm_lb->diffusion->gConcentration, mss);

  sched->addTask(task, patches, matls);
}

void CommonIFConcDiff::sdInterfaceInterpolated(const ProcessorGroup *,
                                               const PatchSubset    * patches,
                                               const MaterialSubset * matls,
                                                     DataWarehouse  * old_dw,
                                                     DataWarehouse  * new_dw
                                              )
{
  int num_matls = d_materialManager->getNumMatls( "MPM" );
  for(int p = 0; p < patches->size(); p++)
  {
    const Patch* patch = patches->get(p);

    std::vector<constNCVariable<double> > gmass(num_matls);
    std::vector<NCVariable<double> >      gconcentration(num_matls);
    for(int m = 0; m < num_matls; m++)
    {
      int dwi = matls->get(m);
      new_dw->get(gmass[m], d_mpm_lb->gMassLabel, dwi, patch,
                  Ghost::AroundNodes, 1);
      new_dw->getModifiable(gconcentration[m], d_mpm_lb->diffusion->gConcentration,
                            dwi, patch);
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++)
    {
      IntVector c = *iter;

      double g_sum_mass = 0;
      double g_sum_concmass = 0;

      for(int m = 0; m < num_matls; m++){
        g_sum_mass += gmass[m][c];
        g_sum_concmass += gconcentration[m][c] * gmass[m][c];
      }

      double g_conc = g_sum_concmass / g_sum_mass;


//      for(unsigned int m = 0; m < d_materials_list.size(); m++)
      for (int m = 0; m < num_matls; ++m)
      {
        if (d_materials_list.requested(m))
        {
          gconcentration[m][c] = g_conc;
        }
//        int mat_idx = d_materials_list[m];
//        if(mat_idx >= 0 && mat_idx < num_matls)
////          gconcentration[mat_idx][c] = g_conc;
      }
    }
  }
}

void CommonIFConcDiff::addComputesAndRequiresDivergence(SchedulerP & sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls)
{
  Task* task = scinew Task("CommonIFConcDiff::sdInterfaceDivergence", this,
                           &CommonIFConcDiff::sdInterfaceDivergence);

  const MaterialSubset* mss = matls->getUnion();
  task->requires( Task::NewDW, d_mpm_lb->gMassLabel, Ghost::None);

  task->modifies(d_mpm_lb->diffusion->gConcentrationRate, mss);

  sched->addTask(task, patches, matls);
}

void CommonIFConcDiff::sdInterfaceDivergence(const ProcessorGroup*,
                                             const PatchSubset* patches,
                                             const MaterialSubset* matls,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
  int num_matls = d_materialManager->getNumMatls( "MPM" );
    for(int p = 0; p < patches->size(); p++){
      const Patch* patch = patches->get(p);

      std::vector<constNCVariable<double> > gmass(num_matls);
      std::vector<NCVariable<double> > gconc_rate(num_matls);
      for(int m = 0; m < num_matls; m++){
        int dwi = matls->get(m);
        new_dw->get(gmass[m], d_mpm_lb->gMassLabel, dwi, patch, Ghost::None, 0);
        new_dw->getModifiable(gconc_rate[m], d_mpm_lb->diffusion->gConcentrationRate, dwi, patch);
      }

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
        IntVector c = *iter;

        double g_sum_mass = 0;
        double g_sum_conc_rate_mass = 0;

        for(int m = 0; m < num_matls; m++){
          g_sum_mass += gmass[m][c];
          g_sum_conc_rate_mass += gconc_rate[m][c] * gmass[m][c];
        }

        double g_conc_rate = g_sum_conc_rate_mass / g_sum_mass;

        for(int m = 0; m < num_matls; m++){
          gconc_rate[m][c] = g_conc_rate;
        }
      }
    }
}

void CommonIFConcDiff::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP sdim_ps = ps;
  sdim_ps = ps->appendChild("diffusion_interface");
  sdim_ps->appendElement("type","common");
  d_materials_list.outputProblemSpec(sdim_ps);
}

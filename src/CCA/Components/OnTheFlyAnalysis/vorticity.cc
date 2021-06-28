/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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

#include <CCA/Components/OnTheFlyAnalysis/vorticity.h>

#include <CCA/Components/ICE/Core/ICELabel.h>
#include <CCA/Components/ICE/Materials/ICEMaterial.h>


#include <CCA/Ports/Scheduler.h>

#include <Core/Grid/DbgOutput.h>
#include <Core/Util/DebugStream.h>

#include <sci_defs/visit_defs.h>

#include <sys/stat.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <cstdio>


using namespace Uintah;
using namespace std;
//__________________________________
//  To turn on the output
//  setenv SCI_DEBUG "VORTICITY_DBG_COUT:+"

static DebugStream cout_doing("VORTICITY_DOING_COUT", false);
static DebugStream cout_dbg("VORTICITY_DBG_COUT", false);

Dout dout_OTF_VTY("vorticity",     "OnTheFlyAnalysis", "Task scheduling and execution.", false);

//______________________________________________________________________
vorticity::vorticity(const ProcessorGroup* myworld,
                     const MaterialManagerP materialManager,
                     const ProblemSpecP& module_spec)
  : AnalysisModule(myworld, materialManager, module_spec)
{
  I_lb  = scinew ICELabel();

  vorticityLabel = VarLabel::create("vorticity", CCVariable<Vector>::getTypeDescription());

  required = false;
}

//______________________________________________________________________
//
vorticity::~vorticity()
{
  DOUTR(dout_OTF_VTY, "Doing destructor vorticity");

  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }

  VarLabel::destroy(vorticityLabel);
  delete I_lb;
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void vorticity::problemSetup(const ProblemSpecP& ,
                             const ProblemSpecP& ,
                             GridP& grid,
                             std::vector<std::vector<const VarLabel* > > &PState,
                             std::vector<std::vector<const VarLabel* > > &PState_preReloc)
{
  cout_doing << "Doing problemSetup \t\t\t\tvorticity" << endl;

  DOUTR(dout_OTF_VTY, "Doing vorticity::problemSetup");

  // determine which material index to compute
  if(m_module_spec->findBlock("material") ){
    d_matl = m_materialManager->parseAndLookupMaterial(m_module_spec, "material");
  }
  else {
    throw ProblemSetupException("ERROR:AnalysisModule:vorticity: Missing <material> tag. \n", __FILE__, __LINE__);
  }

  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();
  d_matl_sub = d_matl_set->getUnion();

#ifdef HAVE_VISIT
  static bool initialized = false;

  if( m_application->getVisIt() && !initialized ) {
    required = true;

    initialized = true;
  }
#endif
}

//______________________________________________________________________
void vorticity::scheduleDoAnalysis(SchedulerP  & sched,
                                   const LevelP& level)
{
  printSchedule( level, dout_OTF_VTY,"vorticity::scheduleDoAnalysis" );

  Task* t = scinew Task("vorticity::doAnalysis",
                   this,&vorticity::doAnalysis);

  Ghost::GhostType gac = Ghost::AroundCells;

  t->requires( Task::NewDW, I_lb->vel_CCLabel, d_matl_sub, gac,1);
  t->computes( vorticityLabel, d_matl_sub);

#ifdef HAVE_VISIT
  if( required ) {
    t->requires( Task::OldDW, vorticityLabel, d_matl_sub, m_gn, 0);
  }
#endif

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
// Compute the vorticity field.
void vorticity::doAnalysis(const ProcessorGroup * pg,
                           const PatchSubset    * patches,
                           const MaterialSubset * matl_sub ,
                           DataWarehouse        * old_dw,
                           DataWarehouse        * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, dout_OTF_VTY,"Doing vorticity::doAnalysis");

    Ghost::GhostType gac = Ghost::AroundCells;

    CCVariable<Vector> vorticity;
    constCCVariable<Vector> vel_CC;

    int indx = d_matl->getDWIndex();
    new_dw->get(vel_CC,               I_lb->vel_CCLabel, indx,patch,gac, 1);
    new_dw->allocateAndPut( vorticity, vorticityLabel,   indx,patch);

    vorticity.initialize(Vector(0.0));

    //__________________________________
    // cell spacing
    Vector dx = patch->dCell();
    double delX = dx.x();
    double delY = dx.y();
    double delZ = dx.z();

    for (CellIterator iter=patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;

      IntVector r   = c + IntVector(1,0,0);   // right
      IntVector l   = c - IntVector(1,0,0);   // left

      IntVector t   = c + IntVector(0,1,0);   // top
      IntVector b   = c - IntVector(0,1,0);   // bottom

      IntVector frt = c + IntVector(0,0,1);   // front
      IntVector bck = c - IntVector(0,0,1);   // back

       // second-order central difference
      double du_dy = (vel_CC[ t ].x() - vel_CC[ b ].x())/(2.0 * delY);
      double du_dz = (vel_CC[frt].x() - vel_CC[bck].x())/(2.0 * delZ);

      double dv_dx = (vel_CC[ r ].y() - vel_CC[ l ].y())/(2.0 * delX);
      double dv_dz = (vel_CC[frt].y() - vel_CC[bck].y())/(2.0 * delZ);

      double dw_dx = (vel_CC[ r ].z() - vel_CC[ l ].z())/(2.0 * delX);
      double dw_dy = (vel_CC[ t ].z() - vel_CC[ b ].z())/(2.0 * delY);

      double omega_x = dw_dy - dv_dz;
      double omega_y = du_dz - dw_dx;
      double omega_z = dv_dx - du_dy;

      vorticity[c] = Vector(omega_x,omega_y,omega_z);
    }
  }  // patches
}

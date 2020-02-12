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

#include <CCA/Components/OnTheFlyAnalysis/vorticity.h>

#include <CCA/Components/ICE/Core/ICELabel.h>
#include <CCA/Components/ICE/Materials/ICEMaterial.h>

#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Parallel/ProcessorGroup.h>
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
//______________________________________________________________________
vorticity::vorticity(const ProcessorGroup* myworld,
                     const MaterialManagerP materialManager,
                     const ProblemSpecP& module_spec)
  : AnalysisModule(myworld, materialManager, module_spec)
{
  d_matl_set = 0;
  v_lb = scinew vorticityLabel();
  I_lb  = scinew ICELabel();

  required = false;
}

//__________________________________
vorticity::~vorticity()
{
  cout_doing << " Doing: destorying vorticity " << endl;
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
  VarLabel::destroy(v_lb->vorticityLabel);
  delete v_lb;
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
  
  v_lb->vorticityLabel = VarLabel::create("vorticity", CCVariable<Vector>::getTypeDescription());
  
  // determine which material index to compute
  d_matl = m_materialManager->parseAndLookupMaterial(m_module_spec, "material");
  
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
  cout_doing << "vorticity::scheduleDoAnalysis " << endl;
  Task* t = scinew Task("vorticity::doAnalysis", 
                   this,&vorticity::doAnalysis);
  
  Ghost::GhostType gac = Ghost::AroundCells;
  
  t->requires(Task::NewDW, I_lb->vel_CCLabel, d_matl_sub, gac,1);
  t->computes(v_lb->vorticityLabel, d_matl_sub);

#ifdef HAVE_VISIT
  if( required )
  {
    Ghost::GhostType  gn = Ghost::None;
    t->requires(Task::OldDW, v_lb->vorticityLabel, d_matl_sub, gn, 0);
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
  const Level* level = getLevel(patches);
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    cout_doing << pg->myRank() << " " 
               << "Doing doAnalysis (vorticity)\t\t\t\tL-"
               << level->getIndex()
               << " patch " << patch->getGridIndex()<< endl;

    Ghost::GhostType gac = Ghost::AroundCells;
    
    CCVariable<Vector> vorticity;
    constCCVariable<Vector> vel_CC;
    
    int indx = d_matl->getDWIndex();
    new_dw->get(vel_CC,               I_lb->vel_CCLabel,    indx,patch,gac, 1);
    new_dw->allocateAndPut(vorticity, v_lb->vorticityLabel, indx,patch);
  
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

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

#include <CCA/Components/MPM/MPMCommon.h> 
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/DamageModel.h>
#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/ErosionModel.h>
#include <CCA/Components/MPM/CohesiveZone/CZMaterial.h>
#include <Core/ProblemSpec/ProblemSpec.h>
using namespace std;
using namespace Uintah;

static DebugStream cout_doing("MPM", false);

MPMCommon::MPMCommon(const ProcessorGroup* myworld)
  : d_myworld(myworld)
{
}

MPMCommon::~MPMCommon()
{
}
//______________________________________________________________________
//
void MPMCommon::materialProblemSetup(const ProblemSpecP& prob_spec, 
                                     SimulationStateP& sharedState,
                                     MPMFlags* flags, 
                                     bool isRestart)
{
  d_sharedState = sharedState;
  d_flags = flags;
  
  //Search for the MaterialProperties block and then get the MPM section
  ProblemSpecP mat_ps     = prob_spec->findBlockWithOutAttribute( "MaterialProperties" );
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock( "MPM" );
  for( ProblemSpecP ps = mpm_mat_ps->findBlock( "material" ); ps != nullptr; ps = ps->findNextBlock( "material" ) ) {
    string index( "" );
    ps->getAttribute( "index",index );
    stringstream id(index);
    const int DEFAULT_VALUE = -1;
    int index_val = DEFAULT_VALUE;

    id >> index_val;

    if( !id ) {
      // stringstream parsing failed... on many (most) systems, the
      // original value assigned to index_val would be left
      // intact... but on some systems it inserts garbage,
      // so we have to manually restore the value.
      index_val = DEFAULT_VALUE;
    }
    // cout << "Material attribute = " << index_val << ", " << index << ", " << id << "\n";

    //Create and register as an MPM material
    MPMMaterial *mat = scinew MPMMaterial(ps, sharedState, flags,isRestart);
    // When doing restart, we need to make sure that we load the materials
    // in the same order that they were initially created.  Restarts will
    // ALWAYS have an index number as in <material index = "0">.
    // Index_val = -1 means that we don't register the material by its 
    // index number.
    if (index_val > -1){
      sharedState->registerMPMMaterial(mat,index_val);
    }
    else{
      sharedState->registerMPMMaterial(mat);
    }
  }
}
//______________________________________________________________________
//
void MPMCommon::cohesiveZoneProblemSetup(const ProblemSpecP& prob_spec, 
                                         SimulationStateP& sharedState,
                                         MPMFlags* flags)
{
  //Search for the MaterialProperties block and then get the MPM section
  ProblemSpecP mat_ps     = prob_spec->findBlockWithOutAttribute( "MaterialProperties" );
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock( "MPM" );
  for( ProblemSpecP ps = mpm_mat_ps->findBlock("cohesive_zone"); ps != nullptr; ps = ps->findNextBlock("cohesive_zone") ) {

    string index("");
    ps->getAttribute("index",index);
    stringstream id(index);
    const int DEFAULT_VALUE = -1;
    int index_val = DEFAULT_VALUE;

    id >> index_val;

    if( !id ) {
      // stringstream parsing failed... on many (most) systems, the
      // original value assigned to index_val would be left
      // intact... but on some systems it inserts garbage,
      // so we have to manually restore the value.
      index_val = DEFAULT_VALUE;
    }
    // cout << "Material attribute = " << index_val << ", " << index << ", " << id << "\n";

    //Create and register as an MPM material
    CZMaterial *mat = scinew CZMaterial(ps, sharedState, flags);

    // When doing restart, we need to make sure that we load the materials
    // in the same order that they were initially created.  Restarts will
    // ALWAYS have an index number as in <material index = "0">.
    // Index_val = -1 means that we don't register the material by its 
    // index number.
    if (index_val > -1){
      sharedState->registerCZMaterial(mat,index_val);
    }
    else{
      sharedState->registerCZMaterial(mat);
    }
  }
}
//______________________________________________________________________
//
void MPMCommon::scheduleUpdateStress_DamageErosionModels(SchedulerP   & sched,
                                                     const PatchSet * patches,
                                                     const MaterialSet * matls )
{
  printSchedule(patches,cout_doing,"MPMCommon::scheduleUpdateStress_DamageErosionModels");
  
  Task* t = scinew Task("MPM::updateStress_DamageErosionModels", this, 
                        &MPMCommon::updateStress_DamageErosionModels);
                        
  int numMatls = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    
    DamageModel* dm = mpm_matl->getDamageModel();
    dm->addComputesAndRequires(t, mpm_matl);
    
    ErosionModel* em = mpm_matl->getErosionModel();
    em->addComputesAndRequires(t, mpm_matl);
  }
  
  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
//
void MPMCommon::updateStress_DamageErosionModels(const ProcessorGroup *,
                                                 const PatchSubset    * patches,
                                                 const MaterialSubset * ,
                                                 DataWarehouse        * old_dw,
                                                 DataWarehouse        * new_dw)
{
  for (int p = 0; p<patches->size(); p++) {
    const Patch* patch = patches->get(p);
   
    printTask(patches, patch,cout_doing,
              "Doing updateStress_DamageModel");

    int numMPMMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMPMMatls; m++){
    
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      
      DamageModel* dm = mpm_matl->getDamageModel();
      dm->computeSomething( pset, mpm_matl, patch, old_dw, new_dw );
      
      ErosionModel* em = mpm_matl->getErosionModel();
      em->updateStress_Erosion( pset, old_dw, new_dw );
    }
  }
}

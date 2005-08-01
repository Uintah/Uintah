#include <Packages/Uintah/CCA/Components/MPM/HeatConduction/ImplicitHeatConduction.h>
#include <Packages/Uintah/Core/Math/Short27.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/TemperatureBoundCond.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMBoundCond.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/CCA/Components/MPM/PetscSolver.h>
#include <Packages/Uintah/CCA/Components/MPM/SimpleSolver.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Grid/LinearInterpolator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Util/DebugStream.h>
#include <Core/Containers/StaticArray.h>

using namespace Uintah;
using namespace SCIRun;

#define EROSION
#undef EROSION

static DebugStream cout_doing("ImplicitHeatConduction", false);
static DebugStream cout_heat("MPMHeat", false);

ImplicitHeatConduction::ImplicitHeatConduction(SimulationStateP& sS,
                                               MPMLabel* labels,MPMFlags* flags)
{
  lb = labels;
  d_flag = flags;
  d_sharedState = sS;
  d_perproc_patches=0;
  do_IHC=d_flag->d_doImplicitHeatConduction;
  d_HC_transient=d_flag->d_doTransientImplicitHeatConduction;

  if(d_flag->d_8or27==8){
    NGP=1;
    NGN=1;
  } else if(d_flag->d_8or27==27){
    NGP=2;
    NGN=2;
  }
}

ImplicitHeatConduction::~ImplicitHeatConduction()
{
 if(do_IHC){
  int numMatls = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMatls; m++){
    delete d_HC_solver[m];
  }

  if(d_perproc_patches && d_perproc_patches->removeReference()) {
    delete d_perproc_patches;
    cout << "Freeing patches!!\n";
  }
 }
}

void ImplicitHeatConduction::problemSetup()
{
   int numMatls = d_sharedState->getNumMPMMatls();

#ifdef HAVE_PETSC
   d_HC_solver = vector<MPMPetscSolver*>(numMatls);
   for(int m=0;m<numMatls;m++){
      d_HC_solver[m]=scinew MPMPetscSolver();
      d_HC_solver[m]->initialize();
   }
#else
   d_HC_solver = vector<SimpleSolver*>(numMatls);
   for(int m=0;m<numMatls;m++){
      d_HC_solver[m]=scinew SimpleSolver();
      d_HC_solver[m]->initialize();
   }
#endif
}

void ImplicitHeatConduction::scheduleDestroyHCMatrix(SchedulerP& sched,
                                                     const PatchSet* patches,
                                                     const MaterialSet* matls)
{
 if(do_IHC){
   Task* t = scinew Task("ImpMPM::destroyHCMatrix",this,
                         &ImplicitHeatConduction::destroyHCMatrix);                                                                                
   sched->addTask(t, patches, matls);
 }
}

void ImplicitHeatConduction::scheduleCreateHCMatrix(SchedulerP& sched,
                                                    const PatchSet* patches,
                                                    const MaterialSet* matls)
{
 if(do_IHC){
  Task* t = scinew Task("ImpMPM::createHCMatrix",this,
                        &ImplicitHeatConduction::createHCMatrix);
                                                                                
  t->requires(Task::OldDW, lb->pXLabel,Ghost::AroundNodes,1);

  if (!d_perproc_patches) {
    d_perproc_patches=patches;
    d_perproc_patches->addReference();
  }

  sched->addTask(t, patches, matls);
 }
}

void ImplicitHeatConduction::scheduleApplyHCBoundaryConditions(SchedulerP& schd,
                                               const PatchSet* patches,
                                               const MaterialSet* matls)
{
 if(do_IHC){
  Task* t = scinew Task("ImpMPM::applyHCBoundaryCondition", this,
                        &ImplicitHeatConduction::applyHCBoundaryConditions);
                                                                                
  t->computes(lb->gTemperatureStarLabel);
                                                                                
  schd->addTask(t, patches, matls);
 }
}

void ImplicitHeatConduction::scheduleFindFixedHCDOF(SchedulerP& sched,
                                                    const PatchSet* patches,
                                                    const MaterialSet* matls)
{
 if(do_IHC){
  Task* t = scinew Task("ImpMPM::findFixedHCDOF", this,
                        &ImplicitHeatConduction::findFixedHCDOF);
                                                                                
  t->requires(Task::NewDW, lb->gMassLabel, Ghost::None, 0);
                                                                                
  sched->addTask(t, patches, matls);
 }
}

void ImplicitHeatConduction::scheduleFormHCStiffnessMatrix(SchedulerP& sched,
                                                     const PatchSet* patches,
                                                     const MaterialSet* matls)
{
 if(do_IHC){
  Task* t = scinew Task("ImpMPM::formHCStiffnessMatrix",this,
                        &ImplicitHeatConduction::formHCStiffnessMatrix);
                                                                                
  t->requires(Task::NewDW,lb->gMassLabel, Ghost::None);
  t->requires(Task::OldDW,d_sharedState->get_delt_label());
                                                                                
  sched->addTask(t, patches, matls);
 }
}

void ImplicitHeatConduction::scheduleFormHCQ(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
 if(do_IHC){
  Task* t = scinew Task("ImpMPM::formHCQ", this,
                        &ImplicitHeatConduction::formHCQ);
                                                                                
  Ghost::GhostType  gnone = Ghost::None;
                                                                                
  t->requires(Task::OldDW,      d_sharedState->get_delt_label());
  t->requires(Task::NewDW,lb->gTemperatureLabel,      gnone,0);
  t->requires(Task::NewDW,lb->gMassLabel,             gnone,0);
  t->requires(Task::NewDW,lb->gExternalHeatRateLabel, gnone,0);
                                                                                
  sched->addTask(t, patches, matls);
 }
}

void ImplicitHeatConduction::scheduleAdjustHCQAndHCKForBCs(SchedulerP& sched,
                                                     const PatchSet* patches,
                                                     const MaterialSet* matls)
{
 if(do_IHC){
  Task* t = scinew Task("ImpMPM::adjustHCQAndHCKForBCs", this,
                        &ImplicitHeatConduction::adjustHCQAndHCKForBCs);

  Ghost::GhostType  gnone = Ghost::None;

  t->requires(Task::NewDW, lb->gTemperatureStarLabel,  gnone,0);

  sched->addTask(t, patches, matls);
 }
}

void ImplicitHeatConduction::scheduleSolveForTemp(SchedulerP& sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)
{
 if(do_IHC){
  Task* t = scinew Task("ImpMPM::solveForTemp", this,
                        &ImplicitHeatConduction::solveForTemp);
                                                                                
  sched->addTask(t, patches, matls);
 }
}

void ImplicitHeatConduction::scheduleGetTemperatureIncrement(SchedulerP& sched,
                                                       const PatchSet* patches,
                                                       const MaterialSet* matls)
{
 if(do_IHC){
  Task* t = scinew Task("ImpMPM::getTemperatureIncrement", this,
                        &ImplicitHeatConduction::getTemperatureIncrement);
                                                                                
  t->requires(Task::NewDW, lb->gTemperatureNoBCLabel, Ghost::None);
  t->requires(Task::NewDW, lb->gMassLabel,            Ghost::None);
  t->modifies(lb->gTemperatureStarLabel);
  t->computes(lb->gTemperatureRateLabel);
                                                                                
  sched->addTask(t, patches, matls);
 }
 else{
  Task* t = scinew Task("ImpMPM::fillgTemperatureRate", this,
                        &ImplicitHeatConduction::fillgTemperatureRate);

  t->computes(lb->gTemperatureRateLabel);

  sched->addTask(t, patches, matls);
 }
}

void ImplicitHeatConduction::destroyHCMatrix(const ProcessorGroup*,
                                             const PatchSubset* patches,
                                             const MaterialSubset* ,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
  if (cout_doing.active())
    cout_doing <<"Doing destroyHCMatrix " <<"\t\t\t\t\t IMPM" << "\n" << "\n";

  for (int m = 0; m < d_sharedState->getNumMPMMatls(); m++ ) {
    d_HC_solver[m]->destroyMatrix(false);
  }
}

void ImplicitHeatConduction::createHCMatrix(const ProcessorGroup* pg,
                                            const PatchSubset* patches,
                                            const MaterialSubset* ,
                                            DataWarehouse* old_dw,
                                            DataWarehouse* new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  for (int m = 0; m < numMatls; m++){
    map<int,int> dof_diag;
    for(int pp=0;pp<patches->size();pp++){
      const Patch* patch = patches->get(pp);
      if (cout_doing.active()) {
        cout_doing <<"Doing createHCMatrix on patch " << patch->getID()
                   << "\t\t\t\t IMPM"    << "\n" << "\n";
      }
                                                                                
      d_HC_solver[m]->createLocalToGlobalMapping(pg,d_perproc_patches,
                                                 patches,1);
                                                                                
      IntVector lowIndex = patch->getNodeLowIndex();
      IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
                                                                                
      Array3<int> l2g(lowIndex,highIndex);
      d_HC_solver[m]->copyL2G(l2g,patch);
                                                                                
      CCVariable<int> visited;
      new_dw->allocateTemporary(visited,patch,Ghost::AroundCells,1);
      visited.initialize(0);
                                                                                
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      constParticleVariable<Point> px;
      ParticleSubset* pset;
                                                                                
      pset = old_dw->getParticleSubset(dwi,patch, Ghost::AroundNodes,1,
                                                          lb->pXLabel);
      old_dw->get(px,lb->pXLabel,pset);
                                                                                
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
        IntVector cell,ni[8];
        patch->findCell(px[idx],cell);
        if (visited[cell] == 0 ) {
          visited[cell] = 1;
          patch->findNodesFromCell(cell,ni);
          vector<int> dof(0);
          int l2g_node_num;
          for (int k = 0; k < 8; k++) {
            if (patch->containsNode(ni[k]) ) {
              l2g_node_num = l2g[ni[k]] - l2g[lowIndex];
              dof.push_back(l2g_node_num);
            }
          }
          for (int I = 0; I < (int) dof.size(); I++) {
            int dofi = dof[I];
            for (int J = 0; J < (int) dof.size(); J++) {
              dof_diag[dofi] += 1;
            }
          }
        }
      }
    }
    d_HC_solver[m]->createMatrix(pg,dof_diag);
  }
}

void ImplicitHeatConduction::applyHCBoundaryConditions(const ProcessorGroup*,
                                                     const PatchSubset* patches,
                                                     const MaterialSubset* ,
                                                     DataWarehouse* old_dw,
                                                     DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing <<"Doing applyHCBoundaryConditions " <<"\t\t\t\t IMPM"
                 << "\n" << "\n";
    }
    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
                                                                                
    // Apply grid boundary conditions to the temperature before storing the data
    for (int m = 0; m < d_sharedState->getNumMPMMatls(); m++ ) {
      d_HC_solver[m]->copyL2G(l2g,patch);
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matl = mpm_matl->getDWIndex();

      NCVariable<double> gtemp;

      new_dw->allocateAndPut(gtemp, lb->gTemperatureStarLabel,matl,patch);
      gtemp.initialize(0.);
                                                                                
      for(Patch::FaceType face = Patch::startFace;
          face <= Patch::endFace; face=Patch::nextFace(face)){
        const BoundCondBase *temp_bcs;
        if (patch->getBCType(face) == Patch::None) {
          int numChildren =
            patch->getBCDataArray(face)->getNumberChildren(matl);
          for (int child = 0; child < numChildren; child++) {
            vector<IntVector> bound,nbound,sfx,sfy,sfz;
            vector<IntVector>::const_iterator boundary;
                                                                                
            temp_bcs = patch->getArrayBCValues(face,matl,"Temperature",bound,
                                               nbound,sfx,sfy,sfz,child);
            if (temp_bcs != 0) {
              const TemperatureBoundCond* bc =
                           dynamic_cast<const TemperatureBoundCond*>(temp_bcs);
              if (bc->getKind() == "Dirichlet") {
                for (boundary=nbound.begin(); boundary != nbound.end();
                     boundary++) {
                  gtemp[*boundary] = bc->getValue();
                }
                IntVector l,h;
                patch->getFaceNodes(face,0,l,h);
                for(NodeIterator it(l,h); !it.done(); it++) {
                  IntVector n = *it;
                  int dof = l2g[n];
                  d_HC_solver[m]->d_DOF.insert(dof);
                }
              }
              delete temp_bcs;
            }
          }
        } else
          continue;
      }  // faces
    }    // matls
  }      // patches
}

void ImplicitHeatConduction::findFixedHCDOF(const ProcessorGroup*,
                                            const PatchSubset* patches,
                                            const MaterialSubset* ,
                                            DataWarehouse* old_dw,
                                            DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing <<"Doing findFixedHCDOF on patch " << patch->getID()
                 <<"\t\t\t\t IMPM"<< "\n" << "\n";
    }
                                                                                
    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
                                                                                
    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      d_HC_solver[m]->copyL2G(l2g,patch);
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      constNCVariable<double> mass;
      new_dw->get(mass,   lb->gMassLabel,matlindex,patch,Ghost::None,0);
                                                                                
      for (NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
        IntVector n = *iter;
        // Just look on the grid to see if the gmass is 0 and then remove that
        if (compare(mass[n],0.)) {
          int dof = l2g[n];
          d_HC_solver[m]->d_DOF.insert(dof);
        }
      }  // node iterator
    }    // loop over matls
  }      // patches
}

void ImplicitHeatConduction::formHCStiffnessMatrix(const ProcessorGroup*,
                                                   const PatchSubset* patches,
                                                   const MaterialSubset* ,
                                                   DataWarehouse* old_dw,
                                                   DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing <<"Doing formHCStiffnessMatrix " << patch->getID()
                 <<"\t\t\t\t IMPM"<< "\n" << "\n";
    }

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);

    Vector dx = patch->dCell();

    LinearInterpolator* interpolator = new LinearInterpolator(patch);

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      d_HC_solver[m]->copyL2G(l2g,patch);
                                                                                
      constNCVariable<double> gmass;
      delt_vartype dt;
      new_dw->get(gmass,lb->gMassLabel, matlindex,patch,Ghost::None,0);
      old_dw->get(dt,d_sharedState->get_delt_label(), patch->getLevel());
                                                                                
      ParticleSubset* pset;
      pset = old_dw->getParticleSubset(matlindex, patch);
      constParticleVariable<Point> px;
      constParticleVariable<double> pvolume;
      constParticleVariable<double> ptemperature;
                                                                                
      old_dw->get(px,             lb->pXLabel,                  pset);
      old_dw->get(pvolume,        lb->pVolumeLabel,             pset);
      old_dw->get(ptemperature,   lb->pTemperatureLabel,        pset);
                                                                                
#ifdef HAVE_PETSC
      PetscScalar v[64];
#else
      double v[64];
#endif
      double kHC[8][8];
      int dof[8];
      double K  = mpm_matl->getThermalConductivity();
      double Cp = mpm_matl->getSpecificHeat();
                                                                                
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        // Get the node indices that surround the cell
        vector<IntVector> ni;
        ni.reserve(8);
        vector<Vector> d_S;
        d_S.reserve(8);
                                                                                
        interpolator->findCellAndShapeDerivatives(px[idx], ni, d_S);
                                                                                
        for(int ii = 0;ii<8;ii++){
          for(int jj = 0;jj<8;jj++){
            kHC[ii][jj]=0.;
          }
        }
                                                                                
        for(int ii = 0;ii<8;ii++){
          int l2g_node_num = l2g[ni[ii]];
          dof[ii]=l2g_node_num;
          for(int jj = 0;jj<8;jj++){
            for(int dir=0;dir<3;dir++){
              kHC[ii][jj]+=d_S[jj][dir]*d_S[ii][dir]*(K/(dx[dir]*dx[dir]))
                                                    *pvolume[idx];
            }
          }
        }
                                                                                
        for (int I = 0; I < 8;I++){
          for (int J = 0; J < 8; J++){
            v[8*I+J] = kHC[I][J];
          }
        }
        d_HC_solver[m]->fillMatrix(8,dof,8,dof,v);
      }
                                                                                
      if (d_HC_transient) {
        // Thermal inertia terms
        for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
          IntVector n = *iter;
          int dof[1];
          dof[0] = l2g[n];
          v[0] = gmass[n]*(Cp/dt);
          d_HC_solver[m]->fillMatrix(1,dof,1,dof,v);
        }  // node iterator
      }
    }    // matls
  }
  for (int m = 0; m < d_sharedState->getNumMPMMatls(); m++)
    d_HC_solver[m]->finalizeMatrix();
}

void ImplicitHeatConduction::formHCQ(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* ,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing <<"Doing formHCQ on patch " << patch->getID()
                 <<"\t\t\t\t\t IMPM"<< "\n" << "\n";
    }
                                                                                
    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
                                                                                
    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      d_HC_solver[m]->copyL2G(l2g,patch);
                                                                                
      delt_vartype dt;
      Ghost::GhostType  gnone = Ghost::None;
                                                                                
      constNCVariable<double> mass,temperature,gextheatrate;
                                                                                
      old_dw->get(dt,d_sharedState->get_delt_label(), patch->getLevel());
      new_dw->get(temperature, lb->gTemperatureLabel,      dwi,patch,gnone,0);
      new_dw->get(mass,        lb->gMassLabel,             dwi,patch,gnone,0);
      new_dw->get(gextheatrate,lb->gExternalHeatRateLabel, dwi,patch,gnone,0);
                                                                                
      double Cp = mpm_matl->getSpecificHeat();
                                                                                
      for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {        IntVector n = *iter;
        int dof[1];
        dof[0] = l2g[n];
        double v = 0.;
        if (d_HC_transient) {
          // Thermal inertia term
          v = temperature[n]*mass[n]*Cp/dt + gextheatrate[n];
        }
        d_HC_solver[m]->fillVector(dof[0],v);
      }
      d_HC_solver[m]->assembleVector();
    }  // matls
  }    // patches
}

void ImplicitHeatConduction::adjustHCQAndHCKForBCs(const ProcessorGroup*,
                                                   const PatchSubset* patches,
                                                   const MaterialSubset* ,
                                                   DataWarehouse* old_dw,
                                                   DataWarehouse* new_dw)
{
  int num_nodes = 0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing <<"Doing adjustHCQAndHCKForBCs on patch " << patch->getID()
                 <<"\t\t\t\t\t IMPM"<< "\n" << "\n";
    }
    IntVector nodes = patch->getNNodes();
    num_nodes += (nodes.x())*(nodes.y())*(nodes.z());

    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      d_HC_solver[m]->copyL2G(l2g,patch);

      Ghost::GhostType  gnone = Ghost::None;
                                                                                
      constNCVariable<double> temperature;
                                                                                
      new_dw->get(temperature, lb->gTemperatureStarLabel,  dwi,patch,gnone,0);

      for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {        IntVector n = *iter;
        int dof[1];
        dof[0] = l2g[n];
        double v = -temperature[n];
        d_HC_solver[m]->fillTemporaryVector(dof[0],v);
      }
      d_HC_solver[m]->assembleTemporaryVector();

      d_HC_solver[m]->removeFixedDOF(num_nodes);

      d_HC_solver[m]->applyBCSToRHS();
    }  // matls
  }    // patches
}

void ImplicitHeatConduction::solveForTemp(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* ,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  int num_nodes = 0;
  for(int p = 0; p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing <<"Doing solveForTemp on patch " << patch->getID()
                 <<"\t\t\t\t IMPM"<< "\n" << "\n";
    }
                                                                                
    IntVector nodes = patch->getNNodes();
    num_nodes += (nodes.x())*(nodes.y())*(nodes.z());
  }
                                                                                
  int numMatls = d_sharedState->getNumMPMMatls();
  for(int m = 0; m < numMatls; m++){
    d_HC_solver[m]->removeFixedDOF(num_nodes);
    d_HC_solver[m]->solve();
  }
}

void ImplicitHeatConduction::getTemperatureIncrement(const ProcessorGroup*,
                                                     const PatchSubset* patches,
                                                     const MaterialSubset* ,
                                                     DataWarehouse* old_dw,
                                                     DataWarehouse* new_dw)
{
  for(int p = 0; p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing <<"Doing getTemperatureIncrement on patch " << patch->getID()
                 <<"\t\t\t\t IMPM"<< "\n" << "\n";
    }

    Ghost::GhostType  gn = Ghost::None;
    delt_vartype dt;
    old_dw->get(dt, d_sharedState->get_delt_label(),patch->getLevel());
                                                                                
    IntVector lowIndex = patch->getNodeLowIndex();
    IntVector highIndex = patch->getNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
                                                                                
    int numMatls = d_sharedState->getNumMPMMatls();

    StaticArray<NCVariable<double> > TempImp(numMatls),tempRate(numMatls);
    StaticArray<constNCVariable<double> >temp(numMatls),gmass(numMatls);
    vector<double> Cp(numMatls);

    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      new_dw->getModifiable(TempImp[m],  lb->gTemperatureStarLabel,dwi,patch);
      new_dw->allocateAndPut(tempRate[m],lb->gTemperatureRateLabel,dwi,patch);
      new_dw->get(temp[m],               lb->gTemperatureLabel, dwi,patch,gn,0);
      new_dw->get(gmass[m],              lb->gMassLabel,        dwi,patch,gn,0);
      Cp[m]=mpm_matl->getSpecificHeat();
      tempRate[m].initialize(0.0);
      TempImp[m].initialize(0.0);

      d_HC_solver[m]->copyL2G(l2g,patch);

      vector<double> x;
      int begin = d_HC_solver[m]->getSolution(x);

      for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
        IntVector n = *iter;
        int dof = l2g[n] - begin;
        TempImp[m][n] = x[dof];
      }
    }

    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      double numerator=0.0;
      double denominator=0.0;
      IntVector c = *iter;
      for(int m = 0; m < numMatls; m++) {
        numerator   += (TempImp[m][c] * gmass[m][c]  * Cp[m]);
        denominator += (gmass[m][c]   * Cp[m]);
      }

      double contactTemperature = numerator/denominator;

      for(int m = 0; m < numMatls; m++) {
        TempImp[m][c]=contactTemperature;
      }
    }

      // Set BCs on final temperature (for now, we should be able to can this)
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Temperature",TempImp[m],8);
      for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
        IntVector n = *iter;
        tempRate[m][n] = (TempImp[m][n]-temp[m][n])/dt;
      }
    } // matls
  }   // patches
}

void ImplicitHeatConduction::fillgTemperatureRate(const ProcessorGroup*,
                                                  const PatchSubset* patches,
                                                  const MaterialSubset* ,
                                                  DataWarehouse* old_dw,
                                                  DataWarehouse* new_dw)
{
  for(int p = 0; p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing <<"Doing fillgTemperatureRate on patch " << patch->getID()
                 <<"\t\t\t\t IMPM"<< "\n" << "\n";
    }
    int numMatls = d_sharedState->getNumMPMMatls();
                                                                                
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
                                                                                
      NCVariable<double> tempRate;
      new_dw->allocateAndPut(tempRate,lb->gTemperatureRateLabel,dwi,patch);
      tempRate.initialize(0.0);
    }
  }
}

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


static DebugStream cout_doing("ImplicitHeatConduction", false);

ImplicitHeatConduction::ImplicitHeatConduction(SimulationStateP& sS,
                                               MPMLabel* labels,MPMFlags* flags)
{
  lb = labels;
  d_flag = flags;
  d_sharedState = sS;
  d_perproc_patches=0;
  do_IHC=d_flag->d_doImplicitHeatConduction;
  d_HC_transient=d_flag->d_doTransientImplicitHeatConduction;

  one_matl = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();

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
  if(one_matl->removeReference())
    delete one_matl;
  
  if(do_IHC){
    delete d_HC_solver;
    
    if(d_perproc_patches && d_perproc_patches->removeReference()) {
      delete d_perproc_patches;
      cout << "Freeing patches!!\n";
    }
  }
}

void ImplicitHeatConduction::problemSetup(string solver_type)
{

  if (solver_type == "petsc") {
    d_HC_solver = scinew MPMPetscSolver();
    d_HC_solver->initialize();
  }
  else {
    d_HC_solver=scinew SimpleSolver();
    d_HC_solver->initialize();
  }
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
                                                                                
  t->computes(lb->gTemperatureStarLabel,one_matl);
                                                                                
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
                                                                                
  t->requires(Task::OldDW,      d_sharedState->get_delt_label());
  t->requires(Task::NewDW,lb->gTemperatureLabel, one_matl,Ghost::AroundCells,
              1);
  t->requires(Task::NewDW,lb->gExternalHeatRateLabel, Ghost::AroundCells,1);
                                                                                
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

  t->requires(Task::NewDW, lb->gTemperatureStarLabel,one_matl,gnone,0);

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
                                                                                

  //t->requires(Task::NewDW, lb->gTemperatureLabel,one_matl,Ghost::None);
  t->modifies(lb->gTemperatureLabel,one_matl);
  t->computes(lb->gTemperatureRateLabel,one_matl);
                                                                                
  sched->addTask(t, patches, matls);
 }
 else{
  Task* t = scinew Task("ImpMPM::fillgTemperatureRate", this,
                        &ImplicitHeatConduction::fillgTemperatureRate);

  t->computes(lb->gTemperatureRateLabel,one_matl);

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

    d_HC_solver->destroyMatrix(false);

}

void ImplicitHeatConduction::createHCMatrix(const ProcessorGroup* pg,
                                            const PatchSubset* patches,
                                            const MaterialSubset* ,
                                            DataWarehouse* old_dw,
                                            DataWarehouse* new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();

  map<int,int> dof_diag;
  for(int pp=0;pp<patches->size();pp++){
    const Patch* patch = patches->get(pp);
    if (cout_doing.active()) {
      cout_doing <<"Doing createHCMatrix on patch " << patch->getID()
                 << "\t\t\t\t IMPM"    << "\n" << "\n";
    }
    
    d_HC_solver->createLocalToGlobalMapping(pg,d_perproc_patches,
                                            patches,1);
    
    IntVector lowIndex = patch->getInteriorNodeLowIndex();
    IntVector highIndex = patch->getInteriorNodeHighIndex()+IntVector(1,1,1);
    
    Array3<int> l2g(lowIndex,highIndex);
    d_HC_solver->copyL2G(l2g,patch);
    
    CCVariable<int> visited;
    new_dw->allocateTemporary(visited,patch,Ghost::AroundCells,1);
    visited.initialize(0);
    
    for (int m = 0; m < numMatls; m++){                                                                                
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
  }
  d_HC_solver->createMatrix(pg,dof_diag);
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
    IntVector lowIndex = patch->getInteriorNodeLowIndex();
    IntVector highIndex = patch->getInteriorNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
                                                                                
    // Apply grid boundary conditions to the temperature before storing the 
    // data -- all data is for material 0 -- single temperature field.
    
    d_HC_solver->copyL2G(l2g,patch);
    int matl = 0;

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
                //  cout << "Node: " << n << " dof: " << dof << endl;
                vector<int> neigh;
                // Find the neighbors for this DOF
                for (int i = -1; i<=1; i++) {
                  for (int j = -1; j<=1; j++) {
                    for (int k = -1; k<=1; k++) {
                      IntVector neighbor = n + IntVector(i,j,k);
                      if (neighbor.x() >= lowIndex.x() && 
                          neighbor.x() < highIndex.x() &&
                          neighbor.y() >= lowIndex.y() &&
                          neighbor.y() < highIndex.y() &&
                          neighbor.z() >= lowIndex.z() &&
                          neighbor.z() < highIndex.z()) {
                        int dof_neighbor = l2g[neighbor];
                        if (dof_neighbor > 0 && dof_neighbor != dof) {
#if 0
                          cout << "neighbor: " << neighbor << " dof: " 
                               << dof_neighbor << endl;
#endif
                          neigh.push_back(dof_neighbor);
                        }
                      }
                    }
                  }
                }
                d_HC_solver->d_DOFNeighbors[dof] = neigh;
                d_HC_solver->d_DOF.insert(dof);
              }
            }
            delete temp_bcs;
          }
        }
      } else
        continue;
    }  // faces
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
                                                                                
    IntVector lowIndex = patch->getInteriorNodeLowIndex();
    IntVector highIndex = patch->getInteriorNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
                                                                                
    int numMatls = d_sharedState->getNumMPMMatls();
    d_HC_solver->copyL2G(l2g,patch);
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
      constNCVariable<double> mass;
      new_dw->get(mass,   lb->gMassLabel,matlindex,patch,Ghost::None,0);
                                                                               
      for (NodeIterator iter = patch->getNodeIterator(); !iter.done();iter++){
        IntVector n = *iter;
        // Just look on the grid to see if the gmass is 0 and then remove that
        if (compare(mass[n],0.)) {
          int dof = l2g[n];
          d_HC_solver->d_DOF.insert(dof);
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

    IntVector lowIndex = patch->getInteriorNodeLowIndex();
    IntVector highIndex = patch->getInteriorNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);

    Vector dx = patch->dCell();

    LinearInterpolator* interpolator = new LinearInterpolator(patch);

    d_HC_solver->copyL2G(l2g,patch);
    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int matlindex = mpm_matl->getDWIndex();
                                                                               
      delt_vartype dt;
      old_dw->get(dt,d_sharedState->get_delt_label(), patch->getLevel());
                                                                                
      ParticleSubset* pset;
      pset = old_dw->getParticleSubset(matlindex, patch);
      constParticleVariable<Point> px;
      constParticleVariable<double> pvolume;
      constParticleVariable<double> ptemperature;
                                                                                
      old_dw->get(px,             lb->pXLabel,                  pset);
      old_dw->get(pvolume,        lb->pVolumeLabel,             pset);
      old_dw->get(ptemperature,   lb->pTemperatureLabel,        pset);
      
      double v[64];
      double kHC[8][8];
      int dof[8];
      double K  = mpm_matl->getThermalConductivity();
      double Cp = mpm_matl->getSpecificHeat();
      double rho = mpm_matl->getInitialDensity();
                                                                                
      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        // Get the node indices that surround the cell
        vector<IntVector> ni(8);
        vector<double> S(8);
        vector<Vector> d_S(8);

        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S);
                                                                               
        for(int ii = 0;ii<8;ii++){
          for(int jj = 0;jj<8;jj++){
            kHC[ii][jj]=0.;
          }
        }
                                
        for(int ii = 0;ii<8;ii++){
          int l2g_node_num = l2g[ni[ii]];
          dof[ii]=l2g_node_num;
          for(int jj = 0;jj<8;jj++){
            // Thermal inertia terms
            if (d_HC_transient) {
              kHC[ii][jj] += S[jj]*S[ii]*Cp*rho*pvolume[idx]/dt;
            }
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
        d_HC_solver->fillMatrix(8,dof,8,dof,v);
      }                                                                                
    }    // matls
    delete interpolator;

  }
  d_HC_solver->finalizeMatrix();

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
                                                                                
    IntVector lowIndex = patch->getInteriorNodeLowIndex();
    IntVector highIndex = patch->getInteriorNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);

    LinearInterpolator* interpolator = new LinearInterpolator(patch);                                                                                
    d_HC_solver->copyL2G(l2g,patch);

    constNCVariable<double> temperature;
    new_dw->get(temperature, lb->gTemperatureLabel, 0,patch,Ghost::AroundCells,
                1);

#if 0
    for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector n = *iter;
      cout << "temperature[" << n << "]= " << temperature[n] << endl;
    }
#endif

    int numMatls = d_sharedState->getNumMPMMatls();
    for(int m = 0; m < numMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
                                                                               
      delt_vartype dt;
                                                                               
      constNCVariable<double> gextheatrate;
                                                                               
      old_dw->get(dt,d_sharedState->get_delt_label(), patch->getLevel());

      new_dw->get(gextheatrate,lb->gExternalHeatRateLabel, dwi,patch,
                  Ghost::AroundCells,1);
                                                                          
      ParticleSubset* pset;
      pset = old_dw->getParticleSubset(dwi, patch);
      constParticleVariable<Point> px;
      constParticleVariable<double> pvolume;

      old_dw->get(px,lb->pXLabel,pset);
      old_dw->get(pvolume,lb->pVolumeLabel,pset);
      
      int dof[8];
      double Cp = mpm_matl->getSpecificHeat();
      double rho = mpm_matl->getInitialDensity();
#if 0
      cout << "Cp = " << Cp << " rho = " << rho << endl;
#endif

      for(ParticleSubset::iterator iter = pset->begin();
                                   iter != pset->end(); iter++){
        particleIndex idx = *iter;
        // Get the node indices that surround the cell
        vector<IntVector> ni(8);
        vector<double> S(8);

        interpolator->findCellAndWeights(px[idx],ni,S);
                                                                               
        bool add = true;


        for(int ii = 0;ii<8;ii++){
          //  cout << "ni = " << ni[ii] << endl;
          int l2g_node_num = l2g[ni[ii]];
          double v = 0;
          dof[ii]=l2g_node_num;
          for(int jj = 0;jj<8;jj++){
            // Thermal inertia terms
            if (d_HC_transient) {
              v += S[jj]*S[ii]*Cp*rho*pvolume[idx]*temperature[ni[jj]]/dt + 
                gextheatrate[ni[ii]];
            }
          }
          // cout << "v[" << l2g_node_num << "]= " << v << endl;
          d_HC_solver->fillVector(dof[ii],v,add);
        }
      }
    }  // matls
    delete interpolator;
  }    // patches
  d_HC_solver->assembleVector();
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
    IntVector nodes = patch->getNInteriorNodes();
    num_nodes += (nodes.x())*(nodes.y())*(nodes.z());

    IntVector lowIndex = patch->getInteriorNodeLowIndex();
    IntVector highIndex = patch->getInteriorNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);

    d_HC_solver->copyL2G(l2g,patch);
    int dwi = 0;

    Ghost::GhostType  gnone = Ghost::None;
    
    constNCVariable<double> temperature;
    
    // gTemperatureStar are the boundary condition values for the temperature
    // determined earlier in applyHCBoundaryConditions()
    
    new_dw->get(temperature, lb->gTemperatureStarLabel,dwi,patch,gnone,0);
    
    for (NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
      IntVector n = *iter;
      int dof[1];
      dof[0] = l2g[n];
      double v = -temperature[n];
      d_HC_solver->fillTemporaryVector(dof[0],v);
    }
  }    // patches
  d_HC_solver->assembleTemporaryVector();
  
  d_HC_solver->applyBCSToRHS();
  
  d_HC_solver->removeFixedDOFHeat(num_nodes);
      

}

void ImplicitHeatConduction::solveForTemp(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* ,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  for(int p = 0; p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    if (cout_doing.active()) {
      cout_doing <<"Doing solveForTemp on patch " << patch->getID()
                 <<"\t\t\t\t IMPM"<< "\n" << "\n";
    }
  }
  d_HC_solver->solve();


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

    delt_vartype dt;
    old_dw->get(dt, d_sharedState->get_delt_label(),patch->getLevel());
                                                                                
    IntVector lowIndex = patch->getInteriorNodeLowIndex();
    IntVector highIndex = patch->getInteriorNodeHighIndex()+IntVector(1,1,1);
    Array3<int> l2g(lowIndex,highIndex);
                                                                                
    NCVariable<double>  tempRate;
    NCVariable<double> temp;

    int dwi = 0;

    new_dw->allocateAndPut(tempRate,lb->gTemperatureRateLabel,dwi,patch);
    //new_dw->get(temp,lb->gTemperatureLabel, dwi,patch,Ghost::None,0);
    new_dw->getModifiable(temp,lb->gTemperatureLabel, dwi,patch);


    tempRate.initialize(0.0);

    d_HC_solver->copyL2G(l2g,patch);

    vector<double> x;
    int begin = d_HC_solver->getSolution(x);

    for (NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
      IntVector n = *iter;
      int dof = l2g[n] - begin;
      tempRate[n] = (x[dof] - temp[n])/dt;
      temp[n] = x[dof];
#if 0
      cout << "tempRate[" << n << "]= " << tempRate[n] << endl;
#endif
    }
  }

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

    int dwi = 0;
    NCVariable<double> tempRate;
    new_dw->allocateAndPut(tempRate,lb->gTemperatureRateLabel,dwi,patch);
    tempRate.initialize(0.0);

  }
}

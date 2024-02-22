/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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
#include <CCA/Components/MPM/Tracer/TracerTasks.h>
#include <CCA/Components/MPM/Tracer/TracerMaterial.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Core/TracerLabel.h>
#include <CCA/Components/MPM/Core/MPMBoundCond.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/LinearInterpolator.h>
#include <Core/Grid/fastCpdiInterpolator.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Util/ProgressiveWarning.h>
#include <fstream>

using namespace Uintah;
using namespace std;

static DebugStream cout_doing("MPM", false);

//______________________________________________________________________
TracerTasks::TracerTasks(MaterialManagerP& ss, MPMFlags* flags)
{
  d_lb = scinew MPMLabel();
  TraL = scinew TracerLabel();

  d_flags = flags;

  if(flags->d_8or27==8){
    NGP=1;
    NGN=1;
  } else{
    NGP=2;
    NGN=2;
  }

  d_materialManager = ss;
}

TracerTasks::~TracerTasks()
{
  delete d_lb;
  delete TraL;
}

void TracerTasks::tracerProblemSetup(const ProblemSpecP& prob_spec, 
                                           MPMFlags* flags)
{
  //Search for the MaterialProperties block and then get the MPM section
  ProblemSpecP mat_ps =  
    prob_spec->findBlockWithOutAttribute("MaterialProperties");
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");
  for (ProblemSpecP ps = mpm_mat_ps->findBlock("tracer"); ps != nullptr;
       ps = ps->findNextBlock("tracer") ) {

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

    //Create and register as an Tracer material
    TracerMaterial *mat = scinew TracerMaterial(ps, d_materialManager, flags);

    mat->registerParticleState( d_tracerState,
                                d_tracerState_preReloc );

    // When doing restart, we need to make sure that we load the materials
    // in the same order that they were initially created.  Restarts will
    // ALWAYS have an index number as in <material index = "0">.
    // Index_val = -1 means that we don't register the material by its 
    // index number.
    if (index_val > -1){
      d_materialManager->registerMaterial("Tracer", mat,index_val);
    }
    else{
      d_materialManager->registerMaterial("Tracer", mat);
    }
  }
}

void TracerTasks::scheduleUpdateTracers(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSubset* mpm_matls,
                                        const MaterialSubset* tracer_matls,
                                        const MaterialSet* matls)
{
  if (!d_flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                             getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"TracerTasks::scheduleUpdateTracers");

  Task* t=scinew Task("TracerTasks::updateTracers",
                      this, &TracerTasks::updateTracers);

  t->requires(Task::OldDW, d_lb->delTLabel );

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, d_lb->gVelocityStarLabel,   mpm_matls,    gac,NGN+1);
  t->requires(Task::NewDW, d_lb->gMassLabel,           mpm_matls,    gac,NGN+1);
  t->requires(Task::NewDW, d_lb->dLdtDissolutionLabel, mpm_matls,    gac,NGN+1);
  t->requires(Task::NewDW, d_lb->gMassLabel,
             d_materialManager->getAllInOneMatls(),Task::OutOfDomain,gac,NGN+1);
  t->requires(Task::NewDW, d_lb->gVelocityLabel,
             d_materialManager->getAllInOneMatls(),Task::OutOfDomain,gac,NGN+1);
  if (d_flags->d_doingDissolution) {
    t->requires(Task::NewDW, d_lb->gSurfNormLabel,     mpm_matls,    gac,NGN+1);
  }

  t->requires(Task::OldDW, d_lb->pXLabel,            tracer_matls, gnone);
  t->requires(Task::OldDW, TraL->tracerIDLabel,      tracer_matls, gnone);
  t->requires(Task::OldDW, TraL->tracerCemVecLabel,  tracer_matls, gnone);
  t->requires(Task::OldDW, TraL->tracerChemDispLabel,tracer_matls, gnone);

  t->computes(d_lb->pXLabel_preReloc,            tracer_matls);
  t->computes(TraL->tracerIDLabel_preReloc,      tracer_matls);
  t->computes(TraL->tracerCemVecLabel_preReloc,  tracer_matls);
  t->computes(TraL->tracerChemDispLabel_preReloc,tracer_matls);

  sched->addTask(t, patches, matls);
}
//______________________________________________________________________
void TracerTasks::updateTracers(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* ,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing updateTracers");

    ParticleInterpolator* interpolator=scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    BBox domain;
    const Level* level = getLevel(patches);
    level->getInteriorSpatialRange(domain);
    Point dom_min = domain.min();
    Point dom_max = domain.max();
    IntVector periodic = level->getPeriodicBoundaries();

    delt_vartype delT;
    old_dw->get(delT, d_lb->delTLabel, getLevel(patches) );

    unsigned int numMPMMatls=d_materialManager->getNumMatls("MPM");
    std::vector<constNCVariable<Vector> > gvelocity(numMPMMatls);
    std::vector<constNCVariable<double> > gmass(numMPMMatls);
    std::vector<constNCVariable<double> > dLdt(numMPMMatls);
    std::vector<constNCVariable<Vector> > gSurfNorm(numMPMMatls);
    Matrix3 size(0.5,0.,0.,0.,0.5,0.,0.,0.,0.5); // Placeholder, not used

    Ghost::GhostType  gac = Ghost::AroundCells;
    constNCVariable<Vector>  gvelocityglobal;
    constNCVariable<double>  gmassglobal;
    new_dw->get(gmassglobal,  d_lb->gMassLabel,
           d_materialManager->getAllInOneMatls()->get(0), patch, gac, NGN+1);
    new_dw->get(gvelocityglobal,  d_lb->gVelocityLabel,
           d_materialManager->getAllInOneMatls()->get(0), patch, gac, NGN+1);
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl=(MPMMaterial*) 
                                     d_materialManager->getMaterial("MPM",m);
      int dwi = mpm_matl->getDWIndex();
      new_dw->get(gvelocity[m], d_lb->gVelocityStarLabel,  dwi,patch,gac,NGN+1);
      new_dw->get(gmass[m],     d_lb->gMassLabel,          dwi,patch,gac,NGN+1);
      new_dw->get(dLdt[m],      d_lb->dLdtDissolutionLabel,dwi,patch,gac,NGN+1);
      if (d_flags->d_doingDissolution){
        new_dw->get(gSurfNorm[m],d_lb->gSurfNormLabel,     dwi,patch,gac,NGN+1);
      } else{
        NCVariable<Vector> gSN_create;
        new_dw->allocateTemporary(gSN_create,                 patch, gac,NGN+1);
        gSN_create.initialize(Vector(0.));
        gSurfNorm[m] = gSN_create;                     // reference created data
      }
    }

    int numTracerMatls=d_materialManager->getNumMatls("Tracer");
    for(int tm = 0; tm < numTracerMatls; tm++){
      TracerMaterial* t_matl = (TracerMaterial *)
                                 d_materialManager->getMaterial("Tracer", tm );
      int dwi = t_matl->getDWIndex();

      int adv_matl = t_matl->getAssociatedMaterial();

      // Not populating the delset, but we need this to satisfy Relocate
      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);
      new_dw->deleteParticles(delset);

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> tx;
      ParticleVariable<Point> tx_new;
      constParticleVariable<long64> tracer_ids;
      ParticleVariable<long64> tracer_ids_new;
      constParticleVariable<Vector> tracerCemVec, tracerChemDisp;
      ParticleVariable<Vector> tracerCemVec_new, tracerChemDisp_new;

      old_dw->get(tx,            d_lb->pXLabel,                       pset);
      old_dw->get(tracer_ids,    TraL->tracerIDLabel,                 pset);
      old_dw->get(tracerCemVec,  TraL->tracerCemVecLabel,             pset);
      old_dw->get(tracerChemDisp,TraL->tracerChemDispLabel,           pset);

      new_dw->allocateAndPut(tx_new,          d_lb->pXLabel_preReloc,     pset);
      new_dw->allocateAndPut(tracer_ids_new,TraL->tracerIDLabel_preReloc, pset);
      new_dw->allocateAndPut(tracerCemVec_new,
                                      TraL->tracerCemVecLabel_preReloc,   pset);
      new_dw->allocateAndPut(tracerChemDisp_new,
                                      TraL->tracerChemDispLabel_preReloc, pset);

      tracer_ids_new.copyData(tracer_ids);
      tracerCemVec_new.copyData(tracerCemVec);
      tx_new.copyData(tx);
      tracerChemDisp_new.copyData(tracerChemDisp);

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        int NN = interpolator->findCellAndWeights(tx[idx],ni,S,size);
        Vector vel(0.0,0.0,0.0);
        Vector surf(0.0,0.0,0.0);
  
        double sumSk=0.0;
        Vector gSN(0.,0.,0.);
        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < NN; k++){
          IntVector node = ni[k];
          vel   += gvelocity[adv_matl][node]*gmass[adv_matl][node]*S[k];
          sumSk += gmass[adv_matl][node]*S[k];
          surf  -= dLdt[adv_matl][node]*gSurfNorm[adv_matl][node]*S[k];
          gSN   += gSurfNorm[adv_matl][node]*S[k];
        }
        if(sumSk > 1.e-90){
          // This is the normal condition, when at least one of the nodes
          // influencing a tracer has mass on it.
          vel/=sumSk;
          tx_new[idx] = tx[idx] + vel*delT;
          Vector chemDisp = (surf/(gSN.length()+1.e-100))*delT;
          tx_new[idx] += chemDisp;
          tracerChemDisp_new[idx] = tracerChemDisp[idx] + chemDisp;
        } else {
          // This is the "just in case" instance that none of the nodes
          // influencing a vertex has mass on it.  In this case, use an
          // interpolator with a larger footprint
          ParticleInterpolator* cpdiInterp=scinew fastCpdiInterpolator(patch);
          vector<IntVector> ni_cpdi(cpdiInterp->size());
          vector<double> S_cpdi(cpdiInterp->size());
          Matrix3 size; size.Identity();
          int N = cpdiInterp->findCellAndWeights(tx[idx],ni_cpdi,S_cpdi,size);
          vel  = Vector(0.0,0.0,0.0);
          surf = Vector(0.0,0.0,0.0);
          sumSk= 0.0;
          for (int k = 0; k < N; k++) {
           IntVector node = ni_cpdi[k];
            vel  += gvelocity[adv_matl][node]*gmass[adv_matl][node]*S_cpdi[k];
            sumSk+= gmass[adv_matl][node]*S_cpdi[k];
            surf -= dLdt[adv_matl][node]*gSurfNorm[adv_matl][node]*S_cpdi[k];
          }
          delete cpdiInterp;

          if(sumSk > 1.e-90){
            vel/=sumSk;
            tx_new[idx] = tx[idx] + vel*delT;
            Vector chemDisp = (surf/(gSN.length()+1.e-100))*delT;
            tx_new[idx] += chemDisp;
            tracerChemDisp_new[idx] = tracerChemDisp[idx] + chemDisp;
          } else {
            // This is the rare "just in case" instance that none of the nodes
            // influencing a tracer has mass on it.  In this case, use the
            // "center of mass" velocity to move the vertex
            double sumSkCoM=0.0;
            Vector velCoM(0.0,0.0,0.0);
            for (int k = 0; k < NN; k++) {
              IntVector node = ni[k];
              sumSkCoM += gmassglobal[node]*S[k];
              velCoM   += gvelocityglobal[node]*gmassglobal[node]*S[k];
            }
            velCoM/=sumSkCoM;
            tx_new[idx] = tx[idx] + velCoM*delT;
            if(sumSkCoM< 1.e-90){
              cout << "Group = " << adv_matl 
                   << ", tracer_id = " << tracer_ids[idx] << endl;
            }
          }
        }

#if 1
        // Check to see if a tracer has left the domain
        if(!domain.inside(tx_new[idx])){
          //cout << "tx[idx] = " << tx[idx] << endl;
          //cout << "tx_new[idx] = " << tx_new[idx] << endl;
          double epsilon = 1.e-15;
          static ProgressiveWarning warn("A tracer has moved outside the domain through an x boundary. Pushing it back in.  This is a ProgressiveWarning.",10);
          Point txn = tx_new[idx];
          if(periodic.x()==0){
           if(tx_new[idx].x()<dom_min.x()){
            tx_new[idx] = Point(dom_min.x()+epsilon, txn.y(), txn.z());
            txn = tx_new[idx];
            warn.invoke();
           }
           if(tx_new[idx].x()>dom_max.x()){
            tx_new[idx] = Point(dom_max.x()-epsilon, txn.y(), txn.z());
            txn = tx_new[idx];
            warn.invoke();
           }
          }
          if(periodic.y()==0){
           if(tx_new[idx].y()<dom_min.y()){
            tx_new[idx] = Point(txn.x(),dom_min.y()+epsilon, txn.z());
            txn = tx_new[idx];
            warn.invoke();
           }
           if(tx_new[idx].y()>dom_max.y()){
            tx_new[idx] = Point(txn.x(),dom_max.y()-epsilon, txn.z());
            txn = tx_new[idx];
            warn.invoke();
           }
          }
          if(periodic.z()==0){
           if(tx_new[idx].z()<dom_min.z()){
            tx_new[idx] = Point(txn.x(),txn.y(),dom_min.z()+epsilon);
            warn.invoke();
           }
           if(tx_new[idx].z()>dom_max.z()){
            tx_new[idx] = Point(txn.x(),txn.y(),dom_max.z()-epsilon);
            warn.invoke();
           }
          }
        } // if tracer has left domain
#endif
      }
    }
    delete interpolator;
  }
}

void TracerTasks::scheduleAddTracers(SchedulerP& sched,
                                     const PatchSet* patches,
                                     const MaterialSet* tracer_matls)

{
  if( !d_flags->doMPMOnLevel( getLevel(patches)->getIndex(), 
                              getLevel(patches)->getGrid()->numLevels() ) ) {
    return;
  }

  printSchedule( patches, cout_doing, "TracerTasks::scheduleAddTracers" );

  Task * t = scinew Task("TracerTasks::addTracers", this,
                         &TracerTasks::addTracers );

  t->modifies(TraL->tracerIDLabel_preReloc, tracer_matls);
  t->modifies(d_lb->pXLabel_preReloc,       tracer_matls);

  sched->addTask(t, patches, tracer_matls);
}

void TracerTasks::addTracers(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* ,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing addTracers");

    if(d_flags->d_doAuthigenesis){
      cout << "Doing addTracers" << endl;

      int tm = 0;  // Only one tracer material now
      int numNewTracers=0;

      TracerMaterial* tr_matl = (TracerMaterial*) 
                                   d_materialManager->getMaterial("Tracer", tm);
      int dwi = tr_matl->getDWIndex();
      Tracer* tr = tr_matl->getTracer();

      string filename = tr_matl->getTracerFilename();
      numNewTracers = tr->countTracers(patch,filename);

//      cout << "numNewTracers = " << numNewTracers << endl;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      ParticleVariable<Point> px;
      ParticleVariable<long64> pids;
      new_dw->getModifiable(px,         d_lb->pXLabel_preReloc,         pset);
      new_dw->getModifiable(pids,     TraL->tracerIDLabel_preReloc,     pset);

      ParticleSubset* psetnew = 
                new_dw->createParticleSubset(numNewTracers,dwi,patch);

      ParticleVariable<Point> pxtmp;
      ParticleVariable<long64> pidstmp;
      new_dw->allocateTemporary(pidstmp,  psetnew);
      new_dw->allocateTemporary(pxtmp,    psetnew);

      std::ifstream is(filename.c_str());

      double p1,p2,p3;
      string line;
      particleIndex start = 0;
      while (getline(is, line)) {
       istringstream ss(line);
       string token;
       long64 tid;
       ss >> token;
       tid = stoull(token);
       ss >> token;
       p1 = stof(token);
       ss >> token;
       p2 = stof(token);
       ss >> token;
       p3 = stof(token);
//     cout << tid << " " << p1 << " " << p2 << " " << p3 << endl;
       Point pos = Point(p1,p2,p3);

       if(patch->containsPoint(pos)){
         particleIndex pidx = start;
         pxtmp[pidx]   = pos;
         pidstmp[pidx] = tid;
         start++;
       }
      }

      is.close();

      // put back temporary data
      new_dw->put(pxtmp,      d_lb->pXLabel_preReloc,           true);
      new_dw->put(pidstmp,  TraL->tracerIDLabel_preReloc,       true);
   }    // if doAuth && AddedNewParticles<1.0....
  }   // for patches
//   flags->d_doAuthigenesis = false;
}

void TracerTasks::scheduleComputeGridCemVec(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSubset* mpm_matls,
                                            const MaterialSubset* tracer_matls,
                                            const MaterialSet* matls)
{
  if (!d_flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"TracerTasks::scheduleComputeGridCemVec");

  Task* t=scinew Task("TracerTasks::computeGridCemVec",
                      this, &TracerTasks::computeGridCemVec);

  Ghost::GhostType  gac = Ghost::AroundCells;

  t->requires(Task::OldDW, d_lb->pXLabel,            tracer_matls, gac, 2);
  t->requires(Task::OldDW, TraL->tracerCemVecLabel,  tracer_matls, gac, 2);
  t->requires(Task::NewDW, d_lb->gMassLabel,         mpm_matls,    gac,NGN+3);

  t->computes(d_lb->gCemVecLabel,                    mpm_matls);

  sched->addTask(t, patches, matls);
}

void TracerTasks::computeGridCemVec(const ProcessorGroup *,
                                    const PatchSubset    * patches,
                                    const MaterialSubset * ,
                                          DataWarehouse  * old_dw,
                                          DataWarehouse  * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing computeGridCemVec");

    ParticleInterpolator* interpolator=scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    string interp_type = d_flags->d_interpolator_type;

    Ghost::GhostType gan   = Ghost::AroundNodes;

    unsigned int numMPMMatls = d_materialManager->getNumMatls( "MPM" );
    std::vector<NCVariable<Vector> >       gcemvec(numMPMMatls);
    std::vector<NCVariable<double> >       SumS(numMPMMatls);

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                     (MPMMaterial*) d_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      new_dw->allocateAndPut(gcemvec[m],    d_lb->gCemVecLabel, dwi, patch);
      new_dw->allocateTemporary(SumS[m],                             patch);
      gcemvec[m].initialize(Vector(0.0,0.0,0.0));
      SumS[m].initialize(0.0);
    }

    int numTraMatls=d_materialManager->getNumMatls("Tracer");
    Matrix3 size; size.Identity();

    for(int tmo = 0; tmo < numTraMatls; tmo++) {
      TracerMaterial* t_matl = (TracerMaterial *) 
                             d_materialManager->getMaterial("Tracer", tmo);
      int dwi_tra = t_matl->getDWIndex();
      int adv_matl = t_matl->getAssociatedMaterial();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi_tra, patch,
                                                        gan, 2, d_lb->pXLabel);
      constParticleVariable<Point>  tx;
      constParticleVariable<Vector> tracerCemVec;
      old_dw->get(tx,               d_lb->pXLabel,           pset);
      old_dw->get(tracerCemVec,   TraL->tracerCemVecLabel,   pset);

      for(ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); iter++){
         particleIndex idx = *iter;
         int nn = interpolator->findCellAndWeights(tx[idx], ni, S, size);
         for (int k = 0; k < nn; k++) {
           IntVector node = ni[k];
           if(patch->containsNode(node)){
             gcemvec[adv_matl][node] += tracerCemVec[idx]*S[k];
             SumS[adv_matl][node]    += S[k];
           }
         }
      } // triangles
    }   // triangle materials

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = 
                   (MPMMaterial*) d_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      for(NodeIterator iter =patch->getExtraNodeIterator();!iter.done();iter++){
        IntVector c = *iter;
        gcemvec[m][c] /= (SumS[m][c]+1.e-100);
      } // Nodes

      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Symmetric",  gcemvec[m],interp_type);
    }   // MPM materials
  }     // patches
}

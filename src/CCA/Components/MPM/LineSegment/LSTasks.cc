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
#include <CCA/Components/MPM/LineSegment/LSTasks.h>
#include <CCA/Components/MPM/LineSegment/LineSegmentMaterial.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Core/LineSegmentLabel.h>
#include <CCA/Components/MPM/Core/MPMBoundCond.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Output.h>
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
LSTasks::LSTasks(MaterialManagerP& ss, MPMFlags* flags, Output* m_output)
{
  lb = scinew MPMLabel();
  LSl= scinew LineSegmentLabel();

  d_flags = flags;
  d_output = m_output;

  if(flags->d_8or27==8){
    NGP=1;
    NGN=1;
  } else{
    NGP=2;
    NGN=2;
  }

  d_materialManager = ss;
}

LSTasks::~LSTasks()
{
  delete lb;
  delete LSl;
}

void LSTasks::lineSegmentProblemSetup(const ProblemSpecP& prob_spec, 
                                      MPMFlags* flags)
{
  //Search for the MaterialProperties block and then get the MPM section
  ProblemSpecP mat_ps =  
    prob_spec->findBlockWithOutAttribute("MaterialProperties");
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");
  for (ProblemSpecP ps = mpm_mat_ps->findBlock("LineSegment"); ps != nullptr;
       ps = ps->findNextBlock("LineSegment") ) {

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

    //Create and register as an LineSegment material
    LineSegmentMaterial *mat = 
                       scinew LineSegmentMaterial(ps, d_materialManager, flags);

    mat->registerParticleState( d_lineseg_state,
                                d_lineseg_state_preReloc );

    // When doing restart, we need to make sure that we load the materials
    // in the same order that they were initially created.  Restarts will
    // ALWAYS have an index number as in <material index = "0">.
    // Index_val = -1 means that we don't register the material by its 
    // index number.
    if (index_val > -1){
      d_materialManager->registerMaterial("LineSegment", mat,index_val);
    }
    else{
      d_materialManager->registerMaterial("LineSegment", mat);
    }
  }
}

void LSTasks::scheduleUpdateLineSegments(SchedulerP& sched,
                                         const PatchSet* patches,
                                         const MaterialSubset* mpm_matls,
                                         const MaterialSubset* lineseg_matls,
                                         const MaterialSet* matls)
{
  if (!d_flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"LSTasks::scheduleUpdateLineSegments");

  Task* t=scinew Task("LSTasks::updateLineSegments",
                      this, &LSTasks::updateLineSegments);

  t->requires(Task::OldDW, lb->delTLabel );

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, lb->gVelocityStarLabel,    mpm_matls,     gac,NGN+1);
  t->requires(Task::NewDW, lb->gMassLabel,            mpm_matls,     gac,NGN+1);
  t->requires(Task::OldDW, lb->pXLabel,               lineseg_matls, gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,            lineseg_matls, gnone);
  t->requires(Task::OldDW, LSl->linesegIDLabel,       lineseg_matls, gnone);
  t->requires(Task::OldDW, LSl->lsMidToEndVectorLabel,lineseg_matls, gnone);
  t->requires(Task::OldDW, lb->pDeformationMeasureLabel,
                                                      lineseg_matls, gnone);
  t->requires(Task::NewDW, lb->dLdtDissolutionLabel,  mpm_matls,     gac,NGN+1);
  if (d_flags->d_doingDissolution) {
    t->requires(Task::NewDW, lb->gSurfNormLabel,      mpm_matls,     gac,NGN+1);
  }

  t->computes(lb->pXLabel_preReloc,                      lineseg_matls);
  t->computes(lb->pSizeLabel_preReloc,                   lineseg_matls);
  t->computes(LSl->linesegIDLabel_preReloc,              lineseg_matls);
  t->computes(LSl->lsMidToEndVectorLabel_preReloc,       lineseg_matls);
  t->computes(lb->pDeformationMeasureLabel_preReloc,     lineseg_matls);

  sched->addTask(t, patches, matls);
}

void LSTasks::updateLineSegments(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* ,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing updateLineSegments");

    ParticleInterpolator* interpolator=scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    Vector dx = patch->dCell();
    Matrix3 size; size.Identity();

    BBox domain;
    const Level* level = getLevel(patches);
    level->getInteriorSpatialRange(domain);
    Point dom_min = domain.min();
    Point dom_max = domain.max();
    IntVector periodic = level->getPeriodicBoundaries();

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );

    unsigned int numMPMMatls=d_materialManager->getNumMatls("MPM");
    std::vector<constNCVariable<Vector> > gvelocity(numMPMMatls);
    std::vector<constNCVariable<double> > gmass(numMPMMatls);
    std::vector<constNCVariable<double> > dLdt(numMPMMatls);
    std::vector<constNCVariable<Vector> > gSurfNorm(numMPMMatls);
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl=(MPMMaterial*) 
                                     d_materialManager->getMaterial("MPM",m);
      int dwi = mpm_matl->getDWIndex();
      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gvelocity[m], lb->gVelocityStarLabel,  dwi, patch, gac,NGN+1);
      new_dw->get(gmass[m],     lb->gMassLabel,          dwi, patch, gac,NGN+1);
      new_dw->get(dLdt[m],      lb->dLdtDissolutionLabel,dwi, patch, gac,NGN+1);
      if (d_flags->d_doingDissolution){
        new_dw->get(gSurfNorm[m],lb->gSurfNormLabel,     dwi, patch, gac,NGN+1);
      } else{
        NCVariable<Vector> gSN_create;
        new_dw->allocateTemporary(gSN_create,                 patch, gac,NGN+1);
        gSN_create.initialize(Vector(0.));
        gSurfNorm[m] = gSN_create;                     // reference created data
      }
    }

    int numLSMatls=d_materialManager->getNumMatls("LineSegment");
    for(int ls = 0; ls < numLSMatls; ls++){
      LineSegmentMaterial* ls_matl = (LineSegmentMaterial *) 
                              d_materialManager->getMaterial("LineSegment", ls);
      int dwi = ls_matl->getDWIndex();

      int adv_matl = ls_matl->getAssociatedMaterial();

      // Not populating the delset, but we need this to satisfy Relocate
      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);
      new_dw->deleteParticles(delset);

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> tx;
      ParticleVariable<Point> tx_new;
      constParticleVariable<Matrix3> tsize, tF;
      ParticleVariable<Matrix3> tsize_new, tF_new;
      constParticleVariable<long64> lineseg_ids;
      ParticleVariable<long64> lineseg_ids_new;
      constParticleVariable<Vector> lsMidToEndVec;
      ParticleVariable<Vector> lsMidToEndVec_new;

      old_dw->get(tx,            lb->pXLabel,                         pset);
      old_dw->get(tsize,         lb->pSizeLabel,                      pset);
      old_dw->get(lineseg_ids,   LSl->linesegIDLabel,                 pset);
      old_dw->get(tF,            lb->pDeformationMeasureLabel,        pset);
      old_dw->get(lsMidToEndVec, LSl->lsMidToEndVectorLabel,           pset);

      new_dw->allocateAndPut(tx_new,         lb->pXLabel_preReloc,        pset);
      new_dw->allocateAndPut(tsize_new,      lb->pSizeLabel_preReloc,     pset);
      new_dw->allocateAndPut(lineseg_ids_new,LSl->linesegIDLabel_preReloc,pset);
      new_dw->allocateAndPut(tF_new,lb->pDeformationMeasureLabel_preReloc,pset);
      new_dw->allocateAndPut(lsMidToEndVec_new,
                                      LSl->lsMidToEndVectorLabel_preReloc,pset);

      lineseg_ids_new.copyData(lineseg_ids);
      tF_new.copyData(tF);

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // First update the position of the "right" end of the segment
        Point right = tx[idx]+lsMidToEndVec[idx];
        Point left  = tx[idx]-lsMidToEndVec[idx];
        // Get the node indices that surround the cell
        int NN = interpolator->findCellAndWeights(right, ni, S, size);
        Vector vel(0.0,0.0,0.0);
        Vector surf(0.0,0.0,0.0);
//        Vector v = left - right;

//        normal = v X (0,0,1)
//        Vector normal = Vector(v.y(), -v.x(), 0.)
//                         / (1.e-100+sqrt(v.y()*v.y()+v.x()*v.x()));
  
        double sumSk =0.0;
        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < NN; k++) {
          IntVector node = ni[k];
          vel   += gvelocity[adv_matl][node]*gmass[adv_matl][node]*S[k];
          sumSk += gmass[adv_matl][node]*S[k];
          surf   -= dLdt[adv_matl][node]*gSurfNorm[adv_matl][node]*S[k];
        }
        vel/=sumSk;
  
        right += vel*delT;
        right += surf*delT;
  
        // Next update the position of the "left" end of the segment
        // Get the node indices that surround the cell
        NN = interpolator->findCellAndWeights(left, ni, S, size);
        vel = Vector(0.0,0.0,0.0);
        surf = Vector(0.0,0.0,0.0);
  
        sumSk=0.0;
        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < NN; k++) {
          IntVector node = ni[k];
          vel   += gvelocity[adv_matl][node]*gmass[adv_matl][node]*S[k];
          sumSk += gmass[adv_matl][node]*S[k];
          surf   -= dLdt[adv_matl][node]*gSurfNorm[adv_matl][node]*S[k];
        }
        vel/=sumSk;
  
        left += vel*delT;
        left += surf*delT;

        tx_new[idx] = 0.5*(left+right);

        Vector lsETE = (right-left);
        lsMidToEndVec_new[idx] = 0.5*lsETE;
        Matrix3 size_new =Matrix3(lsETE.x()/dx.x(), 0.1*lsETE.y()/dx.y(), 0.0,
                                  lsETE.y()/dx.x(), -.1*lsETE.x()/dx.y(), 0.0,
                                              0.0,                  0.0, 1.0);
        tsize_new[idx] = size_new;
  
        // Check to see if a line segment has left the domain
        if(!domain.inside(tx_new[idx])){
          double epsilon = 1.e-15;
          Point txn = tx_new[idx];
          if(periodic.x()==0){
           if(tx_new[idx].x()<dom_min.x()){
            tx_new[idx] = Point(dom_min.x()+epsilon, txn.y(), txn.z());
            txn = tx_new[idx];
           }
           if(tx_new[idx].x()>dom_max.x()){
            tx_new[idx] = Point(dom_max.x()-epsilon, txn.y(), txn.z());
            txn = tx_new[idx];
           }
           static ProgressiveWarning warn("A tracer has moved outside the domain through an x boundary. Pushing it back in.  This is a ProgressiveWarning.",10);
           warn.invoke();
          }
          if(periodic.y()==0){
           if(tx_new[idx].y()<dom_min.y()){
            tx_new[idx] = Point(txn.x(),dom_min.y()+epsilon, txn.z());
            txn = tx_new[idx];
           }
           if(tx_new[idx].y()>dom_max.y()){
            tx_new[idx] = Point(txn.x(),dom_max.y()-epsilon, txn.z());
            txn = tx_new[idx];
           }
           static ProgressiveWarning warn("A tracer has moved outside the domain through a y boundary. Pushing it back in.  This is a ProgressiveWarning.",10);
           warn.invoke();
          }
          if(periodic.z()==0){
           if(tx_new[idx].z()<dom_min.z()){
            tx_new[idx] = Point(txn.x(),txn.y(),dom_min.z()+epsilon);
           }
           if(tx_new[idx].z()>dom_max.z()){
            tx_new[idx] = Point(txn.x(),txn.y(),dom_max.z()-epsilon);
           }
           static ProgressiveWarning warn("A tracer has moved outside the domain through a z boundary. Pushing it back in.  This is a ProgressiveWarning.",10);
           warn.invoke();
          }
        }
      }
    }
    delete interpolator;
  }
}

void LSTasks::scheduleComputeLineSegmentForces(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSubset* mpm_matls,
                                            const MaterialSubset* lineseg_matls,
                                            const MaterialSet* matls)
{
  if (!d_flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"LSTasks::scheduleComputeLineSegmentForces");

  Task* t=scinew Task("LSTasks::computeLineSegmentForces",
                      this, &LSTasks::computeLineSegmentForces);

  Ghost::GhostType  gac = Ghost::AroundCells;

  t->requires(Task::OldDW, lb->pXLabel,              lineseg_matls, gac, 2);
  t->requires(Task::OldDW, lb->pSizeLabel,           lineseg_matls, gac, 2);
  t->requires(Task::OldDW, LSl->lsMidToEndVectorLabel,lineseg_matls,gac, 2);
  t->requires(Task::NewDW, lb->gMassLabel,           mpm_matls,     gac, NGN+2);

  t->computes(lb->gLSContactForceLabel,              mpm_matls);
  t->computes(lb->gSurfaceAreaLabel,                 mpm_matls);
  t->computes(lb->gInContactMatlLabel,               mpm_matls);

  sched->addTask(t, patches, matls);
}

void LSTasks::computeLineSegmentForces(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* ,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing computeLineSegmentForces");

    ParticleInterpolator* interpolator=scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    Ghost::GhostType gac = Ghost::AroundCells;
    Vector dxCell = patch->dCell();
    double cell_length2 = dxCell.length2();

    unsigned int numMPMMatls=d_materialManager->getNumMatls( "MPM" );
    std::vector<NCVariable<Vector> > LSContForce(numMPMMatls);
    std::vector<NCVariable<double> > SurfArea(numMPMMatls);
    std::vector<NCVariable<int> > InContactMatl(numMPMMatls);
    std::vector<constNCVariable<double> > gmass(numMPMMatls);
    std::vector<double> stiffness(numMPMMatls);
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                     (MPMMaterial*) d_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      double inv_stiff = mpm_matl->getConstitutiveModel()->getCompressibility();
      stiffness[m] = 1./inv_stiff;

      new_dw->allocateAndPut(LSContForce[m],lb->gLSContactForceLabel,dwi,patch);
      new_dw->allocateAndPut(SurfArea[m],   lb->gSurfaceAreaLabel,   dwi,patch);
      new_dw->allocateAndPut(InContactMatl[m],
                                            lb->gInContactMatlLabel, dwi,patch);
      new_dw->get(gmass[m],                 lb->gMassLabel,          dwi,patch,
                                                                     gac,NGN+2);
      LSContForce[m].initialize(Vector(0.0));
      SurfArea[m].initialize(1.0e-100);
      InContactMatl[m].initialize(-99);
    }

    int numLSMatls=d_materialManager->getNumMatls("LineSegment");

    // Get the arrays of particle values to be changed
    std::vector<constParticleVariable<Point>  >  tx0(numLSMatls);
    std::vector<constParticleVariable<Matrix3> > tsize0(numLSMatls);
    std::vector<constParticleVariable<Vector>  > lsMidToEndVec0(numLSMatls);
    std::vector<ParticleSubset*> psetvec;

    for(int tmo = 0; tmo < numLSMatls; tmo++) {
      LineSegmentMaterial* t_matl0 = (LineSegmentMaterial *) 
                             d_materialManager->getMaterial("LineSegment", tmo);
      int dwi0 = t_matl0->getDWIndex();

      ParticleSubset* pset0 = old_dw->getParticleSubset(dwi0, patch,
                                                       gac, 2, lb->pXLabel);
      psetvec.push_back(pset0);

      old_dw->get(tx0[tmo],            lb->pXLabel,                   pset0);
      old_dw->get(tsize0[tmo],         lb->pSizeLabel,                pset0);
      old_dw->get(lsMidToEndVec0[tmo], LSl->lsMidToEndVectorLabel,    pset0);
    }

    for(int tmo = 0; tmo < numLSMatls; tmo++) {
      LineSegmentMaterial* t_matl0 = (LineSegmentMaterial *) 
                             d_materialManager->getMaterial("LineSegment", tmo);
      int adv_matl0 = t_matl0->getAssociatedMaterial();

      ParticleSubset* pset0 = psetvec[tmo];

      // Extrapolate area of line segments to the grid for use in dissolution
      if (d_flags->d_doingDissolution){
       for(ParticleSubset::iterator iter0 = pset0->begin();
           iter0 != pset0->end(); iter0++){
         particleIndex idx0 = *iter0;

         Point px0=tx0[tmo][idx0] - lsMidToEndVec0[tmo][idx0];
         Point a = tx0[tmo][idx0] + lsMidToEndVec0[tmo][idx0];
         Vector v = px0 - a;
         double vLength = v.length();
         double LSArea = vLength*dxCell.z();
         Matrix3 size0 = tsize0[tmo][idx0];
         int nn = interpolator->findCellAndWeights(px0,ni,S,size0);
         double totMass = 0.;
         for (int k = 0; k < nn; k++) {
           IntVector node = ni[k];
           totMass += S[k]*gmass[adv_matl0][node];
        }

         for (int k = 0; k < nn; k++) {
           IntVector node = ni[k];
           if(patch->containsNode(node)) {
             SurfArea[adv_matl0][node] += 0.5*LSArea*S[k]*gmass[adv_matl0][node]
                                          /totMass;
           }
         }

         nn = interpolator->findCellAndWeights(a,ni,S,size0);
         totMass = 0.;
         for (int k = 0; k < nn; k++) {
           IntVector node = ni[k];
           totMass += S[k]*gmass[adv_matl0][node];
         }

         for (int k = 0; k < nn; k++) {
           IntVector node = ni[k];
           if(patch->containsNode(node)) {
             SurfArea[adv_matl0][node] += 0.5*LSArea*S[k]*gmass[adv_matl0][node]
                                          /totMass;
           }
         }
        } // loop over particles
      }  // if doingDissolution

      for(int tmi = tmo+1; tmi < numLSMatls; tmi++) {
        LineSegmentMaterial* t_matl1 = (LineSegmentMaterial *) 
                              d_materialManager->getMaterial("LineSegment",tmi);
        int adv_matl1 = t_matl1->getAssociatedMaterial();

        if(adv_matl0==adv_matl1){
          continue;
        }

        ParticleSubset* pset1 = psetvec[tmi];

        int numPar_pset1 = pset1->numParticles();

        double K_l = 10.*(stiffness[adv_matl0] * stiffness[adv_matl1])/
                         (stiffness[adv_matl0] + stiffness[adv_matl1]);

       if(numPar_pset1 > 0){

        // Loop over zeroth line segment subset
        // Only test the "left" end of the line segment for
        // penetration into other segments, because the right
        // end will get checked as the left end of the next segment
        for(ParticleSubset::iterator iter0 = pset0->begin();
            iter0 != pset0->end(); iter0++){
          particleIndex idx0 = *iter0;

          Point px0=tx0[tmo][idx0] - lsMidToEndVec0[tmo][idx0];

          double min_sep  = 9.e99;
          double min_sep2  = 9.e99;
          int closest = 99999;
          int secondClosest = 99999;
          // Loop over other particle subset
          for(ParticleSubset::iterator iter1 = pset1->begin();
              iter1 != pset1->end(); iter1++){
            particleIndex idx1 = *iter1;
            Point px1 = tx0[tmi][idx1];
            double sep = (px1-px0).length2();
            if(sep < min_sep2 && sep < 0.25*cell_length2){
              if(sep < min_sep){
                secondClosest=closest;
                min_sep2=min_sep;
                closest  = idx1;
                min_sep  = sep;
              } 
              else{
                secondClosest=idx1;
                min_sep2  = sep;
              }
            }
          }

          double forceMag=0.0;
          bool done = false;
          double tC1 = 99.9;
          double tC2 = 99.9;
          double overlap1 = 99.9;
          double overlap2 = 99.9;
          Vector normal1(0.,0.,0);
          Vector normal2(0.,0.,0);
          if(closest < 99999){
            // Following the description in stackexchange:
            // https://math.stackexchange.com/questions/2193720/find-a-point-on-a-line-segment-which-is-the-closest-to-other-point-not-on-the-li
           Point A = tx0[tmi][closest] + lsMidToEndVec0[tmi][closest];
           Point B = tx0[tmi][closest] - lsMidToEndVec0[tmi][closest];
           Vector v = B - A;
           double vLength2 = v.length2();

           Vector u = A - px0;
           tC1 = -Dot(v,u)/vLength2;
           Vector fromLineSegToPoint1= px0.asVector() - ((1.-tC1)*A + tC1*B);
           //normal = v X (0,0,1)
           normal1 = Vector(v.y(), -v.x(), 0.)
                         / (1.e-100+sqrt(v.y()*v.y()+v.x()*v.x()));
           overlap1 = Dot(normal1,fromLineSegToPoint1);
           if(tC1 >= 0.0 && tC1 <= 1.0){
              if(overlap1 < 0.0){
               done = true;
               double vLength = sqrt(vLength2);
               double K = K_l*vLength;
               forceMag = overlap1*K;

               Vector tForce1A = (1.-tC1)*forceMag*normal1;
               Vector tForce1B = tC1*forceMag*normal1;

               // See comments in addCohesiveZoneForces for a description of how
               // the force is put on the nodes

               // Get the node indices that surround the cell
               Matrix3 size1 = tsize0[tmi][closest];
               int NN = interpolator->findCellAndWeights(A, ni, S, size1);
  
               double totMass0 = 0.;
               double totMass1 = 0.;
               for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 totMass0 += S[k]*gmass[adv_matl0][node];
                 totMass1 += S[k]*gmass[adv_matl1][node];
//               }
               }

               // Accumulate the contribution from each surrounding vertex
               for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
                 if(patch->containsNode(node)) {
                   // Distribute force according to material mass on the nodes
                   // to get nearly equal contribution to the acceleration
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                   LSContForce[adv_matl0][node] -= tForce1A*S[k]
                                             * gmass[adv_matl0][node]/totMass0;
                   LSContForce[adv_matl1][node] += tForce1A*S[k]
                                             * gmass[adv_matl1][node]/totMass1;
                   InContactMatl[adv_matl0][node] = adv_matl1;
                   InContactMatl[adv_matl1][node] = adv_matl0;
//                 }
                 }
               }

               // Get the node indices that surround the cell
               NN = interpolator->findCellAndWeights(B, ni, S, size1);
  
               totMass0 = 0.;
               totMass1 = 0.;
               for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 totMass0 += S[k]*gmass[adv_matl0][node];
                 totMass1 += S[k]*gmass[adv_matl1][node];
//               }
               }

               // Accumulate the contribution from each surrounding vertex
               for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
                 if(patch->containsNode(node)) {
                   // Distribute force according to material mass on the nodes
                   // to get nearly equal contribution to the acceleration
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                   LSContForce[adv_matl0][node] -= tForce1B*S[k]
                                             * gmass[adv_matl0][node]/totMass0;
                   LSContForce[adv_matl1][node] += tForce1B*S[k]
                                             * gmass[adv_matl1][node]/totMass1;
                   InContactMatl[adv_matl0][node] = adv_matl1;
                   InContactMatl[adv_matl1][node] = adv_matl0;
//                 }
                 }
               }

              } // if overlap1
           } // if(tC1 >= 0.0 && tC1 <= 1.0)

           if(!done && secondClosest < 99999){
            Point A2=tx0[tmi][secondClosest]+lsMidToEndVec0[tmi][secondClosest];
            Point B2=tx0[tmi][secondClosest]-lsMidToEndVec0[tmi][secondClosest];
            Vector v2 = B2 - A2;
            double v2Length2 = v2.length2();

            Vector u2 = A2 - px0;
            tC2 = -Dot(v2,u2)/v2Length2;
            Vector fromLineSegToPoint2= px0.asVector() - ((1.-tC2)*A2 + tC2*B2);
            normal2 = Vector(v2.y(), -v2.x(), 0.)
                        /(1.e-100 + sqrt(v2.y()*v2.y() + v2.x()*v2.x()));
            overlap2 = Dot(normal2,fromLineSegToPoint2);
            if(((tC1 < 0.0 && tC2 > 1.0) || (tC1 > 1.0 && tC2 < 0.0)) && 
               overlap1 < 0.0 && overlap2 < 0.0 && !done){
              done = true;

              double vLength = sqrt(vLength2);
              double K = K_l*vLength;

              Point vertex;
              Vector n;
              double sizeWeight1, sizeWeight2;
              Matrix3 size1,size2;
              if((px0-A).length2() < (px0-B).length2()){ // closest
               vertex = A;
               n=normal1;
               size1 = tsize0[tmi][closest];
               size2 = tsize0[tmi][secondClosest];
               sizeWeight1=fabs(tC2-1.)/(fabs(tC2-1.) + fabs(tC1));
               sizeWeight2=fabs(tC1)/(fabs(tC2-1.) + fabs(tC1));
              } else {  // secondClosest;
               vertex = B;
               n=normal2;
               size1 = tsize0[tmi][secondClosest];
               size2 = tsize0[tmi][closest];
               sizeWeight1=fabs(tC1-1.)/(fabs(tC1-1.) + fabs(tC2));
               sizeWeight2=fabs(tC2)/(fabs(tC1-1.) + fabs(tC2));
              }
//              forceMag = overlap1*K;
//              Vector tForce1A = forceMag*normal1;
//              Vector tForce1A = forceMag*n;
              Vector tForce1A = K*(px0-vertex);
              // Get the node indices that surround the cell
//              Matrix3 size_mean = 0.5*(size1+size2);
              Matrix3 size_mean = sizeWeight1*size1 + sizeWeight2*size2;
              int NN = interpolator->findCellAndWeights(px0, ni, S, size_mean);
 
              double totMass0 = 0.;
              double totMass1 = 0.;
              for (int k = 0; k < NN; k++) {
                IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                totMass0 += S[k]*gmass[adv_matl0][node];
                totMass1 += S[k]*gmass[adv_matl1][node];
//              }
              }

              // Accumulate the contribution from each surrounding vertex
              for (int k = 0; k < NN; k++) {
                IntVector node = ni[k];
                if(patch->containsNode(node)) {
                  // Distribute force according to material mass on the nodes
                  // to get nearly equal contribution to the acceleration
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                  LSContForce[adv_matl0][node] -= tForce1A*S[k]
                                            * gmass[adv_matl0][node]/totMass0;
                  LSContForce[adv_matl1][node] += tForce1A*S[k]
                                            * gmass[adv_matl1][node]/totMass1;
                   InContactMatl[adv_matl0][node] = adv_matl1;
                   InContactMatl[adv_matl1][node] = adv_matl0;
//                }
                }
              }
            } // if(tC1...)
           }  // if secondClosest

          } // closest < 99999

          if(!done && secondClosest < 99999){
            Point A2=tx0[tmi][secondClosest]+lsMidToEndVec0[tmi][secondClosest];
            Point B2=tx0[tmi][secondClosest]-lsMidToEndVec0[tmi][secondClosest];
            Vector v2 = B2 - A2;
            double v2Length2 = v2.length2();

            Vector u2 = A2 - px0;
            tC2 = -Dot(v2,u2)/v2Length2;
            Vector fromLineSegToPoint2= px0.asVector() - ((1.-tC2)*A2 + tC2*B2);
            //normal = v X (0,0,1)
            if(tC2 >= 0.0 && tC2 <= 1.0){
              Vector normal = Vector(v2.y(), -v2.x(), 0.)
                          / (1.e-100+sqrt(v2.y()*v2.y()+v2.x()*v2.x()));
              overlap2 = Dot(normal,fromLineSegToPoint2);
              if(overlap2 < 0.0){
               done = true;
               double v2Length = sqrt(v2Length2);
               double K = K_l*v2Length;
               forceMag = overlap2*K;

               Vector tForce1A = (1.-tC2)*forceMag*normal;
               Vector tForce1B = tC2*forceMag*normal;

               // Get the node indices that surround the cell
               Matrix3 size1 = tsize0[tmi][closest];
               int NN = interpolator->findCellAndWeights(A2, ni, S, size1);
  
               double totMass0 = 0.;
               double totMass1 = 0.;
               for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 totMass0 += S[k]*gmass[adv_matl0][node];
                 totMass1 += S[k]*gmass[adv_matl1][node];
//               }
               }

               // Accumulate the contribution from each surrounding vertex
               for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
                 if(patch->containsNode(node)) {
                   // Distribute force according to material mass on the nodes
                   // to get nearly equal contribution to the acceleration
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                   LSContForce[adv_matl0][node] -= tForce1A*S[k]
                                             * gmass[adv_matl0][node]/totMass0;
                   LSContForce[adv_matl1][node] += tForce1A*S[k]
                                             * gmass[adv_matl1][node]/totMass1;
                   InContactMatl[adv_matl0][node] = adv_matl1;
                   InContactMatl[adv_matl1][node] = adv_matl0;
//                 }
                 }
               }

               // Get the node indices that surround the cell
               NN = interpolator->findCellAndWeights(B2, ni, S, size1);
  
               totMass0 = 0.;
               totMass1 = 0.;
               for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 totMass0 += S[k]*gmass[adv_matl0][node];
                 totMass1 += S[k]*gmass[adv_matl1][node];
//               }
               }

               // Accumulate the contribution from each surrounding vertex
               for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
                 if(patch->containsNode(node)) {
                   // Distribute force according to material mass on the nodes
                   // to get nearly equal contribution to the acceleration
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                   LSContForce[adv_matl0][node] -= tForce1B*S[k]
                                             * gmass[adv_matl0][node]/totMass0;
                   LSContForce[adv_matl1][node] += tForce1B*S[k]
                                             * gmass[adv_matl1][node]/totMass1;
                   InContactMatl[adv_matl0][node] = adv_matl1;
                   InContactMatl[adv_matl1][node] = adv_matl0;
//                 }
                 }
               }
              } // if overlap2
            }
          }
        } //  Outer loop over linesegments
       }
      } // inner loop over line segment materials
    } // outer loop over line segment materials
//    cout << "numOverlap = " << numOverlap << endl;
    delete interpolator;
  }
}

void LSTasks::scheduleComputeLineSegScaleFactor(SchedulerP  & sched,
                                                const PatchSet    * patches,
                                                const MaterialSet * matls )
{
  if (!d_flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels())) {
    return;
  }

  printSchedule( patches, cout_doing, 
                          "LSTasks::scheduleComputeLineSegScaleFactor");

  Task * t = scinew Task( "LSTasks::computeLineSegScaleFactor",this, 
                          &LSTasks::computeLineSegScaleFactor);

  t->requires( Task::NewDW, lb->pSizeLabel_preReloc,              Ghost::None );
  t->computes( lb->pScaleFactorLabel_preReloc );

  sched->addTask( t, patches, matls );
}

void LSTasks::computeLineSegScaleFactor(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* ,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  // This task computes the particles physical size, to be used
  // in scaling particles for the deformed particle vis feature

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
                                   "Doing LSTasks::computeLineSegScaleFactor");

    unsigned int numLSMatls=d_materialManager->getNumMatls( "LineSegment" );
    for(unsigned int m = 0; m < numLSMatls; m++){
      LineSegmentMaterial* ls_matl = 
        (LineSegmentMaterial*) d_materialManager->getMaterial("LineSegment", m);
      int dwi = ls_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      constParticleVariable<Matrix3> psize,pF;
      ParticleVariable<Matrix3> pScaleFactor;
      new_dw->get(psize,        lb->pSizeLabel_preReloc,                  pset);
      new_dw->allocateAndPut(pScaleFactor, lb->pScaleFactorLabel_preReloc,pset);

      if(d_output->isOutputTimeStep()){
        Vector dx = patch->dCell();
        for(ParticleSubset::iterator iter  = pset->begin();
                                     iter != pset->end(); iter++){
          particleIndex idx = *iter;
          pScaleFactor[idx] = ((Matrix3(dx[0],0,0,
                                        0,dx[1],0,
                                        0,0,dx[2])*psize[idx]));
        } // for particles
      } // isOutputTimestep
    } // loop over LineSegment matls
  } // patches
}

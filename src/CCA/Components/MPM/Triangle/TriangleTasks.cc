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
#include <CCA/Components/MPM/Triangle/TriangleTasks.h>
#include <CCA/Components/MPM/Triangle/TriangleMaterial.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Core/TriangleLabel.h>
#include <CCA/Components/MPM/Core/MPMBoundCond.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Output.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/LinearInterpolator.h>
#include <Core/Grid/fastCpdiInterpolator.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Util/ProgressiveWarning.h>

#include <dirent.h>
#include <iostream>
#include <fstream>

using namespace Uintah;
using namespace std;

static DebugStream cout_doing("MPM", false);

//______________________________________________________________________
TriangleTasks::TriangleTasks(MaterialManagerP& ss, MPMFlags* flags, 
                                                   Output* output)
{
  lb = scinew MPMLabel();
  TriL= scinew TriangleLabel();

  d_flags = flags;
  m_output = output;

  if(flags->d_8or27==8){
    NGP=1;
    NGN=1;
  } else{
    NGP=2;
    NGN=2;
  }

  d_materialManager = ss;
}

TriangleTasks::~TriangleTasks()
{
  delete lb;
  delete TriL;
}

void TriangleTasks::triangleProblemSetup(const ProblemSpecP& prob_spec, 
                                         MPMFlags* flags)
{
  //Search for the MaterialProperties block and then get the MPM section
  ProblemSpecP mat_ps =  
    prob_spec->findBlockWithOutAttribute("MaterialProperties");
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");
  for (ProblemSpecP ps = mpm_mat_ps->findBlock("Triangle"); ps != nullptr;
       ps = ps->findNextBlock("Triangle") ) {

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

    //Create and register as an Triangle material
    TriangleMaterial *mat = 
                       scinew TriangleMaterial(ps, d_materialManager, flags);

    mat->registerParticleState( d_triangleState,
                                d_triangleState_preReloc );

    // When doing restart, we need to make sure that we load the materials
    // in the same order that they were initially created.  Restarts will
    // ALWAYS have an index number as in <material index = "0">.
    // Index_val = -1 means that we don't register the material by its 
    // index number.
    if (index_val > -1){
      d_materialManager->registerMaterial("Triangle", mat,index_val);
    }
    else{
      d_materialManager->registerMaterial("Triangle", mat);
    }
  }
}

void TriangleTasks::scheduleUpdateTriangles(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSubset* mpm_matls,
                                           const MaterialSubset* triangle_matls,
                                           const MaterialSet* matls)
{
  if (!d_flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                             getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"TriangleTasks::scheduleUpdateTriangles");

  Task* t=scinew Task("TriangleTasks::updateTriangles",
                      this, &TriangleTasks::updateTriangles);

  t->requires(Task::OldDW, lb->delTLabel );
  t->requires(Task::OldDW, lb->timeStepLabel);

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;

  t->requires(Task::NewDW, lb->gVelocityStarLabel,   mpm_matls,     gac,NGN+2);
  t->requires(Task::NewDW, lb->gMassLabel,           mpm_matls,     gac,NGN+2);
  t->requires(Task::NewDW, lb->dLdtDissolutionLabel, mpm_matls,     gac,NGN+2);
  if (d_flags->d_doingDissolution) {
    t->requires(Task::NewDW, lb->gSurfNormLabel,     mpm_matls,     gac,NGN+2);
  }
  t->requires(Task::NewDW, lb->gMassLabel,
             d_materialManager->getAllInOneMatls(),Task::OutOfDomain,gac,NGN+2);
  t->requires(Task::NewDW, lb->gVelocityLabel,
             d_materialManager->getAllInOneMatls(),Task::OutOfDomain,gac,NGN+2);

  t->requires(Task::OldDW, lb->pXLabel,                  triangle_matls, gnone);
  t->requires(Task::OldDW, lb->pSizeLabel,               triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triangleIDLabel,        triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triMidToN0VectorLabel,  triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triMidToN1VectorLabel,  triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triMidToN2VectorLabel,  triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triUseInPenaltyLabel,   triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triAreaLabel,           triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triAreaAtNodesLabel,    triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triClayLabel,           triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triMassDispLabel,       triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triCementThicknessLabel,triangle_matls, gnone);
  t->requires(Task::OldDW, TriL->triNearbyMatsLabel,     triangle_matls, gnone);

  t->computes(lb->pXLabel_preReloc,                      triangle_matls);
  t->computes(lb->pSizeLabel_preReloc,                   triangle_matls);
  t->computes(TriL->triangleIDLabel_preReloc,            triangle_matls);
  t->computes(TriL->triMidToN0VectorLabel_preReloc,      triangle_matls);
  t->computes(TriL->triMidToN1VectorLabel_preReloc,      triangle_matls);
  t->computes(TriL->triMidToN2VectorLabel_preReloc,      triangle_matls);
  t->computes(TriL->triUseInPenaltyLabel_preReloc,       triangle_matls);
  t->computes(TriL->triAreaLabel_preReloc,               triangle_matls);
  t->computes(TriL->triAreaAtNodesLabel_preReloc,        triangle_matls);
  t->computes(TriL->triClayLabel_preReloc,               triangle_matls);
  t->computes(TriL->triMassDispLabel_preReloc,           triangle_matls);
  t->computes(TriL->triCementThicknessLabel_preReloc,    triangle_matls);
  t->computes(TriL->triNormalLabel_preReloc,             triangle_matls);
  t->computes(TriL->triNearbyMatsLabel_preReloc,         triangle_matls);

  // Reduction Variable
  t->computes(lb->TotalSurfaceAreaLabel);

  sched->addTask(t, patches, matls);
}

void TriangleTasks::updateTriangles(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* ,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing updateTriangles");

    ParticleInterpolator* interpolator=scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );
    Ghost::GhostType  gac = Ghost::AroundCells;

    timeStep_vartype timeStep;
    old_dw->get(timeStep, lb->timeStepLabel);
    int timestep = timeStep;

    // Should we make this an input file parameter?
    int interval=10;

    int doit=timestep%interval;

    unsigned int numMPMMatls=d_materialManager->getNumMatls("MPM");
    std::vector<constNCVariable<Vector> > gvelocity(numMPMMatls);
    std::vector<constNCVariable<double> > gmass(numMPMMatls);
    std::vector<constNCVariable<double> > dLdt(numMPMMatls);
    std::vector<constNCVariable<Vector> > gSurfNorm(numMPMMatls);
    std::vector<bool> PistonMaterial(numMPMMatls);

    constNCVariable<Vector>  gvelocityglobal;
    constNCVariable<double>  gmassglobal;
    new_dw->get(gmassglobal,  lb->gMassLabel,
           d_materialManager->getAllInOneMatls()->get(0), patch, gac, NGN+2);
    new_dw->get(gvelocityglobal,  lb->gVelocityLabel,
           d_materialManager->getAllInOneMatls()->get(0), patch, gac, NGN+2);

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl=(MPMMaterial*) 
                                     d_materialManager->getMaterial("MPM",m);
      int dwi = mpm_matl->getDWIndex();
      new_dw->get(gvelocity[m], lb->gVelocityStarLabel,  dwi, patch, gac,NGN+2);
      new_dw->get(gmass[m],     lb->gMassLabel,          dwi, patch, gac,NGN+2);
      new_dw->get(dLdt[m],      lb->dLdtDissolutionLabel,dwi, patch, gac,NGN+2);
      PistonMaterial[m] = mpm_matl->getIsPistonMaterial();

      if (d_flags->d_doingDissolution){
        new_dw->get(gSurfNorm[m],lb->gSurfNormLabel,     dwi, patch, gac,NGN+2);
      } else{
        NCVariable<Vector> gSN_create;
        new_dw->allocateTemporary(gSN_create,                 patch, gac,NGN+2);
        gSN_create.initialize(Vector(0.));
        gSurfNorm[m] = gSN_create;                     // reference created data
      }
    }

    int numLSMatls=d_materialManager->getNumMatls("Triangle");
    for(int ls = 0; ls < numLSMatls; ls++){
      TriangleMaterial* ls_matl = (TriangleMaterial *) 
                              d_materialManager->getMaterial("Triangle", ls);
      int dwi = ls_matl->getDWIndex();

      int adv_matl = ls_matl->getAssociatedMaterial();

      // Not populating the delset, but we need this to satisfy Relocate
      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> tx;
      ParticleVariable<Point> tx_new;
      constParticleVariable<Matrix3> tsize;
      ParticleVariable<Matrix3> tsize_new;
      constParticleVariable<long64> triangle_ids;
      ParticleVariable<long64> tri_ids_new;
      constParticleVariable<Vector> triMidToN0Vec, triMidToN1Vec, triMidToN2Vec;
      ParticleVariable<Vector> triMidToN0Vec_new, 
                               triMidToN1Vec_new, triMidToN2Vec_new;
      constParticleVariable<IntVector> triUseInPenalty;
      ParticleVariable<IntVector>      triUseInPenalty_new;
      constParticleVariable<double> triArea, triClay, triMassDisp, triCemThick;
      ParticleVariable<double>      triArea_new, triClay_new, triMassDisp_new;
      ParticleVariable<double>      triCemThick_new;
      constParticleVariable<Vector> triAreaAtNodes;
      constParticleVariable<Matrix3> triNearbyMats;
      ParticleVariable<Vector>      triAreaAtNodes_new, triNormal_new;
      ParticleVariable<Matrix3>     triNearbyMats_new;

      old_dw->get(tx,              lb->pXLabel,                         pset);
      old_dw->get(tsize,           lb->pSizeLabel,                      pset);
      old_dw->get(triangle_ids,    TriL->triangleIDLabel,               pset);
      old_dw->get(triMidToN0Vec,   TriL->triMidToN0VectorLabel,         pset);
      old_dw->get(triMidToN1Vec,   TriL->triMidToN1VectorLabel,         pset);
      old_dw->get(triMidToN2Vec,   TriL->triMidToN2VectorLabel,         pset);
      old_dw->get(triUseInPenalty, TriL->triUseInPenaltyLabel,          pset);
      old_dw->get(triArea,         TriL->triAreaLabel,                  pset);
      old_dw->get(triAreaAtNodes,  TriL->triAreaAtNodesLabel,           pset);
      old_dw->get(triClay,         TriL->triClayLabel,                  pset);
      old_dw->get(triMassDisp,     TriL->triMassDispLabel,              pset);
      old_dw->get(triCemThick,     TriL->triCementThicknessLabel,       pset);
      old_dw->get(triNearbyMats,   TriL->triNearbyMatsLabel,            pset);

      new_dw->allocateAndPut(tx_new,         lb->pXLabel_preReloc,        pset);
      new_dw->allocateAndPut(tsize_new,      lb->pSizeLabel_preReloc,     pset);
      new_dw->allocateAndPut(tri_ids_new,  TriL->triangleIDLabel_preReloc,pset);
      new_dw->allocateAndPut(triMidToN0Vec_new,
                                  TriL->triMidToN0VectorLabel_preReloc,   pset);
      new_dw->allocateAndPut(triMidToN1Vec_new,
                                  TriL->triMidToN1VectorLabel_preReloc,   pset);
      new_dw->allocateAndPut(triMidToN2Vec_new,
                                  TriL->triMidToN2VectorLabel_preReloc,   pset);
      new_dw->allocateAndPut(triUseInPenalty_new,
                                   TriL->triUseInPenaltyLabel_preReloc,   pset);
      new_dw->allocateAndPut(triArea_new,
                                   TriL->triAreaLabel_preReloc,           pset);
      new_dw->allocateAndPut(triAreaAtNodes_new,
                                   TriL->triAreaAtNodesLabel_preReloc,    pset);
      new_dw->allocateAndPut(triClay_new,
                                   TriL->triClayLabel_preReloc,           pset);
      new_dw->allocateAndPut(triMassDisp_new,
                                   TriL->triMassDispLabel_preReloc,       pset);
      new_dw->allocateAndPut(triCemThick_new,
                                   TriL->triCementThicknessLabel_preReloc,pset);
      new_dw->allocateAndPut(triNormal_new,
                                   TriL->triNormalLabel_preReloc,         pset);
      new_dw->allocateAndPut(triNearbyMats_new,
                                   TriL->triNearbyMatsLabel_preReloc,     pset);

      tri_ids_new.copyData(triangle_ids);
      triAreaAtNodes_new.copyData(triAreaAtNodes);
      triUseInPenalty_new.copyData(triUseInPenalty);
      triClay_new.copyData(triClay);
      triMassDisp_new.copyData(triMassDisp);
      triNearbyMats_new.copyData(triNearbyMats);
      triCemThick_new.copyData(triCemThick);
      tsize_new.copyData(tsize);  // This isn't really used

      double totalsurfarea = 0.;

      // Loop over triangles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        Point P[3];
        // Update the positions of the triangle vertices
        P[0] = tx[idx] + triMidToN0Vec[idx];
        P[1] = tx[idx] + triMidToN1Vec[idx];
        P[2] = tx[idx] + triMidToN2Vec[idx];
        // Keep track of how much of the triangle's motion is due to mass change
        Vector surf[3] = {Vector(0.0),Vector(0.0),Vector(0.0)};
 
        // Loop over the vertices
        int deleteThisTriangle = 0;
        Vector vertexVel[3];
        double populatedVertex[3]={0.,0.,0.};
        double DisPrecip = 0.;  // Dissolving if > 0, precipitating if < 0.
        IntVector negnn(-99,-99,-99);
        IntVector matls[3]={negnn,negnn,negnn};

        for(int itv = 0; itv < 3; itv++){
          // Get the node indices that surround the point
          int NN = interpolator->findCellAndWeights(P[itv], ni, S, tsize[idx]);
          Vector vel(0.0,0.0,0.0);
          Vector velGlobal(0.0,0.0,0.0);
          double sumSk=0.0;
          double sumSkGlobal=0.0;
          Vector gSN(0.,0.,0.);
          vector< std::pair <double,int> > matlMass(numMPMMatls);
          // matlMass is the mass of other materials near the point
          // This is to limit the number of materials we search through
          // to find intersections.
          for(unsigned int m = 0; m < numMPMMatls; m++){
             matlMass[m].first=0.0;
          }
          // Accumulate the contribution from each surrounding vertex
          for (int k = 0; k < NN; k++) {
            IntVector node = ni[k];
            vel         += gvelocity[adv_matl][node]*gmass[adv_matl][node]*S[k];
            sumSk       += gmass[adv_matl][node]*S[k];
            velGlobal   += gvelocityglobal[node]*gmassglobal[node]*S[k];
            sumSkGlobal += gmassglobal[node]*S[k];
            surf[itv]   -= dLdt[adv_matl][node]*gSurfNorm[adv_matl][node]*S[k];
            gSN         += gSurfNorm[adv_matl][node]*S[k];
            DisPrecip += dLdt[adv_matl][node]*S[k];
          }
          
          if(doit==1){
            // Find which other materials have the largest
            // mass near the current node so we can only check
            // those in looking for intersections
            if(triUseInPenalty[idx](itv)==1){
              for (int k = 0; k < NN; k++) {
                IntVector node = ni[k];
                // skip the node's own material
                for(unsigned int m = 0; m < adv_matl; m++){
                  matlMass[m].first += gmass[m][node]*S[k];
                  matlMass[m].second = m;
                }
                for(unsigned int m = adv_matl+1; m < numMPMMatls; m++){
                  matlMass[m].first += gmass[m][node]*S[k];
                  matlMass[m].second = m;
                }
              } // loop over grid nodes
              sort(matlMass.rbegin(), matlMass.rend());

              // If any of the three top mass materials are zero, don't
              // include them in the materials to be searched.
              for(int im=0; im<3; im++){
                if(matlMass[im].first < 1.e-199){
                   matlMass[im].second = -99;
                }
              }

              // Only going to look at the two most likely materials
              matls[itv]=IntVector(matlMass[0].second,  
                                   matlMass[1].second,  
                                   -99 /*matlMass[2].second*/);
            }   // if a vertex to be used in penalty contact
          }

          if(sumSk > 1.e-90){
            // This is the normal condition, when at least one of the nodes
            // influencing a vertex has mass on it.
            vel/=sumSk;
            P[itv] += vel*delT;
            surf[itv]/=(gSN.length()+1.e-100);
            P[itv] += surf[itv]*delT;
            vertexVel[itv] = vel + surf[itv];
            populatedVertex[itv] = 1.;
          } else {
            if(sumSkGlobal > 1.e-90){
              velGlobal/=sumSkGlobal;
              P[itv] += vel*delT;
            } else {
              deleteThisTriangle++;
            }
          }
        } // loop over vertices

        if(doit==1){
          triNearbyMats_new[idx](0,0)=matls[0].x();
          triNearbyMats_new[idx](0,1)=matls[0].y();
          //triNearbyMats_new[idx](0,2)=matls[0].z();
          triNearbyMats_new[idx](1,0)=matls[1].x();
          triNearbyMats_new[idx](1,1)=matls[1].y();
          //triNearbyMats_new[idx](1,2)=matls[1].z();
          triNearbyMats_new[idx](2,0)=matls[2].x();
          triNearbyMats_new[idx](2,1)=matls[2].y();
          //triNearbyMats_new[idx](2,2)=matls[2].z();
        }

        if(DisPrecip <=0 && !PistonMaterial[adv_matl]){
          totalsurfarea+=triArea[idx];
        }

        // Handle the triangles that have vertices that are 
        // not near nodes with mass
        if(deleteThisTriangle==3){
          cout << "NOTICE: Deleting " << triangle_ids[idx] << " of group " 
               << adv_matl << " at position " << tx[idx] 
               << " because none of its vertices are getting any nodal input." 
               << endl; 
          delset->addParticle(idx);
        } else if(deleteThisTriangle>0){
          Vector velMean(0.);
          double populatedVertices=0.;
          for(int itv = 0; itv < 3; itv++){
            velMean += vertexVel[itv]*populatedVertex[itv];
            populatedVertices+=populatedVertex[itv]; 
          } // loop over vertices
          velMean/=populatedVertices;
          for(int itv = 0; itv < 3; itv++){
            P[itv] += velMean*(1. - populatedVertex[itv])*delT;
          } // loop over vertices
        }

        tx_new[idx] = (P[0]+P[1]+P[2])/3.;
        Vector triNorm = Cross(P[1]-P[0],P[2]-P[0]);
        double triNormLength = triNorm.length()+1.e-100;
        triArea_new[idx]=0.5*triNormLength;
        triNormal_new[idx]=triNorm/triNormLength;
        double tMD = Dot(triNormal_new[idx],(surf[0]+surf[1]+surf[2])*delT/3.);
        triMassDisp_new[idx] += tMD;
        triCemThick_new[idx] += std::max(0.0, tMD);

        triMidToN0Vec_new[idx] = P[0] - tx_new[idx];
        triMidToN1Vec_new[idx] = P[1] - tx_new[idx];
        triMidToN2Vec_new[idx] = P[2] - tx_new[idx];
#if 0
        // No point in updating size unless it is used.  Just carry forward.
        Vector r0 = P[1] - P[0];
        Vector r1 = P[2] - P[0];
        Vector r2 = 0.1*Cross(r1,r0);
        Matrix3 size =Matrix3(r0.x()/dx.x(), r1.x()/dx.x(), r2.x()/dx.x(),
                              r0.y()/dx.y(), r1.y()/dx.y(), r2.y()/dx.y(),
                              r0.z()/dx.z(), r1.z()/dx.z(), r2.z()/dx.z());
        tsize_new[idx] = size;
#endif

      } // Loop over triangles
      new_dw->deleteParticles(delset);

      new_dw->put(sum_vartype(totalsurfarea),      lb->TotalSurfaceAreaLabel);

#if 0
      // This is for computing updated triAreaAtNodes. Need to create a
      // container to replace the modified Stencil7 that I used in a hack here
      // Loop over triangles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Hit each vertex of the triangle
        double area0=0., area1=0., area2=0.;

        // Vertex 0
        for(int itri=0; itri < triNode0TriIDs[idx][29]; itri++) {
          int triID = triNode0TriIDs[idx][itri];
          // Inner Loop over triangles
          for(ParticleSubset::iterator jter = pset->begin();
              jter != pset->end(); jter++){
            particleIndex jdx = *jter;
            if(triID == triangle_ids[jdx]){
              area0+=triArea_new[jdx];
              break;
            } // if IDs are equal
          } // inner loop over triangles
        }
        area0/=3.;

        // Vertex 1
        for(int itri=0; itri < triNode1TriIDs[idx][29]; itri++) {
          int triID = triNode1TriIDs[idx][itri];
          // Inner Loop over triangles
          for(ParticleSubset::iterator jter = pset->begin();
              jter != pset->end(); jter++){
            particleIndex jdx = *jter;
            if(triID == triangle_ids[jdx]){
              area1+=triArea_new[jdx];
              break;
            } // if IDs are equal
          } // inner loop over triangles
        }
        area1/=3.;

        // Vertex 2
        for(int itri=0; itri < triNode2TriIDs[idx][29]; itri++) {
          int triID = triNode2TriIDs[idx][itri];
          // Inner Loop over triangles
          for(ParticleSubset::iterator jter = pset->begin();
              jter != pset->end(); jter++){
            particleIndex jdx = *jter;
            if(triID == triangle_ids[jdx]){
              area2+=triArea_new[jdx];
              break;
            } // if IDs are equal
          } // inner loop over triangles
        }
        area2/=3.;

        triAreaAtNodes_new[idx]=Vector(area0, area1, area2);
      } // Outer loop over triangles for vertex area calculation
#endif
    }  // matls

    delete interpolator;
  }    // patches
}

void TriangleTasks::scheduleComputeTriangleForces(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSubset* mpm_matls,
                                           const MaterialSubset* triangle_matls,
                                           const MaterialSet* matls)
{
  if (!d_flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,
                                "TriangleTasks::scheduleComputeTriangleForces");

  Task* t=scinew Task("TriangleTasks::computeTriangleForces",
                      this, &TriangleTasks::computeTriangleForces);

  Ghost::GhostType  gac = Ghost::AroundCells;

  t->requires(Task::OldDW, lb->simulationTimeLabel);
  t->requires(Task::OldDW, lb->pXLabel,                triangle_matls, gac, 2);
  t->requires(Task::OldDW, lb->pSizeLabel,             triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triMidToN0VectorLabel,triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triMidToN1VectorLabel,triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triMidToN2VectorLabel,triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triUseInPenaltyLabel, triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triangleIDLabel,      triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triAreaAtNodesLabel,  triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triClayLabel,         triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triNearbyMatsLabel,   triangle_matls, gac, 2);
  if (d_flags->d_doingDissolution) {
    t->requires(Task::OldDW, TriL->triMassDispLabel,   triangle_matls, gac, 2);
    t->requires(Task::OldDW, TriL->triCementThicknessLabel,
                                                       triangle_matls, gac, 2);
  }

  t->requires(Task::NewDW, lb->gMassLabel,             mpm_matls,   gac,NGN+3);

  t->computes(lb->gLSContactForceLabel,                mpm_matls);
  t->computes(lb->gInContactMatlLabel,                 mpm_matls);
  if (d_flags->d_doingDissolution) {
    t->computes(lb->gSurfaceAreaLabel,                 mpm_matls);
    t->computes(lb->gSurfaceClayLabel,                 mpm_matls);
    t->computes(lb->gSurfaceCementLabel,               mpm_matls);
  }
//  t->computes(TriL->triInContactLabel,                triangle_matls);

  sched->addTask(t, patches, matls);
}

void TriangleTasks::computeTriangleForces(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* ,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,
              "Doing computeTriangleForces");

    ParticleInterpolator* interpolator=scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    Ghost::GhostType gac = Ghost::AroundCells;
    Vector dxCell = patch->dCell();
    double cell_length2 = dxCell.length2();

    unsigned int numMPMMatls=d_materialManager->getNumMatls( "MPM" );
    std::vector<NCVariable<Vector> > LSContForce(numMPMMatls);
    std::vector<constNCVariable<double> > gmass(numMPMMatls);
    std::vector<NCVariable<double> > SurfArea(numMPMMatls);
    std::vector<NCVariable<double> > SurfClay(numMPMMatls);
    std::vector<NCVariable<double> > SurfCeme(numMPMMatls);
    std::vector<NCVariable<int> > InContactMatl(numMPMMatls);
    std::vector<double> stiffness(numMPMMatls);
    std::vector<bool> PistonMaterial(numMPMMatls);
//    std::vector<Vector> sumTriForce(numMPMMatls);
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                     (MPMMaterial*) d_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      PistonMaterial[m] = mpm_matl->getIsPistonMaterial();

      double inv_stiff = mpm_matl->getConstitutiveModel()->getCompressibility();
      stiffness[m] = 1./inv_stiff;

      new_dw->allocateAndPut(LSContForce[m],lb->gLSContactForceLabel,dwi,patch);
      LSContForce[m].initialize(Vector(0.0));
      new_dw->allocateAndPut(InContactMatl[m],lb->gInContactMatlLabel,dwi,patch);
      InContactMatl[m].initialize(-99);

      if (d_flags->d_doingDissolution) {
        new_dw->allocateAndPut(SurfArea[m], lb->gSurfaceAreaLabel,   dwi,patch);
        new_dw->allocateAndPut(SurfClay[m], lb->gSurfaceClayLabel,   dwi,patch);
        new_dw->allocateAndPut(SurfCeme[m], lb->gSurfaceCementLabel, dwi,patch);
        SurfArea[m].initialize(0.0);
        SurfClay[m].initialize(0.0);
        SurfCeme[m].initialize(0.0);
      }

      new_dw->get(gmass[m],                 lb->gMassLabel,          dwi,patch,
                                                                     gac,NGN+3);
//      sumTriForce[m]=Vector(0.0);
    }

    int numLSMatls=d_materialManager->getNumMatls("Triangle");

    std::vector<constParticleVariable<Point>  >  tx0(numLSMatls);
    std::vector<constParticleVariable<Vector>  > triMidToN0Vec(numLSMatls);
    std::vector<constParticleVariable<Vector>  > triMidToN1Vec(numLSMatls);
    std::vector<constParticleVariable<Vector>  > triMidToN2Vec(numLSMatls);
//    std::vector<ParticleVariable<int>  >         triInContact(numLSMatls);

    std::vector<std::vector<constParticleVariable<Vector>  > >
                                                    triMidToNodeVec(numLSMatls);

    std::vector<constParticleVariable<long64> >     triangle_ids(numLSMatls);
    std::vector<constParticleVariable<double> >     triClay(numLSMatls);
    std::vector<constParticleVariable<IntVector> >  triUseInPenalty(numLSMatls);
    std::vector<constParticleVariable<Vector> >     triAreaAtNodes(numLSMatls);
    std::vector<constParticleVariable<double> >     triMassDisp(numLSMatls);
    std::vector<constParticleVariable<double> >     triCemThick(numLSMatls);
    std::vector<constParticleVariable<Matrix3> >    triNearbyMats(numLSMatls);
    std::vector<ParticleSubset*> psetvec;
    std::vector<int> psetSize(numLSMatls);
//    std::vector<std::vector<int> > triInContact(numLSMatls);
    Matrix3 size; size.Identity();

    FILE* fp;
    if(m_output->isOutputTimeStep()){
      timeStep_vartype timeStep;
      old_dw->get(timeStep, lb->timeStepLabel);
      int timestep = timeStep;

      string udaDir = m_output->getOutputLocation();
      ostringstream tname;
      tname << setw(5) << setfill('0') << timestep;
      string tnames = tname.str();
      string pPath = udaDir + "/results_contacts";
      DIR *check = opendir(pPath.c_str());
      if ( check == nullptr ) {
        MKDIR( pPath.c_str(), 0777 );
      } else {
        closedir(check);
      }

      stringstream pnum;
      pnum << patch->getID();
      string pnums = pnum.str();
      string fname = pPath + "/TriContact." + pnums + "." + tnames;
      fp = fopen(fname.c_str(), "w");
    }

    for(int tmo = 0; tmo < numLSMatls; tmo++) {
      TriangleMaterial* t_matl0 = (TriangleMaterial *) 
                             d_materialManager->getMaterial("Triangle", tmo);
      int dwi0 = t_matl0->getDWIndex();

      ParticleSubset* pset0 = old_dw->getParticleSubset(dwi0, patch,
                                                        gac, 2, lb->pXLabel);
      psetvec.push_back(pset0);
      psetSize[tmo]=(pset0->end() - pset0->begin());
//      triInContact[tmo].resize(psetSize[tmo]);

      old_dw->get(tx0[tmo],            lb->pXLabel,                   pset0);
      old_dw->get(triMidToN0Vec[tmo],  TriL->triMidToN0VectorLabel,   pset0);
      old_dw->get(triMidToN1Vec[tmo],  TriL->triMidToN1VectorLabel,   pset0);
      old_dw->get(triMidToN2Vec[tmo],  TriL->triMidToN2VectorLabel,   pset0);
      old_dw->get(triUseInPenalty[tmo],TriL->triUseInPenaltyLabel,    pset0);
      old_dw->get(triAreaAtNodes[tmo], TriL->triAreaAtNodesLabel,     pset0);
      old_dw->get(triangle_ids[tmo],   TriL->triangleIDLabel,         pset0);
      old_dw->get(triClay[tmo],        TriL->triClayLabel,            pset0);
      old_dw->get(triNearbyMats[tmo],  TriL->triNearbyMatsLabel,      pset0);

      if (d_flags->d_doingDissolution) {
        old_dw->get(triMassDisp[tmo],  TriL->triMassDispLabel,        pset0);
        old_dw->get(triCemThick[tmo],  TriL->triCementThicknessLabel, pset0);
      } else {
        ParticleVariable<double>   triMassDisp_tmp;
        new_dw->allocateTemporary(triMassDisp_tmp,  pset0);
        for(ParticleSubset::iterator iter0 = pset0->begin();
            iter0 != pset0->end(); iter0++){
          particleIndex idx0 = *iter0;
          triMassDisp_tmp[idx0] = 0.;
        }
        triMassDisp[tmo]=triMassDisp_tmp;
      }

      triMidToNodeVec[tmo].push_back(triMidToN0Vec[tmo]);
      triMidToNodeVec[tmo].push_back(triMidToN1Vec[tmo]);
      triMidToNodeVec[tmo].push_back(triMidToN2Vec[tmo]);
//      new_dw->allocateAndPut(triInContact[tmo],TriL->triInContactLabel,pset0);
//      for(ParticleSubset::iterator iter0 = pset0->begin();
//          iter0 != pset0->end(); iter0++){
//        particleIndex idx0 = *iter0;
//        triInContact[tmo][idx0] = -1;
//      }
    } // end loop over triangle materials to get data from DW

    int numOverlap=0;
    int numInside=0;
    double totalContactArea    = 0.0;
    double totalContactAreaTri = 0.0;
    Vector totalForce(0.);
    double timefactor=1.0;
//    double timefactor=min(1.0, time/1.0);
//    proc0cout << "timefactor = " << timefactor << endl;

    for(int tmo = 0; tmo < numLSMatls; tmo++) {
      TriangleMaterial* t_matl0 = (TriangleMaterial *) 
                             d_materialManager->getMaterial("Triangle", tmo);
      int adv_matl0 = t_matl0->getAssociatedMaterial();

      ParticleSubset* pset0 = psetvec[tmo];

      // Extrapolate area of line segments to the grid for use in dissolution
      if (d_flags->d_doingDissolution){
       for(ParticleSubset::iterator iter0 = pset0->begin();
           iter0 != pset0->end(); iter0++){
         particleIndex idx0 = *iter0;

         Point vert[3];

         vert[0] = tx0[tmo][idx0] + triMidToN0Vec[tmo][idx0];
         vert[1] = tx0[tmo][idx0] + triMidToN1Vec[tmo][idx0];
         vert[2] = tx0[tmo][idx0] + triMidToN2Vec[tmo][idx0];
         Vector BA = vert[1]-vert[0];
         Vector CA = vert[2]-vert[0];
         double thirdTriArea = 0.5*Cross(BA,CA).length()/3.;

         for(int itv = 0; itv < 3; itv++){
           int nn = interpolator->findCellAndWeights(vert[itv], ni, S, size);
           double totMass = 0.;
           for (int k = 0; k < nn; k++) {
             IntVector node = ni[k];
             totMass += S[k]*gmass[adv_matl0][node];
           }

           for (int k = 0; k < nn; k++) {
             IntVector node = ni[k];
             if(patch->containsNode(node)) {
               double sArea = thirdTriArea*S[k]*gmass[adv_matl0][node]/totMass;
               SurfArea[adv_matl0][node] += sArea;
               SurfClay[adv_matl0][node] += sArea*triClay[tmo][idx0];
               SurfCeme[adv_matl0][node] += sArea*triCemThick[tmo][idx0];
             }
           }
         }
       } // loop over all triangles

       // Now loop over the nodes and normalize SurfClay by the area
       for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
         IntVector c = *iter;
         SurfClay[adv_matl0][c]/=(SurfArea[adv_matl0][c]+1.e-100);
         SurfCeme[adv_matl0][c]/=(SurfArea[adv_matl0][c]+1.e-100);
       } // loop over all nodes
      }   // only do this if a dissolution problem

      for(int tmi = tmo+1; tmi < numLSMatls; tmi++) {
       TriangleMaterial* t_matl1 = (TriangleMaterial *) 
                             d_materialManager->getMaterial("Triangle",tmi);
       int adv_matl1 = t_matl1->getAssociatedMaterial();

       if(adv_matl0==adv_matl1 || 
          (PistonMaterial[adv_matl0] && PistonMaterial[adv_matl1])){
         continue;
       }

       ParticleSubset* pset1 = psetvec[tmi];

       int numPar_pset1 = pset1->numParticles();

       double K_l = 10.*(stiffness[adv_matl0] * stiffness[adv_matl1])/
                        (stiffness[adv_matl0] + stiffness[adv_matl1]);
       K_l*=timefactor;

       if(numPar_pset1 > 0){

        // Loop over zeroth triangle subset
        // Then loop over the vertices of the triangle
        // Check to see if they are to be "used" in force
        // calculation.  Every vertex should only be used once.
        for(ParticleSubset::iterator iter0 = pset0->begin();
            iter0 != pset0->end(); iter0++){
          particleIndex idx0 = *iter0;

         for(int iu = 0; iu < 3; iu++){

          if(triUseInPenalty[tmo][idx0](iu)==0 ||
            ((int) triNearbyMats[tmo][idx0](iu,0) != adv_matl1 &&
             (int) triNearbyMats[tmo][idx0](iu,1) != adv_matl1/* &&
             (int) triNearbyMats[tmo][idx0](iu,2) != adv_matl1*/)){
            continue;
          }

          Point px0=tx0[tmo][idx0] + triMidToNodeVec[tmo][iu][idx0];

          // Assume the normal at the point is the same as the triangle
          // with which the point is associated.  Ideally, we would use all
          // triangles associated with the point to compute a weighted average
          // but the data structures here won't allow it.
          Vector ptNormal =Cross(triMidToN0Vec[tmo][idx0],
                                 triMidToN1Vec[tmo][idx0]);
          double pNL = ptNormal.length();
          if(pNL>0.0){
            ptNormal /= pNL;
          }

          bool foundOne = false;
          vector<double> triSep;
          vector<int> triIndex;
          vector<double> triOverlap;
          vector<Point> triInPlane;
          vector<Vector> triTriNormal;
          // Loop over other triangle subset to find triangles that are
          // first "near" the point, and then, within those, to find triangles
          // whose planes are penetrated by the point.
          for(ParticleSubset::iterator iter1 = pset1->begin();
              iter1 != pset1->end(); iter1++){
            particleIndex idx1 = *iter1;
            // AP is a vector from the centroid of the test triangle
            // to the test point px0 
            Vector AP = px0 - tx0[tmi][idx1];
            double sep = AP.length2();
            // check to see if the triangle is even in the neighborhood
            // of the test point
            if(sep < 0.25*cell_length2){
              Vector triNormal =Cross(triMidToN0Vec[tmi][idx1],
                                      triMidToN1Vec[tmi][idx1]);
              double tNL = triNormal.length();
              if(tNL>0.0){
                triNormal /= tNL;
              }
              double overlap = Dot(AP,triNormal);
              // The first conditional means the point is past the plane
              // of the triangle.  The second is attempting to prevent
              // detecting overlaps where "thin" objects are present,
              // namely, making sure that the ptNormal and triNormal are
              // pointing in substantially different directions.
              if(overlap < 0.0 && Dot(ptNormal,triNormal) < -.2){
                // Point is past the plane of the triangle
                numOverlap++;
                triSep.push_back(sep);
                triIndex.push_back(idx1);
                triOverlap.push_back(overlap);
                triTriNormal.push_back(triNormal);
                Point inPlane = px0 - overlap*triNormal;
                triInPlane.push_back(inPlane);
              }    // Point px0 overlaps plane of current triangle
            }  // point is in the neighborhood
          } // inner loop over triangles

          // Sort the triangles found above according to triSep, i.e.
          // the triangle centroid that is nearest the point px0 is
          // first, and so on.
          int aLength = triSep.size(); // initialise to a's length
          int numSorted = min(aLength, 6);

          /* advance the position through the entire array */
          for (int i = 0; i < numSorted-1; i++) {
            /* find the min element in the unsorted a[i .. aLength-1] */

            /* assume the min is the first element */
            int jMin = i;
            /* test against elements after i to find the smallest */
            for (int j = i+1; j < aLength; j++) {
              /* if this element is less, then it is the new minimum */
              if (triSep[j] < triSep[jMin]) {
                  /* found new minimum; remember its index */
                  jMin = j;
              }
            }

            if (jMin != i) {
              swap(triSep[i],        triSep[jMin]);
              swap(triIndex[i],      triIndex[jMin]);
              swap(triOverlap[i],    triOverlap[jMin]);
              swap(triInPlane[i],    triInPlane[jMin]);
              swap(triTriNormal[i],  triTriNormal[jMin]);
            }
          } // for loop over unsorted vector

          // Loop over all triangles that the point px0 overlaps
          // (from the sorted list found above)
          // Use the algorithm described in:
          // https://gdbooks.gitbooks.io/3dcollisions/content/Chapter4/point_in_triangle.html
          for (int i = 0; i < numSorted; i++) {
            //Now, see if that point is inside the triangle or not
            int vecIdx = triIndex[i];
            Point A = tx0[tmi][vecIdx] + triMidToN0Vec[tmi][vecIdx];
            Point B = tx0[tmi][vecIdx] + triMidToN1Vec[tmi][vecIdx];
            Point C = tx0[tmi][vecIdx] + triMidToN2Vec[tmi][vecIdx];
            Vector a = A - triInPlane[i];
            Vector b = B - triInPlane[i];
            Vector c = C - triInPlane[i];
            Vector u = Cross(b,c);
            Vector v = Cross(c,a);
            Vector w = Cross(a,b);

            // If the following conditional is true, this means that the
            // normals of the three triangle made using "triInPlane" and the
            // three vertices of the triangle all point in the same direction
            // Thus, the point is inside the triangle.  If they don't point
            // in the same direction, the point is outside the triangle
            if(Dot(u,v) >= 0. && Dot(u,w) >= 0.){
              numInside++;
//              triInContact[tmi][closest] = tmo;
              foundOne=true;
//              double Length=((C-B).length()+(B-A).length()+(A-C).length())/3.;
              double Length = sqrt(triAreaAtNodes[tmo][idx0][iu]);
              double K = K_l*Length;
              double forceMag = triOverlap[i]*K;
              // Find the weights for each of the vertices
              // These are actually twice the areas, doesn't matter, dividing
              double areaA = u.length();
              double areaB = v.length();
              double areaC = w.length();
              double totalArea = areaA+areaB+areaC;
              areaA/=totalArea;
              areaB/=totalArea;
              areaC/=totalArea;
              Vector tForceA  = -forceMag*triTriNormal[i]*areaA;
              Vector tForceB  = -forceMag*triTriNormal[i]*areaB;
              Vector tForceC  = -forceMag*triTriNormal[i]*areaC;
              totalContactArea += triAreaAtNodes[tmo][idx0][iu];
              totalContactAreaTri += 0.5*totalArea;

              if(m_output->isOutputTimeStep()){
                // triangle_ids[tmo][idx0] is the triangle that is penetrating
                // iu is the vertex of the penetrating triangle
                // triangle_ids[tmi][vecIdx] is the penetrated triangle
                 fprintf(fp,"%i %i %i %i %ld %ld %i %8.6e %8.6e %8.6e\n",
                 tmo, tmi, adv_matl0, adv_matl1,
                 triangle_ids[tmo][idx0], triangle_ids[tmi][vecIdx],
                 iu, 0.5*totalArea,
                 triAreaAtNodes[tmo][idx0][iu], triMassDisp[tmi][vecIdx]);
                 fflush(fp);
              }

//                cout << "triAreaAtNodes[" << tmo << "][" << idx0 << "][" << iu << "] = " << triAreaAtNodes[tmo][idx0][iu] << endl;
//                cout << "totalAreaA, closest  = " << 0.5*totalArea 
//                     << " "                      << closest << endl;
              totalForce += tForceA;
              totalForce += tForceB;
              totalForce += tForceC;

              // Distribute the force to the grid from the triangles
              // from the triangle vertices.  Use same spatial location
              // for both adv_matls

              // First for Point A
              // Get the node indices that surround the cell
              int NN = interpolator->findCellAndWeights(A, ni, S, size);

              double totMass0 = 0.; double totMass1 = 0.;
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
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 // Distribute force according to material mass on the nodes
                 // to get nearly equal contribution to the acceleration
                 LSContForce[adv_matl0][node] += tForceA*S[k]
                                           * gmass[adv_matl0][node]/totMass0;
                 LSContForce[adv_matl1][node] -= tForceA*S[k]
                                           * gmass[adv_matl1][node]/totMass1;
                 InContactMatl[adv_matl0][node] = adv_matl1;
                 InContactMatl[adv_matl1][node] = adv_matl0;
//               }
               }
              }

              // Next for Point B
              // Get the node indices that surround the cell
              NN = interpolator->findCellAndWeights(B, ni, S, size);

              totMass0 = 0.; totMass1 = 0.;
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
                 LSContForce[adv_matl0][node] += tForceB*S[k]
                                           * gmass[adv_matl0][node]/totMass0;
                 LSContForce[adv_matl1][node] -= tForceB*S[k]
                                           * gmass[adv_matl1][node]/totMass1;
                 InContactMatl[adv_matl0][node] = adv_matl1;
                 InContactMatl[adv_matl1][node] = adv_matl0;
//               }
               }
              }

              // Finally for Point C
              // Get the node indices that surround the cell
              NN = interpolator->findCellAndWeights(C, ni, S, size);

              totMass0 = 0.; totMass1 = 0.;
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
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 // Distribute force according to material mass on the nodes
                 // to get nearly equal contribution to the acceleration
                 LSContForce[adv_matl0][node] += tForceC*S[k]
                                           * gmass[adv_matl0][node]/totMass0;
                 LSContForce[adv_matl1][node] -= tForceC*S[k]
                                           * gmass[adv_matl1][node]/totMass1;
                 InContactMatl[adv_matl0][node] = adv_matl1;
                 InContactMatl[adv_matl1][node] = adv_matl0;
//               }
               }
              }
            }  // inPlane is inside triangle
            if(foundOne){
              break;
            }
          } // loop over overlapped triangles

          // Here, we consider possible "corner cases" in which the 
          // point px0 projected to the plane of the closest triangle
          // actually lands inside a neighboring triangle.  This 
          // possitiblity is fairly easy to illustrate in 2D with line segments.
          if(!foundOne && triIndex.size()>1){
            // check to see if "triInPlane[0]" is inside another triangle
            // that is overlapped
            for (int i = 1; i < numSorted; i++) {
              //Now, see if that point is inside the triangle or not
              int vecIdx = triIndex[i];
              Point A = tx0[tmi][vecIdx] + triMidToN0Vec[tmi][vecIdx];
              Point B = tx0[tmi][vecIdx] + triMidToN1Vec[tmi][vecIdx];
              Point C = tx0[tmi][vecIdx] + triMidToN2Vec[tmi][vecIdx];
              Vector a = A - triInPlane[0];
              Vector b = B - triInPlane[0];
              Vector c = C - triInPlane[0];
              Vector u = Cross(b,c);
              Vector v = Cross(c,a);
              Vector w = Cross(a,b);

              // If the following conditional is true, this means that the
              // normals of the three triangle made using "triInPlane" and the
              // three vertices of the triangle all point in the same direction
              // Thus, the point is inside the triangle.  If they don't point
              // in the same direction, the point is outside the triangle
              if(Dot(u,v) >= 0. && Dot(u,w) >= 0.){
                numInside++;
                foundOne=true;
                double Length=((C-B).length()+(B-A).length()+(A-C).length())/3.;
                double K = K_l*Length;
                double forceMag = triOverlap[i]*K;
                // Find the weights for each of the vertices
                // These are actually twice the areas, doesn't matter, dividing
                double areaA = u.length();
                double areaB = v.length();
                double areaC = w.length();
                double totalArea = areaA+areaB+areaC;
                areaA/=totalArea;
                areaB/=totalArea;
                areaC/=totalArea;
                Vector tForceA  = -forceMag*triTriNormal[i]*areaA;
                Vector tForceB  = -forceMag*triTriNormal[i]*areaB;
                Vector tForceC  = -forceMag*triTriNormal[i]*areaC;
                totalContactArea += triAreaAtNodes[tmo][idx0][iu];
                totalContactAreaTri += 0.5*totalArea;

                if(m_output->isOutputTimeStep()){
                 fprintf(fp,"%i %i %i %i %ld %ld %i %8.6e %8.6e %8.6e\n",
                 tmo, tmi, adv_matl0, adv_matl1,
                 triangle_ids[tmo][idx0], triangle_ids[tmi][vecIdx],
                 iu, 0.5*totalArea,
                 triAreaAtNodes[tmo][idx0][iu], triMassDisp[tmi][vecIdx]);
                 fflush(fp);
                }

//                cout << "triAreaAtNodes[" << tmo << "][" << idx0 << "][" << iu << "] = " << triAreaAtNodes[tmo][idx0][iu] << endl;
//                cout << "totalAreaA, closest  = " << 0.5*totalArea 
//                     << " "                      << closest << endl;
                totalForce += tForceA;
                totalForce += tForceB;
                totalForce += tForceC;

                // Distribute the force to the grid from the triangles
                // from the triangle vertices.  Use same spatial location
                // for both adv_matls
  
                // First for Point A
                // Get the node indices that surround the cell
                int NN = interpolator->findCellAndWeights(A, ni, S, size);
  
                double totMass0 = 0.; double totMass1 = 0.;
                for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 totMass0 += S[k]*gmass[adv_matl0][node];
                 totMass1 += S[k]*gmass[adv_matl1][node];
//                }
                }
  
                // Accumulate the contribution from each surrounding vertex
                for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
                 if(patch->containsNode(node)) {
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                   // Distribute force according to material mass on the nodes
                   // to get nearly equal contribution to the acceleration
                   LSContForce[adv_matl0][node] += tForceA*S[k]
                                             * gmass[adv_matl0][node]/totMass0;
                   LSContForce[adv_matl1][node] -= tForceA*S[k]
                                             * gmass[adv_matl1][node]/totMass1;
                   InContactMatl[adv_matl0][node] = adv_matl1;
                   InContactMatl[adv_matl1][node] = adv_matl0;
//                 }
                 }
                }
  
                // Next for Point B
                // Get the node indices that surround the cell
                NN = interpolator->findCellAndWeights(B, ni, S, size);
  
                totMass0 = 0.; totMass1 = 0.;
                for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 totMass0 += S[k]*gmass[adv_matl0][node];
                 totMass1 += S[k]*gmass[adv_matl1][node];
//                }
                }
  
                // Accumulate the contribution from each surrounding vertex
                for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
                 if(patch->containsNode(node)) {
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                   // Distribute force according to material mass on the nodes
                   // to get nearly equal contribution to the acceleration
                   LSContForce[adv_matl0][node] += tForceB*S[k]
                                             * gmass[adv_matl0][node]/totMass0;
                   LSContForce[adv_matl1][node] -= tForceB*S[k]
                                             * gmass[adv_matl1][node]/totMass1;
                   InContactMatl[adv_matl0][node] = adv_matl1;
                   InContactMatl[adv_matl1][node] = adv_matl0;
//                 }
                 }
                }
  
                // Finally for Point C
                // Get the node indices that surround the cell
                NN = interpolator->findCellAndWeights(C, ni, S, size);
  
                totMass0 = 0.; totMass1 = 0.;
                for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                 totMass0 += S[k]*gmass[adv_matl0][node];
                 totMass1 += S[k]*gmass[adv_matl1][node];
//                }
                }
  
                // Accumulate the contribution from each surrounding vertex
                for (int k = 0; k < NN; k++) {
                 IntVector node = ni[k];
                 if(patch->containsNode(node)) {
//               if(gmass[adv_matl0][node]>1.e-50 &&
//                  gmass[adv_matl1][node]>1.e-50){
                   // Distribute force according to material mass on the nodes
                   // to get nearly equal contribution to the acceleration
                   LSContForce[adv_matl0][node] += tForceC*S[k]
                                             * gmass[adv_matl0][node]/totMass0;
                   LSContForce[adv_matl1][node] -= tForceC*S[k]
                                             * gmass[adv_matl1][node]/totMass1;
                   InContactMatl[adv_matl0][node] = adv_matl1;
                   InContactMatl[adv_matl1][node] = adv_matl0;
//                 }
                 }
                }
              } // check dot products
              if(foundOne){
                break;
              }
            } // loop over other nearby triangles
          }  // If multiple overlaps, but penetration point not in triangles
         } // loop over the three vertices of the triangle
        } //  Outer loop over triangles
       }  // if num particles in the inner pset is > 0
      } // inner loop over triangle materials
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch, adv_matl0, "Symmetric", 
                              LSContForce[adv_matl0], "linear");
    } // outer loop over triangle materials

    delete interpolator;
    if(m_output->isOutputTimeStep()){
      fclose(fp);
    }
  } // patches
}

void
TriangleTasks::scheduleComputeTriangleScaleFactor(SchedulerP  & sched,
                                                  const PatchSet    * patches,
                                                  const MaterialSet * matls )
{
  if (!d_flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                             getLevel(patches)->getGrid()->numLevels())) {
    return;
  }

  printSchedule( patches, cout_doing,
                          "TriangleTasks::scheduleComputeTriangleScaleFactor");

  Task * t = scinew Task( "TriangleTasks::computeTriangleScaleFactor",this, 
                          &TriangleTasks::computeTriangleScaleFactor);

  t->requires(Task::NewDW, lb->pSizeLabel_preReloc,              Ghost::None);
  t->computes(lb->pScaleFactorLabel_preReloc );

  sched->addTask( t, patches, matls );
}

void TriangleTasks::computeTriangleScaleFactor(const ProcessorGroup*,
                                               const PatchSubset* patches,
                                               const MaterialSubset* ,
                                               DataWarehouse* old_dw,
                                               DataWarehouse* new_dw)
{
  // This task computes the particles initial physical size, to be used
  // in scaling particles for the deformed particle vis feature

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches,patch,cout_doing,"Doing MPM::computeTriangleScaleFactor");

    unsigned int numLSMatls=d_materialManager->getNumMatls( "Triangle" );
    for(unsigned int m = 0; m < numLSMatls; m++){
      TriangleMaterial* ls_matl = 
        (TriangleMaterial*) d_materialManager->getMaterial("Triangle", m);
      int dwi = ls_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      constParticleVariable<Matrix3> psize,pF;
      ParticleVariable<Matrix3> pScaleFactor;
      new_dw->get(psize,        lb->pSizeLabel_preReloc,                  pset);
      new_dw->allocateAndPut(pScaleFactor, lb->pScaleFactorLabel_preReloc,pset);

      if(m_output->isOutputTimeStep()){
        Vector dx = patch->dCell();
        for(ParticleSubset::iterator iter  = pset->begin();
                                     iter != pset->end(); iter++){
          particleIndex idx = *iter;
          pScaleFactor[idx] = ((Matrix3(dx[0],0,0,
                                        0,dx[1],0,
                                        0,0,dx[2])*psize[idx]));
        } // for particles
      } // isOutputTimestep
    } // loop over Triangle matls
  } // patches
}

void TriangleTasks::scheduleComputeNormalsTri(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSubset* mpm_matls,
                                           const MaterialSubset* triangle_matls,
                                           const MaterialSet* matls)
{
  if (!d_flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                             getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,"TriangleTasks::scheduleComputeNormalsTri");

  Task* t=scinew Task("TriangleTasks::computeNormalsTri",
                      this, &TriangleTasks::computeNormalsTri);

  Ghost::GhostType  gac = Ghost::AroundCells;

  t->requires(Task::OldDW, lb->pXLabel,              triangle_matls, gac, 2);
  t->requires(Task::OldDW, TriL->triNormalLabel,     triangle_matls, gac, 2);
  t->requires(Task::NewDW, lb->gMassLabel,           mpm_matls,      gac,NGN+3);

  t->computes(lb->gSurfNormLabel,                    mpm_matls);

  sched->addTask(t, patches, matls);
}

void TriangleTasks::computeNormalsTri(const ProcessorGroup *,
                                      const PatchSubset    * patches,
                                      const MaterialSubset * ,
                                            DataWarehouse  * old_dw,
                                            DataWarehouse  * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, "Doing computeNormalsTri");

    ParticleInterpolator* interpolator=scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    string interp_type = d_flags->d_interpolator_type;

    Ghost::GhostType gan   = Ghost::AroundNodes;

    unsigned int numMPMMatls = d_materialManager->getNumMatls( "MPM" );
    std::vector<NCVariable<Vector> >       gsurfnorm(numMPMMatls);

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl =
                     (MPMMaterial*) d_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();

      new_dw->allocateAndPut(gsurfnorm[m],    lb->gSurfNormLabel,    dwi,patch);
      gsurfnorm[m].initialize(Vector(0.0,0.0,0.0));
    }

    int numTriMatls=d_materialManager->getNumMatls("Triangle");
    Matrix3 size; size.Identity(); size*=0.5;

    for(int tmo = 0; tmo < numTriMatls; tmo++) {
      TriangleMaterial* t_matl = (TriangleMaterial *) 
                             d_materialManager->getMaterial("Triangle", tmo);
      int dwi_tri = t_matl->getDWIndex();
      int adv_matl = t_matl->getAssociatedMaterial();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi_tri, patch,
                                                        gan, 2, lb->pXLabel);
      constParticleVariable<Point>  tx;
      constParticleVariable<Vector> triNormal;
      old_dw->get(tx,         lb->pXLabel,            pset);
      old_dw->get(triNormal,  TriL->triNormalLabel,   pset);

      for(ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); iter++){
         particleIndex idx = *iter;
         int nn = interpolator->findCellAndWeights(tx[idx], ni, S, size);
         for (int k = 0; k < nn; k++) {
           IntVector node = ni[k];
           if(patch->containsNode(node)){
             gsurfnorm[adv_matl][node] += triNormal[idx]*S[k];
           }
         }
      } // triangles
    }   // triangle materials

    for(unsigned int m = 0; m < numMPMMatls; m++){
      for(NodeIterator iter =patch->getExtraNodeIterator();!iter.done();iter++){
        IntVector c = *iter;
        gsurfnorm[m][c] /= (gsurfnorm[m][c].length()+1.e-100);
      } // Nodes
    }   // MPM materials
  }     // patches
}

void TriangleTasks::scheduleRefineTriangles(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  if( !d_flags->doMPMOnLevel(getLevel(patches)->getIndex(), 
                             getLevel(patches)->getGrid()->numLevels())){
    return;
  }

  printSchedule( patches, cout_doing, "TriangleTasks::scheduleRefineTriangles");

  Task * t = scinew Task("TriangleTasks::refineTriangles", this,
                   &TriangleTasks::refineTriangles);

  Ghost::GhostType  gan   = Ghost::AroundNodes;

  t->requires(Task::OldDW, lb->pXLabel,                  gan, NGP);
  t->requires(Task::OldDW, lb->pSizeLabel,               gan, NGP);
  t->modifies(lb->pXLabel_preReloc);
  t->modifies(lb->pSizeLabel_preReloc);
  t->modifies(TriL->triangleIDLabel_preReloc);
  t->modifies(TriL->triMidToN0VectorLabel_preReloc);
  t->modifies(TriL->triMidToN1VectorLabel_preReloc);
  t->modifies(TriL->triMidToN2VectorLabel_preReloc);
  t->modifies(TriL->triUseInPenaltyLabel_preReloc);
  t->modifies(TriL->triAreaLabel_preReloc);
  t->modifies(TriL->triClayLabel_preReloc);
  t->modifies(TriL->triNormalLabel_preReloc);
  t->modifies(TriL->triAreaAtNodesLabel_preReloc);
  t->modifies(TriL->triMassDispLabel_preReloc);
  t->modifies(TriL->triNearbyMatsLabel_preReloc);
  t->modifies(TriL->triCementThicknessLabel_preReloc);

  sched->addTask(t, patches, matls);
}

void TriangleTasks::refineTriangles(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* ,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double dxMinL = dx.minComponent();
    double maxL2 = 4.*dxMinL*dxMinL; // max allowed tri edge length squared
    printTask(patches, patch,cout_doing, "Doing MPM::refineTriangles");
    unsigned int numTriMatls=d_materialManager->getNumMatls("Triangle");

    for(unsigned int m = 0; m < numTriMatls; m++){
      TriangleMaterial* tri_matl =
              (TriangleMaterial*) d_materialManager->getMaterial("Triangle", m);
      int dwi = tri_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      ParticleVariable<Point> px;
      ParticleVariable<Matrix3> pSize, tNearbyMats;
      ParticleVariable<long64> tids;
      ParticleVariable<IntVector> tUseInPenalty;
      ParticleVariable<Vector> tMTN0Vec,tMTN1Vec,tMTN2Vec,tAreaAtNodes,tNormal;
      ParticleVariable<double> tArea, tClay, tMassDisp, tCemThick;
      ParticleVariable<int> tRefine;

      new_dw->getModifiable(px,        lb->pXLabel_preReloc,              pset);
      new_dw->getModifiable(pSize,     lb->pSizeLabel_preReloc,           pset);
      new_dw->getModifiable(tNearbyMats, 
                                     TriL->triNearbyMatsLabel_preReloc,   pset);
      new_dw->getModifiable(tids,    TriL->triangleIDLabel_preReloc,      pset);
      new_dw->getModifiable(tUseInPenalty,
                                     TriL->triUseInPenaltyLabel_preReloc, pset);
      new_dw->getModifiable(tMTN0Vec,TriL->triMidToN0VectorLabel_preReloc,pset);
      new_dw->getModifiable(tMTN1Vec,TriL->triMidToN1VectorLabel_preReloc,pset);
      new_dw->getModifiable(tMTN2Vec,TriL->triMidToN2VectorLabel_preReloc,pset);
      new_dw->getModifiable(tNormal, TriL->triNormalLabel_preReloc,       pset);
      new_dw->getModifiable(tArea,   TriL->triAreaLabel_preReloc,         pset);
      new_dw->getModifiable(tAreaAtNodes, 
                                     TriL->triAreaAtNodesLabel_preReloc,  pset);
      new_dw->getModifiable(tClay,   TriL->triClayLabel_preReloc,         pset);
      new_dw->getModifiable(tMassDisp,
                                     TriL->triMassDispLabel_preReloc,     pset);
      new_dw->getModifiable(tCemThick,
                                   TriL->triCementThicknessLabel_preReloc,pset);

      new_dw->allocateTemporary(tRefine,       pset);

      unsigned int numRef=0;
      bool splitForAny=false;

      // Loop over triangles to see if any meet the refinement criteria
      const unsigned int origNTri = pset->addParticles(0);
      for( unsigned int pp=0; pp<origNTri; ++pp ){
        tRefine[pp]=0;
        Point A = px[pp]+tMTN0Vec[pp];      
        Point B = px[pp]+tMTN1Vec[pp];      
        Point C = px[pp]+tMTN2Vec[pp];      
        if((B-A).length2() > maxL2 ||
           (C-A).length2() > maxL2 ||
           (C-B).length2() > maxL2){
          tRefine[pp]=1;
          splitForAny=true;
          numRef++;
        }
      } // Loop over original triangles

      if(splitForAny){
        numRef*=3;  // Divide triangle into 4, reuse original + 3 new
        const unsigned int oldNumTri = pset->addParticles(numRef);

        ParticleVariable<Point> pxtmp;
        ParticleVariable<Matrix3> pSizetmp, tNearbyMatstmp;
        ParticleVariable<long64> tidstmp;
        ParticleVariable<IntVector> tUseInPenaltytmp;
        ParticleVariable<Vector> tMTN0Vectmp,tMTN1Vectmp,tMTN2Vectmp;
        ParticleVariable<Vector> tAreaAtNodestmp,tNormaltmp;
        ParticleVariable<double> tAreatmp, tClaytmp, tMassDisptmp, tCemThicktmp;
        ParticleVariable<int> tRefinetmp;

        new_dw->allocateTemporary(pxtmp,            pset);
        new_dw->allocateTemporary(pSizetmp,         pset);
        new_dw->allocateTemporary(tNearbyMatstmp,   pset);
        new_dw->allocateTemporary(tidstmp,          pset);
        new_dw->allocateTemporary(tUseInPenaltytmp, pset);
        new_dw->allocateTemporary(tMTN0Vectmp,      pset);
        new_dw->allocateTemporary(tMTN1Vectmp,      pset);
        new_dw->allocateTemporary(tMTN2Vectmp,      pset);
        new_dw->allocateTemporary(tAreaAtNodestmp,  pset);
        new_dw->allocateTemporary(tNormaltmp,       pset);
        new_dw->allocateTemporary(tAreatmp,         pset);
        new_dw->allocateTemporary(tClaytmp,         pset);
        new_dw->allocateTemporary(tMassDisptmp,     pset);
        new_dw->allocateTemporary(tCemThicktmp,     pset);

        // copy data from old variables
        for( unsigned int pp=0; pp<oldNumTri; ++pp ){
           pxtmp[pp]=px[pp];
           pSizetmp[pp]=pSize[pp];
           tNearbyMatstmp[pp]=tNearbyMats[pp];
           tidstmp[pp]=tids[pp];
           tUseInPenaltytmp[pp]=tUseInPenalty[pp];
           tMTN0Vectmp[pp]=tMTN0Vec[pp];
           tMTN1Vectmp[pp]=tMTN1Vec[pp];
           tMTN2Vectmp[pp]=tMTN2Vec[pp];
           tAreaAtNodestmp[pp]=tAreaAtNodes[pp];
           tNormaltmp[pp]=tNormal[pp];
           tAreatmp[pp]=tArea[pp];
           tClaytmp[pp]=tClay[pp];
           tMassDisptmp[pp]=tMassDisp[pp];
           tCemThicktmp[pp]=tCemThick[pp];
        }
        int numRefTri=0;
          double oneThird = (1./3.);
          for( unsigned int idx=0; idx<oldNumTri; ++idx ){
            if(tRefine[idx]==1){
              Point A = px[idx]+tMTN0Vec[idx];      
              Point B = px[idx]+tMTN1Vec[idx];      
              Point C = px[idx]+tMTN2Vec[idx];      

              Point AB = 0.5*(A+B);
              Point AC = 0.5*(A+C);
              Point BC = 0.5*(B+C);

              // Fix area of original triangle
              tAreatmp[idx]=0.5*(Cross(BC-AB,AC-AB).length());
              tMTN0Vectmp[idx] = AB - pxtmp[idx];
              tMTN1Vectmp[idx] = AC - pxtmp[idx];
              tMTN2Vectmp[idx] = BC - pxtmp[idx];
              // Need to fix tAreaAtNodes for original triangle
              // tAreaAtNodestmp[idx]=???;
  
              for(int i = 0;i<3;i++){
                int new_index = oldNumTri + 3*numRefTri+i;
                IntVector tUIP(0,0,0);
                Vector tAAN(0.,0.,0.);
                if(i==0){
                  pxtmp[new_index]=oneThird*(A+AB+AC);
                  tMTN0Vectmp[new_index] = A  - pxtmp[new_index];
                  tMTN1Vectmp[new_index] = AB - pxtmp[new_index];
                  tMTN2Vectmp[new_index] = AC - pxtmp[new_index];
                  tAreatmp[new_index]=(0.5*(Cross(AB-A,AC-A)).length());
                  tUIP = IntVector(tUseInPenalty[idx].x(),0,0);
                  tAAN = Vector(tAreaAtNodes[idx].x(),0.,0.);
                } else if(i==1){
                  pxtmp[new_index]=oneThird*(B+AB+BC);
                  tMTN0Vectmp[new_index] = B  - pxtmp[new_index];
                  tMTN1Vectmp[new_index] = BC - pxtmp[new_index];
                  tMTN2Vectmp[new_index] = AB - pxtmp[new_index];
                  tAreatmp[new_index]=(0.5*(Cross(AB-B,BC-B)).length());
                  tUIP = IntVector(tUseInPenalty[idx].y(),0,0);
                  tAAN = Vector(tAreaAtNodes[idx].y(),0.,0.);
                } else if(i==2){
                  pxtmp[new_index]=oneThird*(C+AC+BC);
                  tMTN0Vectmp[new_index] = C  - pxtmp[new_index];
                  tMTN1Vectmp[new_index] = AC - pxtmp[new_index];
                  tMTN2Vectmp[new_index] = BC - pxtmp[new_index];
                  tAreatmp[new_index]=(0.5*(Cross(AC-C,BC-C)).length());
                  tUIP = IntVector(tUseInPenalty[idx].z(),0,0);
                  tAAN = Vector(tAreaAtNodes[idx].z(),0.,0.);
                }
                tAreaAtNodestmp[new_index]  = tAAN; // Fix these two lines
                tAreaAtNodestmp[idx]        = Vector(0.,0.,0.);
                tUseInPenaltytmp[new_index] = tUIP;
                tUseInPenaltytmp[idx]       = IntVector(1,1,1);
                pSizetmp[new_index]         = pSize[idx];  // not used?
                tNearbyMatstmp[new_index]   = tNearbyMats[idx];
                tNormaltmp[new_index]       = tNormal[idx];
                tClaytmp[new_index]         = tClay[idx];
                tMassDisptmp[new_index]     = tMassDisp[idx];
                tCemThicktmp[new_index]     = tCemThick[idx];
                tidstmp[new_index]          = -tids[idx]*pow(10,i);
              }
              numRefTri++;
            } // This triangle needs to be refined
          } // Loop over old particles

        // Put temporary data in DW
        new_dw->put(pxtmp,        lb->pXLabel_preReloc,                  true);
        new_dw->put(pSizetmp,     lb->pSizeLabel_preReloc,               true);
        new_dw->put(tNearbyMatstmp, 
                                  TriL->triNearbyMatsLabel_preReloc,     true);
        new_dw->put(tidstmp,      TriL->triangleIDLabel_preReloc,        true);
        new_dw->put(tUseInPenaltytmp,
                                  TriL->triUseInPenaltyLabel_preReloc,   true);
        new_dw->put(tMTN0Vectmp,  TriL->triMidToN0VectorLabel_preReloc,  true);
        new_dw->put(tMTN1Vectmp,  TriL->triMidToN1VectorLabel_preReloc,  true);
        new_dw->put(tMTN2Vectmp,  TriL->triMidToN2VectorLabel_preReloc,  true);
        new_dw->put(tNormaltmp,   TriL->triNormalLabel_preReloc,         true);
        new_dw->put(tAreatmp,     TriL->triAreaLabel_preReloc,           true);
        new_dw->put(tAreaAtNodestmp, 
                                  TriL->triAreaAtNodesLabel_preReloc,    true);
        new_dw->put(tClaytmp,     TriL->triClayLabel_preReloc,           true);
        new_dw->put(tMassDisptmp, TriL->triMassDispLabel_preReloc,       true);
        new_dw->put(tCemThicktmp, TriL->triCementThicknessLabel_preReloc,true);
      } // Some triangle(s) needs to be refined
    } // Loop over matls
  } // Loop over patches
}

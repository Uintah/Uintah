/*
 * DiscreteInterface.cc
 *
 *  Created on: Feb 18, 2017
 *      Author: jbhooper
 *
 *
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

#include <CCA/Components/MPM/Diffusion/DiffusionInterfaces/DiscreteInterface.h>

using namespace Uintah;

DiscreteSDInterface::DiscreteSDInterface(
                                         ProblemSpecP     & probSpec  ,
                                         SimulationStateP & simState  ,
                                         MPMFlags         * mFlags    ,
                                         MPMLabel         * mpmLabel
                                        )
                                        : SDInterfaceModel(probSpec, simState,
                                                           mFlags, mpmLabel)

{


  gSurfaceNormalDiffusion = VarLabel::create("g.surfnormdiffusion", NCVariable<Vector>::getTypeDescription() );
  gPositionDiffusion      = VarLabel::create("g.positiondiffusion", NCVariable<Vector>::getTypeDescription() );
}

DiscreteSDInterface::~DiscreteSDInterface()
{
  VarLabel::destroy(gSurfaceNormalDiffusion);
  VarLabel::destroy(gPositionDiffusion);
}

void DiscreteSDInterface::addComputesAndRequiresInterpolated(
                                                                   SchedulerP   & sched   ,
                                                             const PatchSet     * patches ,
                                                             const MaterialSet  * matls
                                                            )
{
  // Shouldn't need to directly modify the concentration.
}

void DiscreteSDInterface::sdInterfaceInterpolated(
                                                  const ProcessorGroup  *         ,
                                                  const PatchSubset     * patches ,
                                                  const MaterialSubset  * matls   ,
                                                        DataWarehouse   * old_dw  ,
                                                        DataWarehouse   * new_dw
                                                 )
{

}

void DiscreteSDInterface::addComputesAndRequiresDivergence(
                                                                 SchedulerP   & sched,
                                                           const PatchSet     * patches,
                                                           const MaterialSet  * matls
                                                          )
{
  Ghost::GhostType  gan   = Ghost::AroundNodes;
  Ghost::GhostType  gnone = Ghost::None;

  Task* task  = scinew Task("DiscreteSDInterface::sdInterfaceDivergence", this, &DiscreteSDInterface::sdInterfaceDivergence);


  Ghost::GhostType gp;
  int              numGhost;
  d_shared_state->getParticleGhostLayer(gp, numGhost);

  const MaterialSubset* mss = matls->getUnion();

  task->requires(Task::OldDW, d_shared_state->get_delt_label());

  MaterialSubset* globalMat = d_shared_state->getAllInOneMatl();
  // Require total nodal quantities
  task->requires(Task::NewDW, d_mpm_lb->gMassLabel,          globalMat, Task::OutOfDomain, gnone);
  task->requires(Task::NewDW, d_mpm_lb->gVolumeLabel,        globalMat, Task::OutOfDomain, gnone);
  task->requires(Task::NewDW, d_mpm_lb->gTemperatureLabel,   globalMat, Task::OutOfDomain, gnone);
  task->requires(Task::NewDW, d_mpm_lb->gConcentrationLabel, globalMat, Task::OutOfDomain, gnone);

  task->requires(Task::OldDW, d_mpm_lb->pXLabel,                  gp, numGhost);
  task->requires(Task::OldDW, d_mpm_lb->pVolumeLabel,             gp, numGhost);
  task->requires(Task::OldDW, d_mpm_lb->pSizeLabel,               gp, numGhost);
  task->requires(Task::OldDW, d_mpm_lb->pDeformationMeasureLabel, gp, numGhost);
  task->requires(Task::OldDW, d_mpm_lb->pMassLabel,               gp, numGhost);

  task->requires(Task::NewDW, d_mpm_lb->gMassLabel,           gan, 1);
  task->requires(Task::NewDW, d_mpm_lb->gVolumeLabel,         gan, 1);
  task->requires(Task::NewDW, d_mpm_lb->gTemperatureLabel,    gan, 1);
  task->requires(Task::NewDW, d_mpm_lb->gConcentrationLabel,  gan, 1);

  task->modifies(d_mpm_lb->gConcentrationRateLabel,           mss);

  task->computes(gSurfaceNormalDiffusion, globalMat, Task::OutOfDomain);
  task->computes(gPositionDiffusion,      globalMat, Task::OutOfDomain);

  task->computes(gSurfaceNormalDiffusion);
  task->computes(gPositionDiffusion);

  sched->addTask(task, patches, matls);
}

void DiscreteSDInterface::sdInterfaceDivergence(
                                                const ProcessorGroup  *         ,
                                                const PatchSubset     * patches ,
                                                const MaterialSubset  * matls   ,
                                                      DataWarehouse   * oldDW   ,
                                                      DataWarehouse   * newDW
                                               )
{
  int numMatls = d_shared_state->getNumMPMMatls();

  Ghost::GhostType  typeGhost;
  int               numGhost;
  d_shared_state->getParticleGhostLayer(typeGhost, numGhost);

  Ghost::GhostType gan    = Ghost::AroundNodes;
  Ghost::GhostType gnone  = Ghost::None;

  StaticArray<constNCVariable<double> > gMass(numMatls);  // m_i,j = gMass[j][i]
  StaticArray<constNCVariable<double> > gVol(numMatls);   // V_i,j = gVol[j][i]
  StaticArray<constNCVariable<double> > gConc(numMatls);  // C_i,j = gConc[j][i]
  StaticArray<constNCVariable<double> > gTemp(numMatls);  // T_i,j = gTemp[j][i]

  StaticArray<NCVariable<Vector> >      gDomGrad(numMatls);  // -grad(Omega(i,j)) = gDomGrad[j][i]
  StaticArray<NCVariable<Vector> >      gPosition(numMatls);
  StaticArray<NCVariable<double> >      gConcRate(numMatls);

  constNCVariable<double> gTotalMass, gTotalVol, gTotalTemp, gTotalConc;

  NCVariable<Vector> gTotalDomain, gTotalPosition;

  delt_vartype delT;
  oldDW->get(delT, d_shared_state->get_delt_label(), getLevel(patches) );

  for (int patchIdx = 0; patchIdx < patches->size(); ++patchIdx)
  {
    const Patch* patch = patches->get(patchIdx);
    Vector dx = patch->dCell();
    Vector oodx = Vector(1.0)/dx;
    double cellVol = dx.x()*dx.y()*dx.z(); // dXdYdZ
    Vector dx2 = dx*dx;

    ParticleInterpolator*   pInterp = d_mpm_flags->d_interpolator->clone(patch);
    int                     nodesPerPoint = pInterp->size();
    std::vector<IntVector>  nIdxList(nodesPerPoint);
    std::vector<double>            S(nodesPerPoint);
    std::vector<Vector>          d_S(nodesPerPoint);

    // Grab aggregate nodal values
    newDW->get(gTotalMass,  d_mpm_lb->gMassLabel,
               d_shared_state->getAllInOneMatl()->get(0),        patch, gan, 1);
    newDW->get(gTotalVol,   d_mpm_lb->gVolumeLabel,
               d_shared_state->getAllInOneMatl()->get(0),        patch, gan, 1);
    newDW->get(gTotalTemp,  d_mpm_lb->gTemperatureLabel,
               d_shared_state->getAllInOneMatl()->get(0),        patch, gan, 1);
    newDW->get(gTotalConc,  d_mpm_lb->gConcentrationLabel,
               d_shared_state->getAllInOneMatl()->get(0),        patch, gan, 1);

    newDW->allocateAndPut(gTotalDomain, gSurfaceNormalDiffusion,
                          d_shared_state->getAllInOneMatl()->get(0), patch);
    newDW->allocateAndPut(gTotalPosition, gPositionDiffusion,
                          d_shared_state->getAllInOneMatl()->get(0), patch);

    gTotalDomain.initialize(Vector(0.0));
    gTotalPosition.initialize(Vector(0.0));

    // Loop over materials and preload our arrays of material based nodal data
    for (int mIdx = 0; mIdx < numMatls; ++mIdx)
    {
      int dwi = matls->get(mIdx);
      newDW->get(gMass[mIdx], d_mpm_lb->gMassLabel,           dwi, patch, gan, 1);
      newDW->get(gVol[mIdx],  d_mpm_lb->gVolumeLabel,         dwi, patch, gan, 1);
      newDW->get(gTemp[mIdx], d_mpm_lb->gTemperatureLabel,    dwi, patch, gan, 1);
      newDW->get(gConc[mIdx], d_mpm_lb->gConcentrationLabel,  dwi, patch, gan, 1);

      newDW->getModifiable(gConcRate[mIdx],
                           d_mpm_lb->gConcentrationRateLabel, dwi, patch);
      newDW->allocateAndPut(gDomGrad[mIdx],
                            gSurfaceNormalDiffusion,          dwi, patch);
      newDW->allocateAndPut(gPosition[mIdx],
                            gPositionDiffusion,               dwi, patch);


      gDomGrad[mIdx].initialize(Vector(0.0));
      gPosition[mIdx].initialize(Vector(0.0));

      // Break out if we're not working with this material contact
      if (!d_materials_list.requested(mIdx)) continue;

      ParticleSubset* pset = oldDW->getParticleSubset(dwi, patch,
                                                      typeGhost, numGhost,
                                                      d_mpm_lb->pXLabel);
      constParticleVariable<Point>    pX;
      constParticleVariable<double>   pVol, pMass;
      constParticleVariable<Matrix3>  pSize, pDeform;

      oldDW->get(pX,          d_mpm_lb->pXLabel,                    pset);
      oldDW->get(pVol,        d_mpm_lb->pVolumeLabel,               pset);
      oldDW->get(pSize,       d_mpm_lb->pSizeLabel,                 pset);
      oldDW->get(pDeform,     d_mpm_lb->pDeformationMeasureLabel,   pset);
      oldDW->get(pMass,       d_mpm_lb->pMassLabel,                 pset);

      // From Nairn, CMES, vol. 92, no. 3, p 271-299, 2013
      //   Specifically, page 282, Eqn. 33:
      // Calculate the individual material domain gradients.
      for (int pIdx = 0; pIdx < pset->numParticles(); ++pIdx)
      {
        int numNodes =
            pInterp->findCellAndWeightsAndShapeDerivatives(pX[pIdx], nIdxList,
                                                           S, d_S, pSize[pIdx],
                                                           pDeform[pIdx]);

        double rho = pMass[pIdx]/pVol[pIdx];
        for (int nIdx = 0; nIdx < numNodes; ++nIdx)
        {
          IntVector node = nIdxList[nIdx];
          if (patch->containsNode(node))
          {
            gDomGrad[mIdx][node] += d_S[nIdx] * pVol[pIdx]; // Omega_i,j
            gPosition[mIdx][node] += pX[pIdx].asVector() * pMass[pIdx] * S[nIdx];
          }
        } // Loop over nodes related to particle
      } // Loop over particles
    } // Loop over materials

    for(int mIdx = 0; mIdx < numMatls; ++mIdx)
    {
      // Apply appropriate boundary conditions to the position and surface
      // normals.
      int dwi = matls->get(mIdx);
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch, dwi, "Symmetric", gDomGrad[mIdx],
                              d_mpm_flags->d_interpolator_type);
      bc.setBoundaryCondition(patch, dwi, "Symmetric", gPosition[mIdx],
                              d_mpm_flags->d_interpolator_type);
    }

    // Calculate the total domain gradient.
    for (NodeIterator nIt = patch->getExtraNodeIterator(); !nIt.done(); ++nIt)
    {
      IntVector node = *nIt;
      double inverseTotalNodalMass = 1.0/gTotalMass[node];
      for (int mIdx = 0; mIdx < numMatls; ++mIdx)
      {
        if (d_materials_list.requested(mIdx) )
        {
          gTotalPosition[node]  += gPosition[mIdx][node];
          gPosition[mIdx][node] *= inverseTotalNodalMass;
          gTotalDomain[node]    += gDomGrad[mIdx][node]; // Omega_i
        }

      }
      gTotalPosition[node] *= inverseTotalNodalMass;
    }


    for (int mIdx = 0; mIdx < numMatls; ++mIdx)
    {
      int dwi = matls->get(mIdx);
      Vector matGrad;
      Vector complGrad;
      for (NodeIterator nIt = patch->getExtraNodeIterator(); !nIt.done(); ++nIt)
      {
        IntVector node = *nIt;
//        gradient = Vector(0.0);
//        double gradMag  = 0.0;
//        double matVolFraction = gVol[mIdx][node]/gTotalVol[node];
        // if matVolFraction = 0 or 1, node has no interface.
//        if ( !(compare(matVolFraction, 0.0) || compare(matVolFraction, 1.0)) )
//        {
        matGrad  = gDomGrad[mIdx][node];
        double matMag   = matGrad.length();

        Vector complGrad  = gTotalDomain[node] - gDomGrad[mIdx][node];
        double complMag   = complGrad.length();
        if (matMag < complMag)
        {
          gDomGrad[mIdx][node]  = -complGrad;
          matMag = complMag;
        }

        if (matMag > 1e-15)
        {
          gDomGrad[mIdx][node] /= matMag;
        }
        else
        {
            // If the gradient is too small to normalize by, there's probably
            // not actually a gradient.
          gDomGrad[mIdx][node] = Vector(0.0);
        }
      } // Loop over nodes
    } // Loop over materials

//    // Calculate the normals for all materials.
//    for (NodeIterator nIt = patch->getExtraNodeIterator(); !nIt.done(); ++nIt)
//    {
//      IntVector node = *nIt;
//      for (int mIdx = 0; mIdx < numMatls; ++mIdx)
//      {
//        Vector gradient = Vector(0.0);
//        double gradMag = 0.0;
//        bool isSame = compare(gVol[mIdx][node],gTotalVol[node]);
//        if (!isSame) // Same volume == no interface
//        {
//          Vector matGrad = gDomGrad[mIdx][node];
//          gradMag = matGrad.length();
//          Vector complGrad = gTotalDomain[node] - gDomGrad[mIdx][node];
//          double complMag = complGrad.length();
//          if (gradMag < complMag)
//          {
//            gradient = complGrad;
//            gradMag = complMag;
//          }
//        }
//        if (gradMag > 1e-15)
//        {
//          gDomGrad[mIdx][node] = gradient / gradMag;
//        }
//        else
//        {
//          // There seems to be a gradient so small it's not safe to normalize by
//          gDomGrad[mIdx][node] = Vector(0.0);
//        }
//
////        // Average Volume Gradient approach
////        Vector gradient = (1.0/2.0*gTotalVol[node])*
////                          ( (gVol[mIdx][node]*matGrad) -
////                            (gTotalVol[node]-gVol[mIdx][node])*(complGrad)
////                          );
//
//      }
//    }


    // Calculate the contact area for all nodes/materials.
    for (NodeIterator nIt = patch->getExtraNodeIterator(); !nIt.done(); ++nIt)
    {
      IntVector node            = *nIt;
      double Vt                 = gTotalVol[node];
      Vector NodalX             = gTotalPosition[node];
      double NodalMass          = gTotalMass[node];
      double invTotalFluxMass   = 1.0/(gTotalConc[node]*NodalMass);

      for (int mIdx = 0; mIdx < numMatls; ++mIdx)
      {
        double gradMag = gDomGrad[mIdx][node].length();
        if (compare(gradMag,0.0)) continue;  // Gradient is zero, node.
        Vector n = gDomGrad[mIdx][node]; // n^   Unit normal to surface separation direction
        Vector delta = (NodalX - gPosition[mIdx][node])*NodalMass
                         / (gTotalMass[node]-gMass[mIdx][node]);

        double perpSeparation = (Dot(n,delta));

        Vector t = delta - n*perpSeparation; // t^  Unit tangent to surface separation direction
        double tMag = t.length();
        if (compare(tMag,0.0)) // surface normal is completely perpendicular to surface
        {
          // Pick an arbitrary vector.
          Vector corner(1.0/sqrt(3.0));  // Pick an arbitrary 1,1,1 vector
          double ndotcorner = Dot(n,corner);
          if (compare(abs(ndotcorner),1.0)) // Check to make sure that's not n.
          {
            // if it is, then this one has to be okay!
            corner=Vector(1.0/sqrt(2.0),1.0/sqrt(2.0),0.0);
          }
          t = corner - Dot(n,corner)*n;
        }
        tMag = t.length();
        t /= tMag;

        Vector tcrossn = Cross(t,n); // n^ X t^
        double tnMag = tcrossn.length();
        tcrossn /= tnMag;

        Vector a = t*t/dx2;
        Vector b = tcrossn*tcrossn/dx2;
        double hPerp = cellVol*sqrt(Dot(a,a)*Dot(b,b));

        double Vm = gVol[mIdx][node];
        double Vother = Vt-Vm;
        double area = sqrt(2.0*Vt*std::min(Vm,Vother))/hPerp;
//        double phi = gConc[mIdx][node]*gMass[mIdx][node]*invTotalFluxMass;
        double D = AlNi::Diffusivity(gTemp[mIdx][node]);
        double deltaC = D*area*(gTotalConc[node]-gConc[mIdx][node]); ///(1.0-phi);

        gConcRate[mIdx][node] += deltaC/delT;

      }
    }
  }
}

//void DiscreteSDInterface::sdInterfaceDivergenceOld(
//                                                const ProcessorGroup  *         ,
//                                                const PatchSubset     * patches ,
//                                                const MaterialSubset  * matls   ,
//                                                      DataWarehouse   * old_dw  ,
//                                                      DataWarehouse   * new_dw
//                                               )
//{
//  // Find grid-based gradient between the materials
//  int num_matls = d_shared_state->getNumMPMMatls();
//
//  Ghost::GhostType ghostRef;
//  int              numGhost;
//  d_shared_state->getParticleGhostLayer(ghostRef, numGhost);
//
//  Ghost::GhostType  gan   = Ghost::AroundNodes;
//  Ghost::GhostType  gnone = Ghost::None;
//
//  StaticArray<constNCVariable<double> > gmass(num_matls);
//  StaticArray<constNCVariable<double> > gconcentration(num_matls);
//  StaticArray<constNCVariable<double> > gtemperature(num_matls);
//
//  StaticArray<NCVariable<double> >      gConcRate(num_matls);
//  StaticArray<NCVariable<Vector> >      gsurfnorm(num_matls);
//  StaticArray<NCVariable<Vector> >      gposition(num_matls);
//
//  for (int patchIdx = 0; patchIdx < patches->size(); ++patchIdx)
//  {
//    delt_vartype delT;
//    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );
//
//
//    const Patch* patch = patches->get(patchIdx);
//    Vector oodx = Vector(1.0)/patch->dCell();
//
//    ParticleInterpolator*   pInterp = d_mpm_flags->d_interpolator->clone(patch);
//    std::vector<IntVector>   ni(pInterp->size());
//    std::vector<double>       S(pInterp->size());
//    std::vector<Vector>     d_S(pInterp->size());
//
//    for (int mIdx = 0; mIdx < num_matls; ++mIdx)
//    {
//      int dwi = matls->get(mIdx);
//
//      new_dw->get(gmass[mIdx],          d_mpm_lb->gMassLabel,           dwi,
//                  patch, gan, 1);
//      new_dw->get(gconcentration[mIdx], d_mpm_lb->gConcentrationLabel,  dwi,
//                  patch, gan, 1);
//      new_dw->get(gtemperature[mIdx],   d_mpm_lb->gTemperatureLabel,    dwi,
//                  patch, gan, 1);
//
//      new_dw->getModifiable(gConcRate[mIdx],
//                            d_mpm_lb->gConcentrationRateLabel,  dwi,    patch);
//      new_dw->allocateAndPut(gsurfnorm[mIdx],
//                             gSurfaceNormalDiffusion,           dwi,    patch);
//
//      gsurfnorm[mIdx].initialize(Vector(0.0));
//
//      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
//                                                       ghostRef,
//                                                       numGhost,
//                                                       d_mpm_lb->pXLabel);
//      constParticleVariable<Point>    pX;
//      constParticleVariable<double>   pMass;
//      constParticleVariable<double>   pVolume;
//      constParticleVariable<Matrix3>  pSize;
//      constParticleVariable<Matrix3>  deformationGradient;
//
//      old_dw->get(pX, d_mpm_lb->pXLabel,      pset);
//      old_dw->get(pSize, d_mpm_lb->pXLabel, pset);
//      old_dw->get(deformationGradient, d_mpm_lb->pDeformationMeasureLabel, pset);
//      old_dw->get(pVolume, d_mpm_lb->pVolumeLabel, pset);
//      old_dw->get(pMass, d_mpm_lb->pMassLabel, pset);
//
//      if (!d_materials_list.requested(mIdx)) continue;
//
//      // Compute the normals for all the interior nodes.
//      for(ParticleSubset::iterator pIt = pset->begin(); pIt != pset->end(); ++pIt)
//      {
//        int pIdx = *pIt;
//        int numNodes = pInterp->findCellAndWeightsAndShapeDerivatives(pX[pIdx],
//                                                                      ni, S,
//                                                                      d_S,
//                                                                      pSize[pIdx],
//                                                                      deformationGradient[pIdx]);
//        double rho = pMass[pIdx]/pVolume[pIdx];
//        for (int nIdx = 0; nIdx < numNodes; ++nIdx)
//        {
//          if (patch->containsNode(ni[nIdx]))
//          {
//            Vector grad=(d_S[nIdx]*oodx);
//            gsurfnorm[mIdx][ni[nIdx]] += pMass[pIdx] * grad;
//            gposition[mIdx][ni[nIdx]] += pX[pIdx].asVector()*pMass[pIdx]*S[nIdx];
//          }
//        } // Loop over nodes
//      } // Loop over particles
//
//    } // Loop over materials
//
//    for (NodeIterator nIt = patch->getExtraNodeIterator(); !nIt.done(); ++nIt)
//    {
//      IntVector nIdx = *nIt;
//      double maxMagnitude = gsurfnorm[0][nIdx].length();
//      int maxMaterial = 0;
//      for (int mIdx = 0; mIdx < num_matls; ++mIdx)
//      {
//        double magnitude = gsurfnorm[mIdx][nIdx].length();
//        if (magnitude > maxMagnitude)
//        {
//          maxMagnitude = magnitude;
//          maxMaterial = mIdx;
//        }
//      } // Find max magnitude
//
//      // This inverts the sense of all but the principle material so that the normals oppose each other.
//      for (int mIdx = 0; mIdx < num_matls; ++mIdx)
//      {
//        gposition[mIdx][nIdx] /= gmass[mIdx][nIdx];
//        if (mIdx != maxMaterial)
//        {
//          gsurfnorm[mIdx][nIdx] = -gsurfnorm[mIdx][nIdx];
//        }
//      } // Loop over materials
//    } // Loop over nodes
//
//    for (int mIdx = 0; mIdx < num_matls; ++mIdx)
//    {
//      int dwi = matls->get(mIdx);
//      MPMBoundCond bc;
//      bc.setBoundaryCondition(patch, dwi, "Symmetric", gsurfnorm[mIdx],
//                              d_mpm_flags->d_interpolator_type);
//
//      for (NodeIterator nIt = patch->getExtraNodeIterator(); !nIt.done(); ++nIt)
//      {
//        IntVector nIdx = *nIt;
//        double length = gsurfnorm[mIdx][nIdx].length();
//        if (length > 1e-15) gsurfnorm[mIdx][nIdx] /= length;
//      }
//    } // Loop over matls
//
//    // Now we have the direction of the inter-surface gradient.  The change
//    // in concentration due to divergence of the flux:
//    //   dC/dt = D * dot(grad, grad(c))
//    //   grad(c) across the interface is in the direction of the interface
//    //   normal, with a magnitude equal to c_m1 - c_m2 where m1, m2 represent
//    //   the materials.  Therefore, for inter-material diffusion at interfaces:
//    //    dC/dt = D*dot(grad, n*(c_m1 - c_m2))
//
//    for (NodeIterator nIt = patch->getExtraNodeIterator(); !nIt.done(); ++nIt)
//    {
//      IntVector nIdx = *nIt;
//      for (int m1_Idx = 0; m1_Idx < num_matls; ++m1_Idx)
//      {
//        if (d_materials_list.requested(m1_Idx) )
//        {
//          double d1 = calculateDiffusivity(gtemperature[m1_Idx][nIdx]);
//          double c1 = gconcentration[m1_Idx][nIdx];
//          for (int m2_Idx = 0; m2_Idx < num_matls; ++m2_Idx)
//          {
//            if (d_materials_list.requested(m2_Idx) )
//            {
//              double d2 = calculateDiffusivity(gtemperature[m2_Idx][nIdx]);
//              double c2 = gconcentration[m2_Idx][nIdx];
//              double D = std::min(d1,d2);
//              double deltaC = c1 - c2;
//              double dCdt = Dot()
//            }
//          }
//
//        }
//      }
//    }
//    for (int mIdx = 0; mIdx < num_matls; ++mIdx)
//    {
//      int dwi = matls->get(mIdx);
//      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, ghostRef,
//                                                       numGhost,
//                                                       d_mpm_lb->pXLabel);
//      constParticleVariable<Point>    pX;
//      constParticleVariable<Matrix3>  pSize;
//      constParticleVariable<Matrix3>  pStress, deformationGradient;
//    }
//
//    }
//
//
//
//  }


void DiscreteSDInterface::outputProblemSpec(
                                            ProblemSpecP  & ps
                                           )
{
  ProblemSpecP sdim_ps = ps;
  sdim_ps = ps->appendChild("diffusion_interface");
  sdim_ps->appendElement("type","discrete");
  d_materials_list.outputProblemSpec(sdim_ps);
}

double DiscreteSDInterface::calculateDiffusivity(double Temp)
{
  double R = 8.3144598; // Gas constant in J/(mol*K)
  double Rinv = 1.0/R;
  double D = 0.0;
  if (Temp < 724)
  {
    double E_activation = 92586; // J/mol
    D = 2.08e-7 * exp(-(E_activation*Rinv/Temp)); // Diffusivity m^2/s
  }
  else if (Temp > 860)
  {
    double E_activation = 98646; // J/mol
    D = 1.81e-6 * exp(-(E_activation*Rinv/Temp)); // Diffusivity m^2/s
  }
  else
  {
    D = ((-1.9915e-18 * Temp) + 4.4889e-15)*Temp -2.1627e-12;
  }
  return D;
}

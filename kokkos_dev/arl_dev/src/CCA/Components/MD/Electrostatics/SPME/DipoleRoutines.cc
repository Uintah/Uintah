/*
 *
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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
 *
 * ----------------------------------------------------------
 * SPME_newChargeDipole.cc
 *
 *  Created on: May 16, 2014
 *      Author: jbhooper
 */

//#include <CCA/Ports/Scheduler.h>

//#include <Core/Grid/Patch.h>
//#include <Core/Parallel/Parallel.h>
//#include <Core/Thread/Thread.h>
//#include <Core/Grid/Variables/ParticleVariable.h>
//#include <Core/Grid/Variables/CCVariable.h>
//#include <Core/Grid/Variables/SoleVariable.h>
//#include <Core/Grid/Variables/ParticleSubset.h>
//#include <Core/Grid/Variables/VarTypes.h>
//#include <Core/Grid/Box.h>
//#include <Core/Grid/DbgOutput.h>
//#include <Core/Geometry/IntVector.h>
//#include <Core/Geometry/Point.h>
//#include <Core/Math/MiscMath.h>
//#include <Core/Util/DebugStream.h>
//
//#include <iostream>
//#include <iomanip>
//#include <cstring>
//#include <cmath>
//
//#include <sci_values.h>
//#include <sci_defs/fftw_defs.h>
//
//#include <CCA/Components/MD/MDSystem.h>
//#include <CCA/Components/MD/MDLabel.h>
//#include <CCA/Components/MD/SimpleGrid.h>
#include <Core/Thread/Thread.h>


#include <CCA/Components/MD/Electrostatics/SPME/SPME.h>
#include <CCA/Components/MD/CoordinateSystems/CoordinateSystem.h>


//#ifdef DEBUG
//#include <Core/Util/FancyAssert.h>
//#endif

//.......1.........2.........3.........4.........5.........6.........7.........8
using namespace Uintah;

// Called from SPME::calculate directly
void SPME::generateChargeMapDipole(const ProcessorGroup*    pg,
                                   const PatchSubset*       patches,
                                   const MaterialSubset*    materials,
                                         DataWarehouse*     oldDW,
                                         DataWarehouse*     newDW,
                                   const MDLabel*           label,
                                         CoordinateSystem*  coordSys)
{
  size_t numPatches     =   patches->size();
  size_t numAtomTypes   =   materials->size();

  Uintah::Matrix3 inverseUnitCell = coordSys->getInverseCell();

  // Step through all the patches on this thread
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* patch = patches->get(patchIndex);
    // Extract SPMEPatch which maps to our current patch
    SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;

    // Step through all the materials in this patch
    for (size_t typeIndex = 0; typeIndex < numAtomTypes; ++typeIndex) {
      int atomType = materials->get(typeIndex);

      ParticleSubset* atomSubset;
      atomSubset = oldDW->getParticleSubset(atomType, patch);

      constParticleVariable<Point> atomPositions;
      constParticleVariable<long64> atomIDs;

      oldDW->get(atomPositions, label->global->pX, atomSubset);
      oldDW->get(atomIDs, label->global->pID, atomSubset);

      int numAtoms = atomSubset->numParticles();
      // Verify we have enough memory to hold the charge map for the
      // current atom type
      currentSPMEPatch->verifyChargeMapAllocation(numAtoms, atomType);

      // Pull the location for the SPMEPatch's copy of the charge map
      // for this material type
      std::vector<SPMEMapPoint>* gridMap;
      gridMap = currentSPMEPatch->getChargeMap(atomType);

      // begin loop to generate the charge map
      ParticleSubset::iterator atom;
      for (atom = atomSubset->begin(); atom != atomSubset->end(); ++atom) {
        particleIndex   atomIndex       =  *atom;
        particleId      ID              =   atomIDs[atomIndex];
        Point           position        =   atomPositions[atomIndex];

        SCIRun::Vector  kReal           =   d_kLimits.asVector();

        SCIRun::Vector  atomGridCoordinates;
        coordSys->toReduced(position.asVector(),atomGridCoordinates);
        atomGridCoordinates *= kReal;

        SCIRun::IntVector atomGridOffset =
                              IntVector(atomGridCoordinates.asPoint());

        // Set map point to associate with the current particle
        SPMEMapPoint*   currMapPoint    = &((*gridMap)[atomIndex]);
        currMapPoint->setParticleID(ID);
        currMapPoint->setGridOffset(atomGridOffset);


        SCIRun::Vector splineValues = atomGridOffset.asVector()
                                        - atomGridCoordinates;

        size_t support = d_interpolatingSpline.getSupport();
//                SCIRun::IntVector SupportVector(support, support, support);
        std::vector<SCIRun::Vector> baseLevel(support);
        std::vector<SCIRun::Vector> firstDerivative(support);
        std::vector<SCIRun::Vector> secondDerivative(support);

        // All simple grids are already created with:
        //   extents = SupportVector
        //   offset  = SciRun::IntVector (0,0,0)
        //   We simply need to map our variables to the imported map
        //   Point's grids and set their offset appropriately.


        doubleGrid* chargeGrid = currMapPoint->getChargeGridModifiable();
        chargeGrid->setOffset(atomGridOffset);

        vectorGrid* forceGrid = currMapPoint->getForceGridModifiable();
        forceGrid->setOffset(atomGridOffset);

        matrixGrid* dipoleGrid = currMapPoint->getDipoleGridModifiable();
        dipoleGrid->setOffset(atomGridOffset);

        d_interpolatingSpline.evaluateThroughSecondDerivative(splineValues,
                                                              baseLevel,
                                                              firstDerivative,
                                                              secondDerivative);

//                SimpleGrid<double>          chargeGrid(SupportVector,
//                                                       atomGridOffset,
//                                                       MDConstants::IV_ZERO,
//                                                       0);
//                SimpleGrid<SCIRun::Vector>  forceGrid(SupportVector,
//                                                      atomGridOffset,
//                                                      MDConstants::IV_ZERO,
//                                                      0);
//                SimpleGrid<Uintah::Matrix3> dipoleGrid(SupportVector,
//                                                       atomGridOffset,
//                                                       MDConstants::IV_ZERO,
//                                                       0);


        double kX = kReal.x();
        double kY = kReal.y();
        double kZ = kReal.z();
        for (size_t xIndex = 0; xIndex < support; ++xIndex) {
          double   Sx = baseLevel[xIndex].x();
          double  dSx = firstDerivative[xIndex].x()*kX;
          double d2Sx = secondDerivative[xIndex].x()*kX*kX;
          for (size_t yIndex = 0; yIndex < support; ++yIndex) {
            double   Sy = baseLevel[yIndex].y();
            double  dSy = firstDerivative[yIndex].y()*kY;
            double d2Sy = secondDerivative[yIndex].y()*kY*kY;
            double   Sx_Sy =  Sx*Sy;
            double  dSx_Sy = dSx*Sy;
            double  Sx_dSy = Sx*dSy;
            double dSx_dSy = dSx*dSy;
            for (size_t zIndex = 0; zIndex < support; ++zIndex) {
              double   Sz = baseLevel[zIndex].z();
              double  dSz = firstDerivative[zIndex].z()*kZ;
              double d2Sz = secondDerivative[zIndex].z()*kZ*kZ;
              double   Sx_Sz =   Sx*Sz;
              double   Sy_Sz =   Sy*Sz;
              double dSx_dSy_Sz = dSx_dSy*Sz;
              double dSx_Sy_dSz = dSx_Sy*dSz;
              double Sx_dSy_dSz = Sx_dSy*dSz;
              // Pure spline multiplication for Q/Q interactions
              (*chargeGrid)(xIndex,yIndex,zIndex) = Sx_Sy*Sz;
              // dPhi/du for reciprocal contribution to electric field
              // (also charge contribution to force)
              (*forceGrid)(xIndex,yIndex,zIndex)  =
                  Vector(dSx*Sy_Sz,dSy*Sx_Sz,dSz*Sx_Sy);
              // dE/du for reciprocal dipole contribution to the force
              (*dipoleGrid)(xIndex,yIndex,zIndex) =
                  Matrix3(d2Sx*Sy_Sz,dSx_dSy_Sz,dSx_Sy_dSz,
                          dSx_dSy_Sz,d2Sy*Sx_Sz,Sx_dSy_dSz,
                          dSx_Sy_dSz,Sx_dSy_dSz,d2Sz*Sx_Sy);
            } // zIndex
          } // yIndex
        } // xIndex

        /*
         *SPMEMapPoint holds ONLY the spline and related derivatives;
         *charge and dipole values and derivatives
         *don't get baked in, so that we don't need to recalculate
         *these quantities within the inner loop.
         */
//                SPMEMapPoint currentMapPoint(ID,
//                                             atomGridOffset,
//                                             chargeGrid,
//                                             forceGrid,
//                                             dipoleGrid);
//                gridMap->push_back(currentMapPoint);

      } // atom
    } // material
  } // patch
} // method

// Called from calculatePreTransformDipole
void SPME::mapChargeToGridDipole(SPMEPatch*                     spmePatch,
                                 const spmeMapVector*           gridMap,
                                 ParticleSubset*                pset,
                                 double                         charge,
                                 constParticleVariable<Vector>& p_Dipole,
                                 CoordinateSystem*              coordSys) {

  // grab local Q grid
  SimpleGrid<dblcomplex>*   Q_patchLocal    = spmePatch->getQ();
  IntVector                 patchOffset     = spmePatch->getGlobalOffset();
  //IntVector               patchExtent = Q_patchLocal->getExtentWithGhost();

  // Method global vector to avoid unnecessary temporary creation
  SCIRun::Vector            d_dot_nabla;
  ParticleSubset::iterator particleIt;
  for (particleIt = pset->begin(); particleIt != pset->end(); ++particleIt) {
    particleIndex atom = *particleIt;

    //SPMEMapPoint
    const doubleGrid*       chargeMap = (*gridMap)[atom].getChargeGrid();
    const vectorGrid*       gradMap   = (*gridMap)[atom].getGradientGrid();
    const SCIRun::Vector    dipole    = p_Dipole[atom];

    // Pull offset from the mapPoint, not each individual grid, since they
    // should all be the same and should be equivalent to map's offset for
    // the particle.
    IntVector QAnchor = chargeMap->getOffset();
    IntVector supportExtent = chargeMap->getExtents();
    IntVector Base = QAnchor - patchOffset;

    int x_Base = Base[0];
    int y_Base = Base[1];
    int z_Base = Base[2];

    int xExtent = supportExtent[0];
    int yExtent = supportExtent[1];
    int zExtent = supportExtent[2];

    coordSys->toReduced(dipole,d_dot_nabla);

    for (int xmask = 0; xmask < xExtent; ++xmask) {
      int x_anchor = x_Base + xmask;
      for (int ymask = 0; ymask < yExtent; ++ymask) {
        int y_anchor = y_Base + ymask;
        for (int zmask = 0; zmask < zExtent; ++zmask) {
          int z_anchor = z_Base + zmask;
          dblcomplex val = charge*(*chargeMap)(xmask,ymask,zmask)
                          + Dot(d_dot_nabla,(*gradMap)(xmask,ymask,zmask));
          (*Q_patchLocal)(x_anchor, y_anchor, z_anchor) += val;
        }
      }
    }
  }
}

void SPME::calculateRealspaceTholeDipole(const ProcessorGroup*      pg,
                                         const PatchSubset*         patches,
                                         const MaterialSubset*      materials,
                                               DataWarehouse*       subOldDW,
                                               DataWarehouse*       subNewDW,
                                         const SimulationStateP*    sharedState,
                                         const MDLabel*             label,
                                               CoordinateSystem*    coordSys,
                                               DataWarehouse*       parentOldDW)
{
  size_t numPatches = patches->size();
  size_t numTypes   = materials->size();

  // Static constants we'll use but that are calculated once
  double twobeta2 = 2.0 * d_ewaldBeta * d_ewaldBeta;
  double Pi = acos(-1.0);
  double rootPi = sqrt(Pi);
  double one_over_betarootPi = 1.0/(d_ewaldBeta*rootPi);
  SCIRun::Vector ZERO_VECTOR(0.0, 0.0, 0.0);

  double cutoff2 = d_electrostaticRadius * d_electrostaticRadius;
  double realElectrostaticEnergy = 0;
  Matrix3 realElectrostaticStress = Matrix3(0.0,0.0,0.0,
                                            0.0,0.0,0.0,
                                            0.0,0.0,0.0);

  // Method global vector to catch the distance offset and avoid lots of
  // spurious temporary vector creations.
  SCIRun::Vector offset;

  // Step through all the patches on this thread
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* patch = patches->get(patchIndex);

    // step through the materials for the reference sites
    for (size_t localIndex = 0; localIndex < numTypes; ++localIndex) {
      int    localType   = materials->get(localIndex);
      double localCharge = (*sharedState)->getMDMaterial(localType)->getCharge();
      double localPol    = (*sharedState)->getMDMaterial(localType)
                                          ->getPolarizability();
      ParticleSubset* localSubset = parentOldDW->getParticleSubset(localType,
                                                                   patch);
      constParticleVariable<Point>  localX;
      constParticleVariable<long64> localID;
      constParticleVariable<Vector> localMu;
      parentOldDW->get(localX, label->global->pX, localSubset);
      parentOldDW->get(localID, label->global->pID, localSubset);
      subOldDW->get(localMu, label->electrostatic->pMu, localSubset);

      size_t numLocalAtoms = localSubset->numParticles();

      ParticleVariable<SCIRun::Vector> localForce, localField;

      subNewDW->allocateAndPut(localForce,
                             label->electrostatic->pF_electroReal_preReloc,
                             localSubset);

      subNewDW->allocateAndPut(localField,
                             label->electrostatic->pE_electroReal_preReloc,
                             localSubset);

      for (size_t Index = 0; Index < numLocalAtoms; ++ Index) {
        localForce[Index] = ZERO_VECTOR;
        localField[Index] = ZERO_VECTOR;
      }

      for (size_t neighborIndex = 0; neighborIndex < numTypes; ++neighborIndex)
      {
        int    neighborType   = materials->get(neighborIndex);
        double neighborCharge = (*sharedState)->getMDMaterial(neighborType)
                                                ->getCharge();
        double neighborPol    = (*sharedState)->getMDMaterial(neighborType)
                                                ->getPolarizability();
        double sqrt_alphai_alphaj = sqrt(localPol*neighborPol);
        ParticleSubset* neighborSubset;
        neighborSubset =  parentOldDW->getParticleSubset(neighborType,
                                                         patch,
                                                         Ghost::AroundNodes,
                                                         d_electrostaticGhostCells,
                                                         label->global->pX);

        constParticleVariable<Point>  neighborX;
        constParticleVariable<long64> neighborID;
        constParticleVariable<Vector> neighborMu;
        parentOldDW->get(neighborX, label->global->pX, neighborSubset);
        parentOldDW->get(neighborID, label->global->pID, neighborSubset);
        subOldDW->get(neighborMu, label->electrostatic->pMu, neighborSubset);

        size_t numNeighborAtoms = neighborSubset->numParticles();

        // loop over the local atoms
        for (size_t patchAtom=0; patchAtom < numLocalAtoms; ++patchAtom) {
          SCIRun::Vector atomDipole = localMu[patchAtom];
          localForce[patchAtom]=ZERO_VECTOR;
          localField[patchAtom]=ZERO_VECTOR;
          // loop over the neighbors
          for (size_t neighborAtom=0; neighborAtom < numNeighborAtoms; ++neighborAtom) {
            // Ensure i != j
            if (localID[patchAtom] != neighborID[neighborAtom]) {
              // d_offsetProxy contains the proper implementation for distance offsets based on
              // orthorhombic nature of the cell and the current inverse cell
              coordSys->minimumImageDistance(neighborX[neighborAtom],localX[patchAtom],offset);
//              SCIRun::Vector atomicDistanceVector = neighborX[neighborIdx]-localX[localIdx];
              // Periodic boundary condition; should eventually check against actual BC of system
//              atomicDistanceVector -= (atomicDistanceVector / box).vec_rint() * box; // For orthorhombic only
              double radius2 = offset.length2();

              // only calculate if neighbor within spherical cutoff around local atom
              if (radius2 < cutoff2 ) {
                // double a_Thole = forcefield->getTholeScreeningParameter();
                double a_Thole = 0.2;
                SCIRun::Vector neighborDipole = neighborMu[neighborAtom];
                double radius = sqrt(radius2);
                double B0, B1, B2, B3;
                generatePointScreeningMultipliers(radius, B0, B1, B2, B3);
                double T1, T2, T3;
                generateTholeScreeningMultipliers(a_Thole,
                                                  sqrt_alphai_alphaj,
                                                  radius,
                                                  T1, T2, T3);
                double G0, G1_mu_q, G1_mu_mu, G2, mu_jDOTr_ij;
                SCIRun::Vector gradG0, gradG1, gradG2;
                generateDipoleFunctionalTerms(localCharge, neighborCharge,
                                              atomDipole, neighborDipole,
                                              offset,
                                              mu_jDOTr_ij,
                                              G0, G1_mu_q, G1_mu_mu, G2,
                                              gradG0, gradG1, gradG2);
                // Dipole only terms:  G1_mu_mu, G2, gradG2
                // FIXME  The below setup is designed to apply Thole screening
                //        ONLY to pure Dipole terms.  This may not be the
                //        proper way to do it for forcefields other than
                //        Lucretius.  However, differences should be relatively
                //        small.
                realElectrostaticEnergy += ( B0*G0 + B1*G1_mu_q +
                                            (B1-T1)*G1_mu_mu + (B2-T2)*G2);
                SCIRun::Vector localForceVector = offset*(G0*B1
                                                        + G1_mu_q*B2 + G1_mu_mu * (B2 - T2)
                                                        + G2 * (B3-T3))
                                                 +(B1 * gradG1 + (B2-T2) * gradG2);
                localForce[patchAtom] += localForceVector;
                localField[patchAtom] += (localCharge*B1-mu_jDOTr_ij*B2)*offset
                                         + B1*neighborDipole;
                realElectrostaticStress += OuterProduct(offset, localForceVector);
              } // Interaction within cutoff
            } // If atoms are different
          } // Loop over neighbors
        } // Loop over local atoms
      } // Loop over neighbor materials
    } // Loop over local materials
  } // Loop over patches
  // put updated values for reduction variables into the DW
  subNewDW->put(sum_vartype(0.5 * realElectrostaticEnergy),
              label->electrostatic->rElectrostaticRealEnergy);
  subNewDW->put(matrix_sum(0.5 * realElectrostaticStress),
              label->electrostatic->rElectrostaticRealStress);
  return;
} // End method

// Called from the SPME::calculate subscheduler as part of the polarizable
// iteration loop.
void SPME::calculateRealspacePointDipole(const ProcessorGroup*      pg,
                                         const PatchSubset*         patches,
                                         const MaterialSubset*      materials,
                                               DataWarehouse*       subOldDW,
                                               DataWarehouse*       subNewDW,
                                         const SimulationStateP*    sharedState,
                                         const MDLabel*             label,
                                               CoordinateSystem*    coordSys,
                                               DataWarehouse*       parentOldDW)
{
  size_t numPatches = patches->size();
  size_t numMaterials = materials->size();

  // Static constants we'll use but that are calculated once
  double twobeta2 = 2.0 * d_ewaldBeta * d_ewaldBeta;
  double Pi = acos(-1.0);
  double rootPi = sqrt(Pi);
  double one_over_betarootPi = 1.0/(d_ewaldBeta*rootPi);
  SCIRun::Vector ZERO_VECTOR(0.0, 0.0, 0.0);

  double cutoff2 = d_electrostaticRadius * d_electrostaticRadius;
  double realElectrostaticEnergy = 0;
  Matrix3 realElectrostaticStress = Matrix3(0.0,0.0,0.0,
                                            0.0,0.0,0.0,
                                            0.0,0.0,0.0);

  // Method global vector to catch the distance offset and avoid lots of
  // spurious temporary vector creations.
  SCIRun::Vector offset;

  // Step through all the patches on this thread
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* patch = patches->get(patchIndex);

    // step through the materials for the reference sites
    for (size_t localIndex = 0; localIndex < numMaterials; ++localIndex) {
      int atomType = materials->get(localIndex);
      double atomCharge = (*sharedState)->getMDMaterial(atomType)->getCharge();
      ParticleSubset* atomSubset = parentOldDW->getParticleSubset(atomType,
                                                                  patch);
      constParticleVariable<Point>  localX;
      constParticleVariable<long64> localID;
      constParticleVariable<Vector> localMu;
      parentOldDW->get(localX, label->global->pX, atomSubset);
      parentOldDW->get(localID, label->global->pID, atomSubset);
      subOldDW->get(localMu, label->electrostatic->pMu, atomSubset);

      size_t numLocalAtoms = atomSubset->numParticles();

      ParticleVariable<SCIRun::Vector> localForce, localField;

      subNewDW->allocateAndPut(localForce,
                             label->electrostatic->pF_electroReal_preReloc,
                             atomSubset);

      subNewDW->allocateAndPut(localField,
                             label->electrostatic->pE_electroReal_preReloc,
                             atomSubset);

      for (size_t Index = 0; Index < numLocalAtoms; ++ Index) {
        localForce[Index] = ZERO_VECTOR;
        localField[Index] = ZERO_VECTOR;
      }

      for (size_t neighborIndex = 0; neighborIndex < numMaterials; ++neighborIndex) {
        int neighborType = materials->get(neighborIndex);
        double neighborCharge = (*sharedState)->getMDMaterial(neighborType)->getCharge();
        ParticleSubset* neighborSubset;
        neighborSubset =  parentOldDW->getParticleSubset(neighborType,
                                                         patch,
                                                         Ghost::AroundNodes,
                                                         d_electrostaticGhostCells,
                                                         label->global->pX);

        constParticleVariable<Point>  neighborX;
        constParticleVariable<long64> neighborID;
        constParticleVariable<Vector> neighborMu;
        parentOldDW->get(neighborX, label->global->pX, neighborSubset);
        parentOldDW->get(neighborID, label->global->pID, neighborSubset);
        subOldDW->get(neighborMu, label->electrostatic->pMu, neighborSubset);

        size_t numNeighborAtoms = neighborSubset->numParticles();

        // loop over the local atoms
        for (size_t patchAtom=0; patchAtom < numLocalAtoms; ++patchAtom) {
          SCIRun::Vector atomDipole = localMu[patchAtom];
          localForce[patchAtom]=ZERO_VECTOR;
          localField[patchAtom]=ZERO_VECTOR;
          // loop over the neighbors
          for (size_t neighborAtom=0; neighborAtom < numNeighborAtoms; ++neighborAtom) {
            // Ensure i != j
            if (localID[patchAtom] != neighborID[neighborAtom]) {
              // d_offsetProxy contains the proper implementation for distance offsets based on
              // orthorhombic nature of the cell and the current inverse cell
              coordSys->minimumImageDistance(neighborX[neighborAtom],localX[patchAtom],offset);
//              SCIRun::Vector atomicDistanceVector = neighborX[neighborIdx]-localX[localIdx];
              // Periodic boundary condition; should eventually check against actual BC of system
//              atomicDistanceVector -= (atomicDistanceVector / box).vec_rint() * box; // For orthorhombic only
              double radius2 = offset.length2();

              // only calculate if neighbor within spherical cutoff around local atom
              if (radius2 < cutoff2 ) {
                SCIRun::Vector neighborDipole = neighborMu[neighborAtom];
                double radius = sqrt(radius2);
                double B0, B1, B2, B3;
                generatePointScreeningMultipliers(radius, B0, B1, B2, B3);
                double G0, G1_mu_q, G1_mu_mu, G2, mu_jDOTr_ij;
                SCIRun::Vector gradG0, gradG1, gradG2;
                generateDipoleFunctionalTerms(atomCharge, neighborCharge,
                                              atomDipole, neighborDipole,
                                              offset,
                                              mu_jDOTr_ij,
                                              G0, G1_mu_q, G1_mu_mu, G2,
                                              gradG0, gradG1, gradG2);
                double G1 = G1_mu_q + G1_mu_mu; // No dipole screening term
                realElectrostaticEnergy         += (B0*G0 + B1*G1 + B2*G2);
                SCIRun::Vector localForceVector  = offset*(G0*B1+G1*B2+G2*B3)
                                                     + (B1*gradG1 + B2*gradG2);
                localForce[patchAtom]   += localForceVector;
                localField[patchAtom]   += (atomCharge*B1-mu_jDOTr_ij*B2)*offset
                                            + B1*neighborDipole;
                realElectrostaticStress += OuterProduct(offset, localForceVector);
              } // Interaction within cutoff
            } // If atoms are different
          } // Loop over neighbors
        } // Loop over local atoms
      } // Loop over neighbor materials
    } // Loop over local materials
  } // Loop over patches
  // put updated values for reduction variables into the DW
  subNewDW->put(sum_vartype(0.5 * realElectrostaticEnergy),
              label->electrostatic->rElectrostaticRealEnergy);
  subNewDW->put(matrix_sum(0.5 * realElectrostaticStress),
              label->electrostatic->rElectrostaticRealStress);
  return;
} // End method

// --> Fourier dipole calculation components
void SPME::calculatePreTransformDipole(const ProcessorGroup*    pg,
                                       const PatchSubset*       patches,
                                       const MaterialSubset*    materials,
                                       DataWarehouse*           oldDW,
                                       DataWarehouse*           newDW,
                                       const SimulationStateP*  simState,
                                       const MDLabel*           label,
                                       CoordinateSystem*        coordSys)
{
  size_t numPatches   = patches->size();
  size_t numAtomTypes = materials->size();
  Uintah::Matrix3 inverseUnitCell = coordSys->getInverseCell();


  for (size_t currPatch = 0; currPatch < numPatches; ++currPatch) {
    const Patch* patch = patches->get(currPatch);

    // Extract current SPMEPatch
    SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;

    // SPMEPatches map to Patches 1:1; shouldn't need to lock for anything
    // that's local to a patch
    SimpleGrid<dblcomplex>* Q_threadLocal = currentSPMEPatch->getQ();

    //Initialize once before mapping any type of atoms
    Q_threadLocal->initialize(dblcomplex(0.0,0.0));
    for (size_t currType = 0; currType < numAtomTypes; ++currType) {

      int       atomType    = materials->get(currType);
      double    atomCharge  = (*simState)->getMDMaterial(atomType)->getCharge();

      ParticleSubset* atomSet = oldDW->getParticleSubset(atomType, patch);

      constParticleVariable<Vector> p_Dipole;
      oldDW->get(p_Dipole, label->electrostatic->pMu, atomSet);

      std::vector<SPMEMapPoint>* gridMap;
      gridMap = currentSPMEPatch->getChargeMap(atomType);

      SPME::mapChargeToGridDipole(currentSPMEPatch,
                                  gridMap,
                                  atomSet,
                                  atomCharge,
                                  p_Dipole,
                                  coordSys);
      // Dummy variable for maintaining graph layout.
      PerPatch<int> preTransformDep(1);
      newDW->put(preTransformDep,
                 label->SPME_dep->dPreTransform,
                 atomType,
                 patch);

    } // end Atom Type Loop



    } // end Patch Loop

    // TODO keep an eye on this to make sure it works like we think it should
    if (Thread::self()->myid() == 0) {
      d_Q_nodeLocal->initialize(dblcomplex(0.0, 0.0));
    }
}

void SPME::calculatePostTransformDipole(const ProcessorGroup*   pg,
                                        const PatchSubset*      patches,
                                        const MaterialSubset*   materials,
                                        DataWarehouse*          oldDW,
                                        DataWarehouse*          newDW,
                                        const SimulationStateP* simState,
                                        const MDLabel*          label,
                                        CoordinateSystem*       coordSystem) {
  size_t numPatches     = patches->size();
  size_t numAtomTypes   = materials->size();

  Uintah::Matrix3 inverseCell = coordSystem->getInverseCell();
  std::vector<SCIRun::Vector> delU;

  for (size_t principle = 0; principle < 3; ++principle) {
    delU.push_back(inverseCell.getRow(principle)*d_kLimits(principle));
  }

  // Instantiate temporaries outside the inner loops
  SCIRun::Vector    Mu, de_dU;
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* patch = patches->get(patchIndex);
    SPMEPatch* spmePatch = d_spmePatchMap.find(patch->getID())->second;

    for (size_t typeIndex = 0; typeIndex < numAtomTypes; ++typeIndex) {
      int       atomType    = materials->get(typeIndex);
      double    charge      = (*simState)->getMDMaterial(atomType)->getCharge();

      ParticleSubset* particles = oldDW->getParticleSubset(atomType, patch);

      constParticleVariable<Vector>     pDipole;
      ParticleVariable<Vector>          pForceRecip;

      newDW->get(pDipole,
                 label->electrostatic->pMu_preReloc,
                 particles);
      newDW->allocateAndPut(pForceRecip,
                            label->electrostatic->pF_electroInverse_preReloc,
                            particles);
      std::vector<SPMEMapPoint>*    gridMap  = spmePatch->getChargeMap(atomType);

      SimpleGrid<dblcomplex>*   Q_patchLocal = spmePatch->getQ();
      SCIRun::IntVector         patchOffset  = spmePatch->getGlobalOffset();
      SCIRun::IntVector         patchExtent  = Q_patchLocal->getExtentWithGhost();

      ParticleSubset::iterator atomIter, atomBegin, atomEnd;
      atomBegin = particles->begin();
      atomEnd   = particles->end();

      for (atomIter = atomBegin; atomIter != atomEnd; ++atomIter) {
        size_t atom = *atomIter;

        const vectorGrid*   forceMap = (*gridMap)[atom].getForceGrid();
        const matrixGrid*   gradMap  = (*gridMap)[atom].getDipoleGrid();

        // Zero out de/dU
        de_dU   = MDConstants::V_ZERO;
        Mu      = pDipole[atom];

        SCIRun::Vector muDivU(Dot(Mu,delU[0]),Dot(Mu,delU[1]),Dot(Mu,delU[2]));

        SCIRun::IntVector   QAnchor         = forceMap->getOffset();
        SCIRun::IntVector   supportExtent   = forceMap->getExtents();
        SCIRun::IntVector   base            = QAnchor - patchOffset;

        int xBase   =   base[0];
        int yBase   =   base[1];
        int zBase   =   base[2];

        int xExtent =   supportExtent[0];
        int yExtent =   supportExtent[1];
        int zExtent =   supportExtent[2];

        for (int xmask = 0; xmask < xExtent; ++xmask ) {
          int xAnchor = xBase + xmask;
          for (int ymask = 0; ymask < yExtent; ++ymask) {
            int yAnchor = yBase + ymask;
            for (int zmask = 0; zmask < zExtent; ++zmask) {
              int zAnchor = zBase + zmask;
              const Uintah::Matrix3& dipoleMap = (*gradMap)(xmask, ymask, zmask);
              double QReal = std::real((*Q_patchLocal)(xAnchor, yAnchor, zAnchor));
              de_dU += QReal*(charge*(*forceMap)(xmask, ymask, zmask)
                              + muDivU[0] * dipoleMap.getColumn(0)
                              + muDivU[1] * dipoleMap.getColumn(1)
                              + muDivU[2] * dipoleMap.getColumn(2) );
            } // Loop over Z
          } // Loop over Y
        } // Loop over X
        pForceRecip[atom] = -de_dU[0]*delU[0] -de_dU[1]*delU[1] - de_dU[2]*delU[2];
      } // Loop over Atoms

//      // Calculate electrostatic reciprocal contribution to F_ij(r)
//      SPME::mapForceFromGridDipole(spmePatch,
//                                   gridMap,
//                                   particles,
//                                   charge,
//                                   pDipole,
//                                   pForceRecip,
//                                   coordSystem);

    } // Loop over atom types
  } // Loop over patches
}

void SPME::mapForceFromGridDipole(const SPMEPatch*                  spmePatch,
                                  const spmeMapVector*              gridMap,
                                  ParticleSubset*                   particles,
                                  double                            charge,
                                  const ParticleVariable<Vector>&   pDipole,
                                  ParticleVariable<Vector>&         pForceRecip,
                                  CoordinateSystem*                 coordSystem) {

  SimpleGrid<dblcomplex>*   Q_patchLocal = spmePatch->getQ();
  IntVector                 patchOffset  = spmePatch->getGlobalOffset();
  IntVector                 patchExtent  = Q_patchLocal->getExtentWithGhost();

  Uintah::Matrix3 inverseCell = coordSystem->getInverseCell();
  std::vector<SCIRun::Vector> delU;

  for (size_t principle = 0; principle < 3; ++principle) {
    delU.push_back(inverseCell.getRow(principle)*d_kLimits(principle));
  }

  ParticleSubset::iterator atomIter, atomBegin, atomEnd;
  atomBegin = particles->begin();
  atomEnd   = particles->end();

  SCIRun::Vector    Mu;
  for (atomIter = atomBegin; atomIter != atomEnd; ++atomIter) {
    size_t atom = *atomIter;

    const vectorGrid*   forceMap    = (*gridMap)[atom].getForceGrid();
    const matrixGrid*   gradMap     = (*gridMap)[atom].getDipoleGrid();

    SCIRun::Vector          de_du(0.0);
    Mu  =   pDipole[atom];

    IntVector   QAnchor         =   forceMap->getOffset();
    IntVector   supportExtent   =   forceMap->getExtents();
    IntVector   Base            =   QAnchor - patchOffset;

    int         xBase           =   Base[0];
    int         yBase           =   Base[1];
    int         zBase           =   Base[2];

    int         xExtent         =   supportExtent[0];
    int         yExtent         =   supportExtent[1];
    int         zExtent         =   supportExtent[2];

    SCIRun::Vector muDotU(Dot(Mu,delU[0]),Dot(Mu,delU[1]),Dot(Mu,delU[2]));

    for (int xmask = 0; xmask < xExtent; ++xmask) {
      int xAnchor = xBase + xmask;
      for (int ymask = 0; ymask < yExtent; ++ymask) {
        int yAnchor = yBase + ymask;
        for (int zmask = 0; zmask < zExtent; ++zmask) {
          int zAnchor = zBase + zmask;
          Uintah::Matrix3 dipoleMap = (*gradMap)(xmask, ymask, zmask);
          double QReal = std::real((*Q_patchLocal)(xAnchor, yAnchor, zAnchor));
          de_du += QReal*(charge*(*forceMap)(xmask, ymask, zmask)
                          + muDotU[0] * dipoleMap.getColumn(0)
                          + muDotU[1] * dipoleMap.getColumn(1)
                          + muDotU[2] * dipoleMap.getColumn(2) );
        } // Loop over Z
      } // Loop over Y
    } // Loop over X
    pForceRecip[atom] = -de_du[0]*delU[0] -de_du[1]*delU[1] -de_du[2]*delU[2];
  } // Loop over atoms
}

void SPME::dipoleUpdateFieldAndStress(const ProcessorGroup* pg,
                                      const PatchSubset*    patches,
                                      const MaterialSubset* materials,
                                      DataWarehouse*        oldDW,
                                      DataWarehouse*        newDW,
                                      const MDLabel*        label,
                                      CoordinateSystem*     coordSystem) {

  size_t numPatches = patches->size();
  size_t numAtomTypes = materials->size();

  Uintah::Matrix3 inverseCell = coordSystem->getInverseCell();
  std::vector<SCIRun::Vector> delU;

  for (size_t principle = 0; principle < 3; ++principle) {
    delU.push_back(inverseCell.getRow(principle)*d_kLimits(principle));
  }

  Uintah::Matrix3 localStressAddendum;
  SCIRun::Vector  potentialDeriv;
  SCIRun::Vector  currentDipole;

  // Step through all the patches on this thread
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch*    patch               = patches->get(patchIndex);
    SPMEPatch*      spmePatch           =
                      d_spmePatchMap.find(patch->getID())->second;

    SimpleGrid<dblcomplex>* QConvoluted = spmePatch->getQ();
    IntVector               patchOffset = spmePatch->getGlobalOffset();

    // step through the materials on this patch
    for (size_t atomType = 0; atomType < numAtomTypes; ++atomType) {
      int             currType = materials->get(atomType);

      std::vector<SPMEMapPoint>* gridMap;
                      gridMap  = spmePatch->getChargeMap(currType);
      ParticleSubset* pset     = oldDW->getParticleSubset(currType, patch);

      constParticleVariable<Vector> pDipole;
      ParticleVariable<Vector>      pRecipField;
      oldDW->get(pDipole, label->electrostatic->pMu, pset);
      newDW->allocateAndPut(pRecipField,
                            label->electrostatic->pE_electroInverse_preReloc, pset);

      ParticleSubset::iterator pSetIter, pSetBegin, pSetEnd;
      pSetBegin = pset->begin();
      pSetEnd   = pset->end();
      for (pSetIter = pSetBegin; pSetIter != pSetEnd; ++pSetIter) {
        size_t  atom                = *pSetIter;
                pRecipField[atom]  *= 0.0;

        const vectorGrid* potentialDerivMap;
        potentialDerivMap = (*gridMap)[atom].getPotentialDerivativeGrid();


        IntVector QAnchor           = potentialDerivMap->getOffset();
        IntVector supportExtent     = potentialDerivMap->getExtents();
        IntVector Base              = QAnchor - patchOffset;

        int xBase = Base[0];
        int yBase = Base[1];
        int zBase = Base[2];

        int xExtent = supportExtent[0];
        int yExtent = supportExtent[1];
        int zExtent = supportExtent[2];

        for (int xmask = 0; xmask < xExtent; ++xmask) {
          int xAnchor = xBase + xmask;
          for (int ymask = 0; ymask < yExtent; ++ymask) {
            int yAnchor = yBase + ymask;
            for (int zmask = 0; zmask < zExtent; ++zmask) {
              int zAnchor = zBase + zmask;
              potentialDeriv = (*potentialDerivMap)(xmask, ymask, zmask);
              double    Q = std::abs((*QConvoluted)(xAnchor, yAnchor, zAnchor));
              pRecipField[atom]    += Q * ( potentialDeriv[0] * delU[0] +
                                            potentialDeriv[1] * delU[1] +
                                            potentialDeriv[2] * delU[2] );
            }
          }
        }
        currentDipole = pDipole[atom];
        localStressAddendum += OuterProduct(pRecipField[atom],currentDipole);
      } // Iterate through particle set
    } // Iterate through atom types
  } // iterate through patches

  // Update it with the current addendum factor
  newDW->put(matrix_sum(0.5 * localStressAddendum),
             label->electrostatic->rElectrostaticInverseStressDipole);
}

void SPME::calculateNewDipoles(const ProcessorGroup*    pg,
                               const PatchSubset*       patches,
                               const MaterialSubset*    materials,
                                     DataWarehouse*     oldDW,
                                     DataWarehouse*     newDW,
                               const SimulationStateP*  sharedState,
                               const MDLabel*           label) {

  size_t numPatches = patches->size();
  size_t numMaterials = materials->size();

  double newMix = 1.0 - d_dipoleMixRatio;
  // Step through all the patches on this thread
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* patch = patches->get(patchIndex);
    // step through the materials for the reference sites
    for (size_t localIndex = 0; localIndex < numMaterials; ++localIndex) {
      int       atomType        = materials->get(localIndex);
      double    polarizability  = (*sharedState)->getMDMaterial(atomType)
                                                  ->getPolarizability();

      ParticleSubset* localSet = oldDW->getParticleSubset(atomType, patch);


      constParticleVariable<Vector> oldDipoles;
      oldDW->get(oldDipoles, label->electrostatic->pMu, localSet);

      SCIRun::Vector selfField = MDConstants::V_ZERO;
      // For Ewald E_s = (4.0/(3.0*sqrt(PI)))*beta^3*oldDipole

      // We need the field estimation from the current iteration
      constParticleVariable<SCIRun::Vector> reciprocalField, realField;
      newDW->get(reciprocalField, label->electrostatic->pE_electroInverse_preReloc, localSet);
      newDW->get(realField, label->electrostatic->pE_electroReal_preReloc, localSet);
      ParticleVariable<SCIRun::Vector> newDipoles;

      // And we'll be calculating into the new dipole container
      newDW->allocateAndPut(newDipoles, label->electrostatic->pMu_preReloc, localSet);
      size_t localAtoms = localSet->numParticles();
      for (size_t Index = 0; Index < localAtoms; ++ Index) {
        newDipoles[Index] = polarizability*(reciprocalField[Index] +
                                            realField[Index] +
                                            selfField); // Total field
        newDipoles[Index] *= (1.0 - d_dipoleMixRatio);
        newDipoles[Index] += d_dipoleMixRatio*oldDipoles[Index];
//        newDipoles[Index] = oldDipoles[Index];
      }
      ParticleVariable<Point> X;
      ParticleVariable<long64> ID;
      newDW->allocateAndPut(X, label->global->pX_preReloc, localSet);
      oldDW->copyOut(X, label->global->pX, localSet);
      newDW->allocateAndPut(ID, label->global->pID_preReloc, localSet);
      oldDW->copyOut(ID, label->global->pID, localSet);
      ParticleSubset* delset = scinew ParticleSubset(0, atomType, patch);
      newDW->deleteParticles(delset);
    }
  }


  // TODO fixme [APH]
}

void SPME::checkConvergence(const ProcessorGroup*       pg,
                            const PatchSubset*          patches,
                            const MaterialSubset*       materials,
                            DataWarehouse*              subOldDW,
                            DataWarehouse*              subNewDW,
                            const MDLabel*              label) {

  // Subroutine determines if polarizable component has converged
  double sumSquaredDeviation = 0.0;
  size_t numPatches = patches->size();
  size_t numAtomTypes = materials->size();
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* patch = patches->get(patchIndex);
    for (size_t typeIndex = 0; typeIndex < numAtomTypes; ++typeIndex) {
      int atomType = materials->get(typeIndex);
      ParticleSubset* atomSubset;
      atomSubset = subOldDW->getParticleSubset(atomType, patch);

      // Particle subsets are constant, so can get the subset once and index into it
      // for both old and new DW.
      constParticleVariable<SCIRun::Vector> oldDipole;
      constParticleVariable<SCIRun::Vector> newDipole;
      subOldDW->get(oldDipole,
                    label->electrostatic->pMu,
                    atomSubset);
      subNewDW->get(newDipole,
                    label->electrostatic->pMu_preReloc,
                    atomSubset);

      size_t numAtoms = atomSubset->numParticles();
      for (size_t atom = 0; atom < numAtoms; ++atom) {
        SCIRun::Vector deviation = newDipole[atom] - oldDipole[atom];
        sumSquaredDeviation += deviation.length2();
      } // Loop over atoms
    } // Loop over atom types
  } // Loop over patches

  subNewDW->put(sum_vartype(sumSquaredDeviation),
                label->electrostatic->rPolarizationDeviation);
//
//
//  if (!f_polarizable) {
//    return true;
//  } else {
//    double sumSquaredDeviation = 0.0;
//    size_t numPatches = patches->size();
//    // throw an exception for now, but eventually will check convergence here.
//    throw InternalError("Error: Polarizable force field not yet implemented!", __FILE__, __LINE__);
//  }

//  // TODO keep an eye on this to make sure it works like we think it should
//  if (Thread::self()->myid() == 0) {
//    d_Q_nodeLocal->initialize(dblcomplex(0.0, 0.0));
//  }
}






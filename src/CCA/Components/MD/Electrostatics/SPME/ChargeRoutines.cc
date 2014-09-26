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
 * SPME_newChargeOnly.cc
 *
 *  Created on: May 21, 2014
 *      Author: jbhooper
 */
//.......1.........2.........3.........4.........5.........6.........7.........8

#include <CCA/Ports/Scheduler.h>

#include <Core/Thread/Thread.h>

#include <CCA/Components/MD/Electrostatics/SPME/SPME.h>

using namespace Uintah;

void SPME::calculateRealspace(const ProcessorGroup*     pg,
                              const PatchSubset*        patches,
                              const MaterialSubset*     materials,
                                    DataWarehouse*      subOldDW,
                                    DataWarehouse*      subNewDW,
                              const SimulationStateP*   simState,
                              const MDLabel*            label,
                                    CoordinateSystem*   coordSys,
                                    DataWarehouse*      parentOldDW)
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
      double atomCharge = (*simState)->getMDMaterial(atomType)->getCharge();
      ParticleSubset* atomSubset = parentOldDW->getParticleSubset(atomType,
                                                                  patch);

      constParticleVariable<Point>  localX;
      constParticleVariable<long64> localID;
      parentOldDW->get(localX, label->global->pX, atomSubset);
      parentOldDW->get(localID, label->global->pID, atomSubset);

      size_t numLocalAtoms = atomSubset->numParticles();

      ParticleVariable<SCIRun::Vector> localForce;
      subNewDW->allocateAndPut(localForce,
                               label->electrostatic->pF_electroReal_preReloc,
                               atomSubset);

      for (size_t Index = 0; Index < numLocalAtoms; ++ Index) {
        localForce[Index] = ZERO_VECTOR;
      }

      for (size_t neighborIndex = 0; neighborIndex < numMaterials; ++neighborIndex) {
        int neighborType = materials->get(neighborIndex);
        double neighborCharge = (*simState)->getMDMaterial(neighborType)->getCharge();
        ParticleSubset* neighborSubset;
        neighborSubset = parentOldDW->getParticleSubset(neighborType,
                                                        patch,
                                                        Ghost::AroundNodes,
                                                        d_electrostaticGhostCells,
                                                        label->global->pX);

        constParticleVariable<Point>  neighborX;
        constParticleVariable<long64> neighborID;
        parentOldDW->get(neighborX, label->global->pX, neighborSubset);
        parentOldDW->get(neighborID, label->global->pID, neighborSubset);

        size_t numNeighborAtoms = neighborSubset->numParticles();

        // loop over the local atoms
        for (size_t patchAtom=0; patchAtom < numLocalAtoms; ++patchAtom) {
          localForce[patchAtom]=ZERO_VECTOR;
          // loop over the neighbors
          for (size_t neighborAtom=0; neighborAtom < numNeighborAtoms; ++neighborAtom) {
            // Ensure i != j
            if (localID[patchAtom] != neighborID[neighborAtom]) {
              coordSys->minimumImageDistance(neighborX[neighborAtom],
                                             localX[patchAtom],
                                             offset);
              // Periodic boundary condition; should eventually check against actual BC of system
              double radius2 = offset.length2();

              // only calculate if neighbor within spherical cutoff around local atom
              if (radius2 < cutoff2 ) {
                double rad2inv = 1.0/radius2;
                double radius = sqrt(radius2);
                double radinv = 1.0/radius;
                double betar = d_ewaldBeta*radius;
                double expnegbeta2r2_over_betarootpi = exp(-(d_ewaldBeta*d_ewaldBeta)*radius2)*one_over_betarootPi;
                double B0 = erfc(betar)*radinv;
                double B1 = (B0 + twobeta2*expnegbeta2r2_over_betarootpi);
                double G0 = atomCharge*neighborCharge;

                realElectrostaticEnergy += (B0*G0);
                SCIRun::Vector localForceVector = offset*(G0*B1);
                localForce[patchAtom] += localForceVector;
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

void SPME::generateChargeMap(const ProcessorGroup*    pg,
                             const PatchSubset*       patches,
                             const MaterialSubset*    materials,
                             DataWarehouse*           oldDW,
                             DataWarehouse*           newDW,
                             const MDLabel*           label,
                             CoordinateSystem*        coordSys)
{
    size_t          numPatches          =   patches->size();
    size_t          numMaterials        =   materials->size();
    Uintah::Matrix3 inverseUnitCell     =   coordSys->getInverseCell();

    // Step through all the patches on this thread
    for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
        const Patch* patch = patches->get(patchIndex);

        // Extract SPMEPatch which maps to our current patch
        SPMEPatch* currentSPMEPatch =
                        d_spmePatchMap.find(patch->getID())->second;

        // Step through all the materials in this patch
        for (size_t material = 0; material < numMaterials; ++material) {
            ParticleSubset* atomSubset;
            atomSubset = oldDW->getParticleSubset(material, patch);

            constParticleVariable<Point> atomPositions;
            constParticleVariable<long64> atomIDs;

            oldDW->get(atomPositions, label->global->pX, atomSubset);
            oldDW->get(atomIDs, label->global->pID, atomSubset);

            int numAtoms = atomSubset->numParticles();
            // Verify we have enough memory to hold the charge map for the
            // current atom type
            currentSPMEPatch->verifyChargeMapAllocation(numAtoms,material);

            // Pull the location for the SPMEPatch's copy of the charge map
            // for this material type
            std::vector<SPMEMapPoint>* gridMap;
            gridMap = currentSPMEPatch->getChargeMap(material);

            // begin loop to generate the charge map
            ParticleSubset::iterator atom, setBegin, setEnd;
            setBegin = atomSubset->begin();
            setEnd   = atomSubset->end();
            for (atom = setBegin; atom != setEnd; ++atom) {

                particleIndex atomIndex =  *atom;
                particleId    ID        =   atomIDs[atomIndex];
                Point         position  =   atomPositions[atomIndex];

                Vector atomGridCoordinates;
                coordSys->toReduced(position.asVector(),atomGridCoordinates);

                SCIRun::Vector kReal    =   d_kLimits.asVector();
                atomGridCoordinates    *=   kReal;

                SCIRun::IntVector atomGridOffset;
                atomGridOffset = IntVector(atomGridCoordinates.asPoint());
                SCIRun::Vector splineValues;
                splineValues = atomGridOffset.asVector() - atomGridCoordinates;

                size_t support = d_interpolatingSpline.getSupport();

                SCIRun::IntVector SupportVector(support, support, support);
                std::vector<SCIRun::Vector> baseLevel(support);
                std::vector<SCIRun::Vector> firstDerivative(support);
                std::vector<SCIRun::Vector> secondDerivative(support);

                d_interpolatingSpline.evaluateThroughSecondDerivative
                  (splineValues,baseLevel, firstDerivative, secondDerivative);

                SimpleGrid<double>          chargeGrid(SupportVector,
                                                       atomGridOffset,
                                                       MDConstants::IV_ZERO,
                                                       0);
                SimpleGrid<SCIRun::Vector>  forceGrid(SupportVector,
                                                      atomGridOffset,
                                                      MDConstants::IV_ZERO,
                                                      0);
                // Not used here, but still need to instantiate it
                // Should probably split the dipole/charge only routines
                // and make the dipole a subclass at some point, but this
                // works for now.
                SimpleGrid<Uintah::Matrix3> dipoleGrid(SupportVector,
                                                       atomGridOffset,
                                                       MDConstants::IV_ZERO,
                                                       0);

                double kX = kReal.x();
                double kY = kReal.y();
                double kZ = kReal.z();
                for (size_t xIndex = 0; xIndex < support; ++xIndex) {
                  double   Sx = baseLevel[xIndex].x();
                  double  dSx = firstDerivative[xIndex].x()*kX;
                  for (size_t yIndex = 0; yIndex < support; ++yIndex) {
                    double   Sy = baseLevel[yIndex].y();
                    double  dSy = firstDerivative[yIndex].y()*kY;
                    double   Sx_Sy =  Sx*Sy;
                    for (size_t zIndex = 0; zIndex < support; ++zIndex) {
                      double   Sz = baseLevel[zIndex].z();
                      double  dSz = firstDerivative[zIndex].z()*kZ;
                      double   Sx_Sz =   Sx*Sz;
                      double   Sy_Sz =   Sy*Sz;
                      // Pure spline multiplication for Q/Q interactions
                      chargeGrid(xIndex,yIndex,zIndex) = Sx_Sy*Sz;
                      // Charge contribution to force
                      forceGrid(xIndex,yIndex,zIndex)  =
                        Vector(dSx*Sy_Sz,dSy*Sx_Sz,dSz*Sx_Sy);
                    }
                  }
                }
                /*
                 *SPMEMapPoint holds ONLY the spline and related derivatives;
                 *charge and dipole values and derivatives
                 *don't get baked in, so that we don't need to recalculate
                 *these quantities within the inner loop.
                 */
                SPMEMapPoint currentMapPoint(ID,
                                             atomGridOffset,
                                             chargeGrid,
                                             forceGrid,
                                             dipoleGrid);
                gridMap->push_back(currentMapPoint);
            }
        }
    }
}

void SPME::calculatePreTransform(const ProcessorGroup*      pg,
                                 const PatchSubset*         patches,
                                 const MaterialSubset*      materials,
                                       DataWarehouse*       oldDW,
                                       DataWarehouse*       newDW,
                                 const SimulationStateP*    simState,
                                 const MDLabel*             label,
                                       CoordinateSystem*    coordSys,
                                       DataWarehouse*       parentOldDW)
{

  size_t numPatches   = patches->size();
  size_t numAtomTypes = materials->size();

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
        std::vector<SPMEMapPoint>* gridMap;
        gridMap = currentSPMEPatch->getChargeMap(atomType);

        SPME::mapChargeToGrid(currentSPMEPatch,
                              gridMap,
                              atomSet,
                              atomCharge,
                              coordSys);
      } // end Atom Type Loop

      // Dummy variable for maintaining graph layout.
      PerPatch<int> preTransformDep;
      newDW->put(preTransformDep,
                 label->SPME_dep->dPreTransform,
                 -1,
                 patch);

  } // end Patch Loop

//  // TODO keep an eye on this to make sure it works like we think it should
//  if (Thread::self()->myid() == 0) {
//    d_Q_nodeLocal->initialize(dblcomplex(0.0, 0.0));
//  }

//  bool replace = true;
//  // FIXME ????
//  newDW->transferFrom(oldDW, label->global->pX, patches, materials, replace);
}

void SPME::mapChargeToGrid(SPMEPatch*           spmePatch,
                           const spmeMapVector* gridMap,
                           ParticleSubset*      atomSet,
                           double               charge,
                           CoordinateSystem*    coordSystem) {
  // grab local Q grid
  SimpleGrid<dblcomplex>* Q_patchLocal = spmePatch->getQ();
  IntVector patchOffset = spmePatch->getGlobalOffset();
  //IntVector patchExtent = Q_patchLocal->getExtentWithGhost();

  ParticleSubset::iterator current, setStart, setEnd;
  setStart  =   atomSet->begin();
  setEnd    =   atomSet->end();
  for (current = setStart; current != setEnd; ++current) {
    particleIndex atom = *current;

    const doubleGrid* chargeMap = (*gridMap)[atom].getChargeGrid();

    // Location of the 0,0,0 origin for the charge map grid
    IntVector QAnchor       = chargeMap->getOffset();
    // Size of charge map grid
    IntVector supportExtent = chargeMap->getExtents();

    IntVector Base          = QAnchor - patchOffset;

    int x_Base = Base[0];
    int y_Base = Base[1];
    int z_Base = Base[2];

    int xExtent = supportExtent[0];
    int yExtent = supportExtent[1];
    int zExtent = supportExtent[2];

    for (int xmask = 0; xmask < xExtent; ++xmask) {
      int x_anchor = x_Base + xmask;
      for (int ymask = 0; ymask < yExtent; ++ymask) {
        int y_anchor = y_Base + ymask;
        for (int zmask = 0; zmask < zExtent; ++zmask) {
          int z_anchor = z_Base + zmask;
          // Local patch has no wrapping, we have ghost cells to write into
          dblcomplex val = charge * (*chargeMap)(xmask, ymask, zmask);
          (*Q_patchLocal)(x_anchor, y_anchor, z_anchor) += val;
        }
      }
    }
  }
}

void SPME::calculatePostTransform(const ProcessorGroup*   pg,
                                        const PatchSubset*      patches,
                                        const MaterialSubset*   materials,
                                        DataWarehouse*          oldDW,
                                        DataWarehouse*          newDW,
                                        const SimulationStateP* simState,
                                        const MDLabel*          label,
                                        CoordinateSystem*       coordSystem) {
  size_t numPatches     = patches->size();
  size_t numAtomTypes   = materials->size();
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* patch = patches->get(patchIndex);
    SPMEPatch* spmePatch = d_spmePatchMap.find(patch->getID())->second;

    for (size_t typeIndex = 0; typeIndex < numAtomTypes; ++typeIndex) {
      int       atomType    = materials->get(typeIndex);
      double    charge      = (*simState)->getMDMaterial(atomType)->getCharge();

      ParticleSubset* particles = oldDW->getParticleSubset(atomType, patch);

      ParticleVariable<Vector>          pForceRecip;

      newDW->allocateAndPut(pForceRecip,
                            label->electrostatic->pF_electroInverse_preReloc,
                            particles);
      std::vector<SPMEMapPoint>* gridMap = spmePatch->getChargeMap(atomType);

      // Calculate electrostatic reciprocal contribution to F_ij(r)
      SPME::mapForceFromGrid(spmePatch,
                             gridMap,
                             particles,
                             charge,
                             pForceRecip,
                             coordSystem);

    }
  }
}

void SPME::mapForceFromGrid(const SPMEPatch*                  spmePatch,
                           const spmeMapVector*              gridMap,
                            ParticleSubset*                   particles,
                            double                            charge,
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

  for (atomIter = atomBegin; atomIter != atomEnd; ++atomIter) {
    size_t atom = *atomIter;

    const vectorGrid*   forceMap    = (*gridMap)[atom].getForceGrid();
//    const matrixGrid*   gradMap     = (*gridMap)[atom].getDipoleGrid();

    SCIRun::Vector          de_du(0.0);

    IntVector   QAnchor         =   forceMap->getOffset();
    IntVector   supportExtent   =   forceMap->getExtents();
    IntVector   Base            =   QAnchor - patchOffset;

    int         xBase           =   Base[0];
    int         yBase           =   Base[1];
    int         zBase           =   Base[2];

    int         xExtent         =   supportExtent[0];
    int         yExtent         =   supportExtent[1];
    int         zExtent         =   supportExtent[2];

    for (int xmask = 0; xmask < xExtent; ++xmask) {
      int xAnchor = xBase + xmask;
      for (int ymask = 0; ymask < yExtent; ++ymask) {
        int yAnchor = yBase + ymask;
        for (int zmask = 0; zmask < zExtent; ++zmask) {
          int zAnchor = zBase + zmask;
          double QReal = std::real((*Q_patchLocal)(xAnchor, yAnchor, zAnchor));
          // Since charge is constant for any atom type, we can move it outside
          // the loop (see below).  We would need to fix this if we had
          // per-atom charges instead of per-type
          de_du += QReal*(*forceMap)(xmask, ymask, zmask);
        } // Loop over Z
      } // Loop over Y
    } // Loop over X
    pForceRecip[atom]  = -de_du[0]*delU[0] -de_du[1]*delU[1] -de_du[2]*delU[2];
    pForceRecip[atom] *= charge; // Factored out charge from above
  } // Loop over atoms
}

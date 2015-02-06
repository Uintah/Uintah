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

#include <Core/Thread/Thread.h>

#include <CCA/Components/MD/Electrostatics/SPME/SPME.h>

using namespace Uintah;
// Called from SPME::calculate directly
void SPME::generateChargeMap(const ProcessorGroup*    pg,
                             const PatchSubset*       patches,
                             const MaterialSubset*    atomTypes,
                                   DataWarehouse*     oldDW,
                                   DataWarehouse*     newDW,
                             const MDLabel*           label,
                                   CoordinateSystem*  coordSys)
{
  size_t            numPatches      =   patches->size();
  size_t            numAtomTypes    =   atomTypes->size();

  Uintah::Matrix3   inverseUnitCell =   coordSys->getInverseCell();
  SCIRun::Vector    kReal           =   d_kLimits.asVector();

  // Step through all the patches on this thread
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* patch = patches->get(patchIndex);

    // Extract SPMEPatch which maps to our current patch
    SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;

    // Step through all the atomTypes in this patch
    for (size_t typeIndex = 0; typeIndex < numAtomTypes; ++typeIndex) {
      int atomIndex = atomTypes->get(typeIndex);

      ParticleSubset* atomSubset;
      atomSubset = oldDW->getParticleSubset(atomIndex, patch);

      constParticleVariable<Point> atomPositions;
      constParticleVariable<long64> atomIDs;

      oldDW->get(atomPositions, label->global->pX, atomSubset);
      oldDW->get(atomIDs, label->global->pID, atomSubset);

      int numAtoms = atomSubset->numParticles();
      // Verify we have enough memory to hold the charge map for the
      // current atom type
      currentSPMEPatch->verifyChargeMapAllocation(numAtoms,atomIndex);

      // Pull the location for the SPMEPatch's copy of the charge map
      // for this material type
      std::vector<SPMEMapPoint>* gridMap;
      gridMap = currentSPMEPatch->getChargeMap(atomIndex);

      // begin loop to generate the charge map
      ParticleSubset::iterator atom;
      for (atom = atomSubset->begin(); atom != atomSubset->end(); ++atom)
      {
        particleIndex atomIndex =  *atom;
        particleId    ID        =   atomIDs[atomIndex];
        Point         position  =   atomPositions[atomIndex];

        SCIRun::Vector atomGridCoordinates;
        coordSys->toReduced(position.asVector(),atomGridCoordinates);
        atomGridCoordinates    *=   kReal;

        SCIRun::IntVector atomGridOffset;
        atomGridOffset = IntVector(atomGridCoordinates.asPoint());

        // Set map point to associate with he current particle
        SPMEMapPoint*   currMapPoint = &((*gridMap)[atomIndex]);
        currMapPoint->setParticleID(ID);
        currMapPoint->setGridOffset(atomGridOffset);

        SCIRun::Vector splineValues;
        splineValues = atomGridOffset.asVector() - atomGridCoordinates;

        size_t support = d_interpolatingSpline.getSupport();
        std::vector<SCIRun::Vector> baseLevel(support);
        std::vector<SCIRun::Vector> firstDerivative(support);
        std::vector<SCIRun::Vector> secondDerivative(support);

        // All simple grids are already created with:
        //   extents = SupportVector
        //   offset  = SCIRun::IntVector(0,0,0)
        // We simply need to map our variables to the imported map
        // Point's grids and set their offset appropriately.

        doubleGrid* chargeGrid = currMapPoint->getChargeGridModifiable();
        chargeGrid->setOffset(atomGridOffset);

        vectorGrid* forceGrid  = currMapPoint->getForceGridModifiable();
        forceGrid->setOffset(atomGridOffset);

        matrixGrid* dipoleGrid = currMapPoint->getDipoleGridModifiable();
        dipoleGrid->setOffset(atomGridOffset);

        d_interpolatingSpline.evaluateThroughSecondDerivative(splineValues,
                                                              baseLevel,
                                                              firstDerivative,
                                                              secondDerivative);
        double kX = kReal.x();
        double kY = kReal.y();
        double kZ = kReal.z();
        for (size_t xIndex = 0; xIndex < support; ++xIndex) {
          double        Sx = baseLevel[xIndex].x();
          double       dSx = firstDerivative[xIndex].x()*kX;
          for (size_t yIndex = 0; yIndex < support; ++yIndex) {
            double      Sy = baseLevel[yIndex].y();
            double     dSy = firstDerivative[yIndex].y()*kY;
            double   Sx_Sy =  Sx*Sy;
            for (size_t zIndex = 0; zIndex < support; ++zIndex) {
              double    Sz = baseLevel[zIndex].z();
              double   dSz = firstDerivative[zIndex].z()*kZ;
              double Sx_Sz =   Sx*Sz;
              double Sy_Sz =   Sy*Sz;
              // Pure spline multiplication for Q/Q interactions
              (*chargeGrid)(xIndex,yIndex,zIndex) = Sx_Sy*Sz;
              // dPhi/du Charge contribution to force
              (*forceGrid)(xIndex,yIndex,zIndex)  =
                       SCIRun::Vector(dSx*Sy_Sz,dSy*Sx_Sz,dSz*Sx_Sy);
            } // zIndex
          } // yIndex
        } // xIndex
      } // atom
    } // material
  } // patch
} // SPME::generateChargeMap

void SPME::mapChargeToGrid(      SPMEPatch*         spmePatch,
                           const spmeMapVector*     gridMap,
                                 ParticleSubset*    atomSet,
                                 double             charge,
                                 CoordinateSystem*  coordSystem)
{
  // grab local Q grid
  SimpleGrid<dblcomplex>*   Q_patchLocal    =   spmePatch->getQ();
  IntVector                 patchOffset     =   spmePatch->getGlobalOffset();

  ParticleSubset::iterator  currAtom;
  for (currAtom = atomSet->begin(); currAtom != atomSet->end(); ++currAtom) 
  {
    particleIndex atom = *currAtom;
    const doubleGrid* chargeMap = (*gridMap)[atom].getChargeGrid();

    // Pull offset from the mapPoint, not each individual grid, since they
    // should all be the same and should be equivalent to map's offset for
    // the particle.
    IntVector QAnchor       = chargeMap->getOffset();
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

void SPME::calculateRealspace(const ProcessorGroup*     pg,
                              const PatchSubset*        patches,
                              const MaterialSubset*     atomTypes,
                                    DataWarehouse*      subOldDW,
                                    DataWarehouse*      subNewDW,
                              const SimulationStateP*   simState,
                              const MDLabel*            label,
                                    CoordinateSystem*   coordSys,
                                    DataWarehouse*      parentOldDW)
{
  size_t    numPatches      = patches->size();
  size_t    numAtomTypes    = atomTypes->size();
  double    cutoff2         = d_electrostaticRadius * d_electrostaticRadius;

  double    realElectrostaticEnergy = 0;
  Matrix3   realElectrostaticStress = MDConstants::M3_0;
  
  // Method global vector to catch the distance offset and avoid lots of
  // spurious temporary vector creations.
  SCIRun::Vector offset;

  // Step through all the patches on this thread
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex)
  {
    const Patch* patch = patches->get(patchIndex);
    // step through the materials for the reference sites
    for (size_t localIndex = 0; localIndex < numAtomTypes; ++localIndex) {

      int               localType   = atomTypes->get(localIndex);
      double            localCharge = (*simState)->getMDMaterial(localType)
                                                   ->getCharge();
      ParticleSubset*   localSet    = parentOldDW->getParticleSubset(localType,
                                                                     patch);
      size_t            numLocal = localSet->numParticles();

      constParticleVariable<Point>  localX;
      parentOldDW->get(localX,  label->global->pX,  localSet);

      constParticleVariable<long64> localID;
      parentOldDW->get(localID, label->global->pID, localSet);

      ParticleVariable<SCIRun::Vector> localForce;
      subNewDW->allocateAndPut(localForce,
                               label->electrostatic->pF_electroReal_preReloc,
                               localSet);

//      // Zero out local patch's atoms' force.
//      for (size_t localAtom = 0; localAtom < numLocalAtoms; ++localAtom)
//      {
//        localForce[localAtom] = MDConstants::V_ZERO;
//      }

      for (size_t neighborIndex = 0; neighborIndex < numAtomTypes; ++neighborIndex)
      {
        int     neighborType    = atomTypes->get(neighborIndex);
        double  neighborCharge  = (*simState)->getMDMaterial(neighborType)
                                               ->getCharge();
        ParticleSubset* neighborSet;
        neighborSet = parentOldDW->getParticleSubset(neighborType,
                                                     patch,
                                                     Ghost::AroundCells,
                                                     d_electrostaticGhostCells,
                                                     label->global->pX);
        size_t numNeighbor      = neighborSet->numParticles();

        constParticleVariable<Point>  neighborX;
        parentOldDW->get(neighborX,  label->global->pX,  neighborSet);

        constParticleVariable<long64> neighborID;
        parentOldDW->get(neighborID, label->global->pID, neighborSet);

        // loop over the local atoms
        for (size_t localAtom=0; localAtom < numLocal; ++localAtom)
        {
          localForce[localAtom]=MDConstants::V_ZERO;
          // loop over the neighbors
          for (size_t neighborAtom=0; neighborAtom < numNeighbor; ++neighborAtom)
          { // Ensure i != j
            if (localID[localAtom] != neighborID[neighborAtom])
            {
              offset = neighborX[neighborAtom] - localX[localAtom];
//              coordSys->minimumImageDistance(localX[localAtom],
//                                             neighborX[neighborAtom],
//                                             offset);
              double radius2 = offset.length2();
              // only calculate if neighbor within spherical cutoff around 
              // local atom
              if (radius2 < cutoff2 ) {
                double radius = sqrt(radius2);
                double B0, B1, B2, B3;
                generatePointScreeningMultipliers(radius, B0, B1, B2, B3);
                double G0 = localCharge*neighborCharge;
                realElectrostaticEnergy += (B0*G0);
                SCIRun::Vector localForceVector = offset*(G0*B1);
                localForce[localAtom] += localForceVector;
                realElectrostaticStress += OuterProduct(offset,
                                                        localForceVector);
              } // Interaction within cutoff
            } // If atoms are different
          } // Loop over neighbors
          localForce[localAtom] *= 1.0; // Insert dimensionalization constant here
          realElectrostaticStress *= 1.0;
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

void SPME::calculatePreTransform(const ProcessorGroup*      pg,
                                 const PatchSubset*         patches,
                                 const MaterialSubset*      atomTypes,
                                       DataWarehouse*       oldDW,
                                       DataWarehouse*       newDW,
                                 const SimulationStateP*    simState,
                                 const MDLabel*             label,
                                       CoordinateSystem*    coordSys,
                                       DataWarehouse*       parentOldDW)
{
  size_t numPatches   = patches->size();
  size_t numAtomTypes = atomTypes->size();

  Uintah::Matrix3 inverseUnitCell = coordSys->getInverseCell();

  for (size_t currPatch = 0; currPatch < numPatches; ++currPatch)
  {
    const Patch* patch = patches->get(currPatch);
    // Extract current SPMEPatch
    SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;
    // SPMEPatches map to Patches 1:1; shouldn't need to lock for anything
    // that's local to a patch
    SimpleGrid<dblcomplex>* Q_threadLocal = currentSPMEPatch->getQ();

    //Initialize once before mapping any type of atoms
    Q_threadLocal->initialize(dblcomplex(0.0,0.0));
    for (size_t currType = 0; currType < numAtomTypes; ++currType)
    {
      int       atomType    = atomTypes->get(currType);
      double    atomCharge  = (*simState)->getMDMaterial(atomType)->getCharge();

      ParticleSubset* atomSet = parentOldDW->getParticleSubset(atomType, patch);

      std::vector<SPMEMapPoint>* gridMap;
      gridMap = currentSPMEPatch->getChargeMap(atomType);
      SPME::mapChargeToGrid(currentSPMEPatch,
                            gridMap,
                            atomSet,
                            atomCharge,
                            coordSys);
      // Dummy variable for maintaining graph layout.
       PerPatch<int> preTransformDep(1);
       newDW->put(preTransformDep,
                  label->SPME_dep->dPreTransform,
                  atomType,
                  patch);
    } // end Atom Type Loop
  } // end Patch Loop
}

void SPME::calculatePostTransform(const ProcessorGroup*   pg,
                                  const PatchSubset*      patches,
                                  const MaterialSubset*   atomTypes,
                                  DataWarehouse*          oldDW,
                                  DataWarehouse*          newDW,
                                  const SimulationStateP* simState,
                                  const MDLabel*          label,
                                  CoordinateSystem*       coordSystem) {
  size_t numPatches     = patches->size();
  size_t numAtomTypes   = atomTypes->size();

  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* patch = patches->get(patchIndex);
    SPMEPatch* spmePatch = d_spmePatchMap.find(patch->getID())->second;

    for (size_t typeIndex = 0; typeIndex < numAtomTypes; ++typeIndex) {
      int       atomType    = atomTypes->get(typeIndex);
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

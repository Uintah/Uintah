/*
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
 */

#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Patch.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Thread/Thread.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/Matrix3.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>

#include <sci_values.h>
#include <sci_defs/fftw_defs.h>

#include <CCA/Components/MD/MDSystem.h>
#include <CCA/Components/MD/MDLabel.h>
#include <CCA/Components/MD/SimpleGrid.h>
#include "../../../attic/SPME-busted/ShiftedCardinalBSpline.h"
#include "../../../attic/SPME-busted/SPME.h"
#include "../../../attic/SPME-busted/SPMEMapPoint.h"

#ifdef DEBUG
#include <Core/Util/FancyAssert.h>
#endif

#define IV_ZERO IntVector(0,0,0)

using namespace OldSPME;

extern SCIRun::Mutex cerrLock;

void SPME::generateChargeMapDipole(const ProcessorGroup*    pg,
                                   const PatchSubset*       patches,
                                   const MaterialSubset*    materials,
                                   DataWarehouse*           old_dw,
                                   DataWarehouse*           new_dw,
                                   coordinateSystem*        coordSys)
{
	size_t numPatches = patches->size();
	size_t numMaterials = materials->size();
	Uintah::Matrix3 inverseUnitCell = coordSys->getInverseCell();

	// Step through all the patches on this thread
	for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
		const Patch* patch = patches->get(patchIndex);

		// Extract SPMEPatch which maps to our current patch
		SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;

		// Step through all the materials in this patch
		for (size_t materialIndex = 0; materialIndex < numMaterials; ++materialIndex) {
			ParticleSubset* atomSubset = old_dw->getParticleSubset(materialIndex, patch);
			constParticleVariable<Point> atomPositions;
			constParticleVariable<long64> atomIDs;

			old_dw->get(atomPositions, d_label->global->pX, atomSubset);
			old_dw->get(atomIDs, d_label->global->pID, atomSubset);
//			old_dw->get(atomPositions, d_label->pXLabel, atomSubset);
//			old_dw->get(atomIDs, d_label->pParticleIDLabel, atomSubset);

			// Verify we have enough memory to hold the charge map for the current atom type
			currentSPMEPatch->verifyChargeMapAllocation(atomSubset->numParticles(),materialIndex);

			// Pull the location for the SPMEPatch's copy of the charge map for this material type
			std::vector<SPMEMapPoint>* gridMap = currentSPMEPatch->getChargeMap(materialIndex);

			// begin loop to generate the charge map
			for (ParticleSubset::iterator atom = atomSubset->begin(); atom != atomSubset->end(); ++atom) {
                particleIndex atomIndex = *atom;

				particleId ID = atomIDs[atomIndex];
				Point position = atomPositions[atomIndex];

				Vector atomGridCoordinates = position.asVector() * inverseUnitCell;
				// ^^^ Note:  We may want to replace a matrix/vector multiplication with optimized orthorhombic multiplications

				Vector kReal = d_kLimits.asVector();
				atomGridCoordinates *= kReal;
				IntVector atomGridOffset(atomGridCoordinates.asPoint());
				Vector splineValues = atomGridOffset.asVector() - atomGridCoordinates;

				size_t support = d_interpolatingSpline.getSupport();

				SCIRun::IntVector SupportVector = IntVector(support, support, support);
				std::vector<Vector> baseLevel(support), firstDerivative(support), secondDerivative(support);

				d_interpolatingSpline.evaluateThroughSecondDerivative(splineValues,baseLevel,firstDerivative,secondDerivative);
				SimpleGrid<double> chargeGrid(SupportVector, atomGridOffset, IV_ZERO, 0);
				SimpleGrid<Vector> forceGrid(SupportVector, atomGridOffset, IV_ZERO, 0);
				SimpleGrid<Matrix3> dipoleGrid(SupportVector, atomGridOffset, IV_ZERO, 0);

				for (size_t xIndex = 0; xIndex < support; ++xIndex) {
				  double   Sx = baseLevel[xIndex].x();
				  double  dSx = firstDerivative[xIndex].x()*kReal.x();
				  double d2Sx = secondDerivative[xIndex].x()*kReal.x()*kReal.x();
				  for (size_t yIndex = 0; yIndex < support; ++yIndex) {
					double   Sy = baseLevel[yIndex].y();
					double  dSy = firstDerivative[yIndex].y()*kReal.y();
					double d2Sy = secondDerivative[yIndex].y()*kReal.y()*kReal.y();
					double   Sx_Sy =  Sx*Sy;
					double  dSx_Sy = dSx*Sy;
					double  Sx_dSy = Sx*dSy;
					double dSx_dSy = dSx*dSy;
					for (size_t zIndex = 0; zIndex < support; ++zIndex) {
					  double   Sz = baseLevel[zIndex].z();
					  double  dSz = firstDerivative[zIndex].z()*kReal.z();
					  double d2Sz = secondDerivative[zIndex].z()*kReal.z()*kReal.z();
					  double   Sx_Sz =   Sx*Sz;
					  double   Sy_Sz =   Sy*Sz;
					  double dSx_dSy_Sz = dSx_dSy*Sz;
					  double dSx_Sy_dSz = dSx_Sy*dSz;
					  double Sx_dSy_dSz = Sx_dSy*dSz;
					  // Pure spline multiplication for charge/charge interactions
					  chargeGrid(xIndex,yIndex,zIndex) = Sx_Sy*Sz;
					  // dPhi/du for reciprocal contribution to electric field (also charge contribution to force)
					  forceGrid(xIndex,yIndex,zIndex)  = Vector(dSx*Sy_Sz,dSy*Sx_Sz,dSz*Sx_Sy);
					  // dE/du for reciprocal dipole contribution to the force
					  dipoleGrid(xIndex,yIndex,zIndex) = Matrix3(d2Sx*Sy_Sz,dSx_dSy_Sz,dSx_Sy_dSz,
					                                             dSx_dSy_Sz,d2Sy*Sx_Sz,Sx_dSy_dSz,
					                                             dSx_Sy_dSz,Sx_dSy_dSz,d2Sz*Sx_Sy);
					}
				  }
				}
				//SPMEMapPoint holds ONLY the spline and related derivatives; charge and dipole values and derivatives
				//  don't get baked in, so that we don't need to recalculate these quantities within the inner loop.
 			    SPMEMapPoint currentMapPoint(ID, atomGridOffset, chargeGrid, forceGrid, dipoleGrid);
			    gridMap->push_back(currentMapPoint);
			}
		}
	}
}

void SPME::calculatePreTransformDipole(const ProcessorGroup*    pg,
                                       const PatchSubset*       patches,
                                       const MaterialSubset*    materials,
                                       DataWarehouse*           old_dw,
                                       DataWarehouse*           new_dw,
                                       coordinateSystem*        coordSys,
                                       SimulationStateP&        simState)
{
	size_t numPatches   = patches->size();
	size_t numAtomTypes = materials->size();
	Uintah::Matrix3 inverseUnitCell = coordSys->getInverseCell();

	for (size_t currPatch = 0; currPatch < numPatches; ++currPatch) {
		const Patch* patch = patches->get(currPatch);

		// Extract current SPMEPatch
		SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;

		// SPMEPatches map to Patches 1:1; shouldn't need to lock for anything that's local to a patch
		SimpleGrid<dblcomplex>* Q_threadLocal = currentSPMEPatch->getQ();

		//Initialize once before mapping any type of atoms
		Q_threadLocal->initialize(dblcomplex(0.0,0.0));
		for (size_t currAtomType = 0; currAtomType < numAtomTypes; ++currAtomType) {
			int atomType = materials->get(currAtomType);
			ParticleSubset* pset = old_dw->getParticleSubset(atomType, patch);
			double atomCharge = simState->getMDMaterial(atomType)->getCharge();
//			double atomCharge = d_system->getAtomicCharge(atomType);
			constParticleVariable<Vector> p_Dipole;
			old_dw->get(p_Dipole, d_label->electrostatic->pMu, pset);
			std::vector<SPMEMapPoint>* gridMap = currentSPMEPatch->getChargeMap(atomType);
			SPME::dipoleMapChargeToGrid(currentSPMEPatch, gridMap, pset, atomCharge, p_Dipole, inverseUnitCell);
		} // end Atom Type Loop
	} // end Patch Loop

	// TODO keep an eye on this to make sure it works like we think it should
	if (Thread::self()->myid() == 0) {
	  d_Q_nodeLocal->initialize(dblcomplex(0.0, 0.0));
	}

	bool replace = true;
	new_dw->transferFrom(old_dw, d_label->global->pX, patches, materials, replace);
//	new_dw->transferFrom(old_dw, d_label->pXLabel, patches, materials, replace);

}

void SPME::dipoleMapChargeToGrid(SPMEPatch* spmePatch,
                                 const std::vector<SPMEMapPoint>* gridMap,
		                         ParticleSubset* pset,
                                 double charge,
		                         constParticleVariable<Vector>& p_Dipole,
		                         Uintah::Matrix3 inverseUnitCell) {

  // grab local Q grid
  SimpleGrid<dblcomplex>* Q_patchLocal 	= spmePatch->getQ();
  IntVector 				patchOffset = spmePatch->getGlobalOffset();
  //IntVector 				patchExtent = Q_patchLocal->getExtentWithGhost();

  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); ++iter) {
    particleIndex atom = *iter;

    const SimpleGrid<double> chargeMap = (*gridMap)[atom].getChargeGrid();
    const SimpleGrid<Vector> gradMap   = (*gridMap)[atom].getGradientGrid();
    const Vector             dipole    = p_Dipole[atom];

    IntVector QAnchor = chargeMap.getOffset();
    IntVector supportExtent = chargeMap.getExtents();
    IntVector Base = QAnchor - patchOffset;

    int x_Base = Base[0];
    int y_Base = Base[1];
    int z_Base = Base[2];

    int xExtent = supportExtent[0];
    int yExtent = supportExtent[1];
    int zExtent = supportExtent[2];

    Vector d_dot_nabla = dipole * inverseUnitCell;

    for (int xmask = 0; xmask < xExtent; ++xmask) {
      int x_anchor = x_Base + xmask;
      for (int ymask = 0; ymask < yExtent; ++ymask) {
        int y_anchor = y_Base + ymask;
        for (int zmask = 0; zmask < zExtent; ++zmask) {
          int z_anchor = z_Base + zmask;
          dblcomplex val = charge*chargeMap(xmask,ymask,zmask) + Dot(d_dot_nabla,gradMap(xmask,ymask,zmask));
          (*Q_patchLocal)(x_anchor, y_anchor, z_anchor) += val;
		}
      }
    }
  }
}

void SPME::dipoleUpdateFieldAndStress(const ProcessorGroup* pg,
                                      const PatchSubset* patches,
                                      const MaterialSubset* materials,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw,
                                      coordinateSystem* coordSys) {

  size_t numPatches = patches->size();
  size_t numAtomTypes = materials->size();

  SCIRun::Vector delU0 = coordSys->getInverseCell().getRow(0);
  SCIRun::Vector delU1 = coordSys->getInverseCell().getRow(1);
  SCIRun::Vector delU2 = coordSys->getInverseCell().getRow(2);

  // Step through all the patches on this thread
  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* patch = patches->get(patchIndex);
    SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;
    SimpleGrid<dblcomplex>* QConvoluted = currentSPMEPatch->getQ();
    IntVector patchOffset = currentSPMEPatch->getGlobalOffset();

    // step through the materials on this patch
    for (size_t currAtomType = 0; currAtomType < numAtomTypes; ++currAtomType) {
      int atomType = materials->get(currAtomType);
      std::vector<SPMEMapPoint>* gridMap = currentSPMEPatch->getChargeMap(atomType);
      ParticleSubset* pset = old_dw->getParticleSubset(atomType, patch);
      ParticleVariable<Vector> p_ReciprocalField;
      new_dw->getModifiable(p_ReciprocalField, d_label->electrostatic->pE_electroInverse, pset);
//      new_dw->getModifiable(p_ReciprocalField, d_label->pElectrostaticsReciprocalField, pset);
      for (ParticleSubset::iterator pIter = pset->begin(); pIter != pset->end(); ++pIter) {
        particleIndex atom = *pIter;
        p_ReciprocalField[atom] = SCIRun::Vector(0.0); // Zero out field
        const SimpleGrid<Vector> potentialDerivMap = (*gridMap)[atom].getPotentialDerivativeGrid();
        IntVector QAnchor = potentialDerivMap.getOffset();
        IntVector supportExtent = potentialDerivMap.getExtents();
        IntVector Base = QAnchor - patchOffset;
        int x_Base = Base[0];
        int y_Base = Base[1];
        int z_Base = Base[2];
// XXX TODO FIXME FINISH LOGIC!!!
        int xExtent = supportExtent[0];
        int yExtent = supportExtent[1];
        int zExtent = supportExtent[2];

      }
    }
  }
  // TODO fixme [APH]
}

void SPME::calculatePreTransform(const ProcessorGroup* pg,
                                 const PatchSubset* patches,
                                 const MaterialSubset* materials,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{

  size_t numPatches   = patches->size();
  size_t numAtomTypes = materials->size();

  for (size_t currPatch = 0; currPatch < numPatches; ++currPatch) {
      const Patch* patch = patches->get(currPatch);

      // Extract current SPMEPatch
      SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;

      // SPMEPatches map to Patches 1:1; shouldn't need to lock for anything that's local to a patch
      SimpleGrid<dblcomplex>* Q_threadLocal = currentSPMEPatch->getQ();

      //Initialize once before mapping any type of atoms
      Q_threadLocal->initialize(dblcomplex(0.0,0.0));
      for (size_t currAtomType = 0; currAtomType < numAtomTypes; ++currAtomType) {
          int atomType = materials->get(currAtomType);
          ParticleSubset* pset = old_dw->getParticleSubset(atomType, patch);
          double atomCharge = d_system->getAtomicCharge(atomType);
          std::vector<SPMEMapPoint>* gridMap = currentSPMEPatch->getChargeMap(atomType);
          SPME::mapChargeToGrid(currentSPMEPatch, gridMap, pset, atomCharge);
      } // end Atom Type Loop
  } // end Patch Loop

  // TODO keep an eye on this to make sure it works like we think it should
  if (Thread::self()->myid() == 0) {
    d_Q_nodeLocal->initialize(dblcomplex(0.0, 0.0));
  }

  bool replace = true;
  new_dw->transferFrom(old_dw, d_label->global->pX, patches, materials, replace);
//  new_dw->transferFrom(old_dw, d_label->pXLabel, patches, materials, replace);
}

void SPME::generateChargeMap(std::vector<SPMEMapPoint>* chargeMap,
                             ParticleSubset* pset,
                             constParticleVariable<Point>& particlePositions,
                             constParticleVariable<long64>& particleIDs)
{
  // TODO FIXME Rewrite to use in place memory.
  // Loop through particles
  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); ++iter) {
    particleIndex pidx = *iter;

    SCIRun::Point position = particlePositions[pidx];
    particleId pid = particleIDs[pidx];
    SCIRun::Vector particleGridCoordinates;

    //Calculate reduced coordinates of point to recast into charge grid
    particleGridCoordinates = (position.asVector()) * d_inverseUnitCell;
    // ** NOTE: JBH --> We may want to do this with a bit more thought eventually, since multiplying by the InverseUnitCell
    //                  is expensive if the system is orthorhombic, however it's not clear it's more expensive than dropping
    //                  to call MDSystem->isOrthorhombic() and then branching the if statement appropriately.

    SCIRun::Vector kReal = d_kLimits.asVector();
    particleGridCoordinates *= kReal;
    SCIRun::IntVector particleGridOffset(particleGridCoordinates.asPoint());
    SCIRun::Vector splineValues = particleGridOffset.asVector() - particleGridCoordinates;

    std::vector<double> xSplineArray = d_interpolatingSpline.evaluateGridAligned(splineValues.x());
    std::vector<double> ySplineArray = d_interpolatingSpline.evaluateGridAligned(splineValues.y());
    std::vector<double> zSplineArray = d_interpolatingSpline.evaluateGridAligned(splineValues.z());

    std::vector<double> xSplineDeriv = d_interpolatingSpline.derivativeGridAligned(splineValues.x());
    std::vector<double> ySplineDeriv = d_interpolatingSpline.derivativeGridAligned(splineValues.y());
    std::vector<double> zSplineDeriv = d_interpolatingSpline.derivativeGridAligned(splineValues.z());

    SCIRun::IntVector extents(xSplineArray.size(), ySplineArray.size(), zSplineArray.size());

    SimpleGrid<double> chargeGrid(extents, particleGridOffset, IV_ZERO, 0);
    SimpleGrid<SCIRun::Vector> forceGrid(extents, particleGridOffset, IV_ZERO, 0);
    SimpleGrid<Matrix3> sg_Matrix3Null(extents, particleGridOffset, IV_ZERO, 0);
    sg_Matrix3Null.fill(Matrix3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));

    size_t XExtent = xSplineArray.size();
    size_t YExtent = ySplineArray.size();
    size_t ZExtent = zSplineArray.size();

    for (size_t xidx = 0; xidx < XExtent; ++xidx) {
      double dampX = xSplineArray[xidx];
      for (size_t yidx = 0; yidx < YExtent; ++yidx) {
        double dampY = ySplineArray[yidx];
        double dampXY = dampX * dampY;
        for (size_t zidx = 0; zidx < ZExtent; ++zidx) {
          double dampZ = zSplineArray[zidx];
          double dampYZ = dampY * dampZ;
          double dampXZ = dampX * dampZ;

          chargeGrid(xidx, yidx, zidx) = dampXY * dampZ;
          forceGrid(xidx, yidx, zidx) = Vector(dampYZ * xSplineDeriv[xidx] * kReal.x(), dampXZ * ySplineDeriv[yidx] * kReal.y(),
                                               dampXY * zSplineDeriv[zidx] * kReal.z());

        }
      }
    }

    // TODO -> Look at building these in place in the chargeMap to save time.
    SPMEMapPoint currentMapPoint(pid, particleGridOffset, chargeGrid, forceGrid, sg_Matrix3Null);
    chargeMap->push_back(currentMapPoint);
  }
}

void SPME::mapChargeToGrid(SPMEPatch* spmePatch,
                           const std::vector<SPMEMapPoint>* gridMap,
                           ParticleSubset* pset,
                           double Charge) {
  // grab local Q grid
  SimpleGrid<dblcomplex>* Q_patchLocal = spmePatch->getQ();
  IntVector patchOffset = spmePatch->getGlobalOffset();
  IntVector patchExtent = Q_patchLocal->getExtentWithGhost();

  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); ++iter) {
    particleIndex pidx = *iter;

    const SimpleGrid<double> chargeMap = (*gridMap)[pidx].getChargeGrid();

    IntVector QAnchor = chargeMap.getOffset();         // Location of the 0,0,0 origin for the charge map grid
    IntVector supportExtent = chargeMap.getExtents();  // Extents of the charge map grid
    IntVector Base = QAnchor - patchOffset;

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
          dblcomplex val = Charge * chargeMap(xmask, ymask, zmask);
          (*Q_patchLocal)(x_anchor, y_anchor, z_anchor) += val;
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------
// Post transform calculation related routines

void SPME::calculatePostTransform(const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset* materials,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  SimulationStateP simState = d_system->getStatePointer();
  size_t numPatches = patches->size();
  size_t numLocalAtomTypes = materials->size();
  for (size_t p = 0; p < numPatches; ++p) {
    const Patch* patch = patches->get(p);
    SPMEPatch* currentSPMEPatch = d_spmePatchMap.find(patch->getID())->second;

    for (size_t localAtomTypeIndex = 0; localAtomTypeIndex < numLocalAtomTypes; ++localAtomTypeIndex) {
      int globalAtomType = materials->get(localAtomTypeIndex);
      double currentCharge = simState->getMDMaterial(globalAtomType)->getCharge();

      ParticleSubset* pset = old_dw->getParticleSubset(globalAtomType, patch);
      constParticleVariable<double> pcharge;
      ParticleVariable<Vector> pforcenew;
      new_dw->allocateAndPut(pforcenew, d_label->electrostatic->pF_electroInverse_preReloc, pset);
//      new_dw->allocateAndPut(pforcenew, d_label->pElectrostaticsReciprocalForce_preReloc, pset);
      std::vector<SPMEMapPoint>* gridMap = currentSPMEPatch->getChargeMap(globalAtomType);

      // Calculate electrostatic contribution to f_ij(r)
      SPME::mapForceFromGrid(currentSPMEPatch, gridMap, pset, currentCharge, pforcenew);
    }
  }
}

void SPME::mapForceFromGrid(SPMEPatch* spmePatch,
                            const std::vector<SPMEMapPoint>* gridMap,
                            ParticleSubset* pset,
                            double charge,
                            ParticleVariable<Vector>& pforcenew)
{
  SimpleGrid<std::complex<double> >* Q_patchLocal = spmePatch->getQ();
  IntVector patchOffset = spmePatch->getGlobalOffset();
  IntVector patchExtent = Q_patchLocal->getExtentWithGhost();

  for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); ++iter) {
    particleIndex pidx = *iter;

    SimpleGrid<SCIRun::Vector> forceMap = (*gridMap)[pidx].getForceGrid();

    SCIRun::Vector newForce = Vector(0, 0, 0);
    IntVector QAnchor = forceMap.getOffset();         // Location of the 0,0,0 origin for the force map grid
    IntVector supportExtent = forceMap.getExtents();  // Extents of the force map grid
    IntVector Base = QAnchor - patchOffset;

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

          //--------------------------< DEBUG >--------------------------------
          if (spme_dbg.active()) {
            if (x_anchor > patchExtent.x()) {
              std::cerr << " Error:  x_anchor exceeds patch Extent in mapForceFromGrid"
                        << " xBase: " << x_Base << " xMask: " << xmask << " xAnchor: " << x_anchor
                        << " xPatchExtent: " << patchExtent.x();
            }
            if (y_anchor > patchExtent.y()) {
              std::cerr << " Error:  y_anchor exceeds patch Extent in mapForceFromGrid"
                        << " yBase: " << y_Base << " yMask: " << ymask << " yAnchor: " << y_anchor
                        << " yPatchExtent: " << patchExtent.y();
            }
            if (z_anchor > patchExtent.z()) {
              std::cerr << " Error:  z_anchor exceeds patch Extent in mapForceFromGrid"
                        << " zBase: " << z_Base << " zMask: " << zmask << " zAnchor: " << z_anchor
                        << " zPatchExtent: " << patchExtent.z();
            }
          }
          //--------------------------< DEBUG >--------------------------------

          // Local grid should have appropriate ghost cells, so no wrapping necessary.
          double QReal = std::real((*Q_patchLocal)(x_anchor, y_anchor, z_anchor));
          newForce += forceMap(xmask, ymask, zmask) * QReal * charge * d_inverseUnitCell;
        }
      }
    }
    // sanity check
    if (spme_dbg.active()) {
      if (pidx < 5) {
        cerrLock.lock();
        std::cerr << " Force Check (" << pidx << "): " << newForce << std::endl;
        pforcenew[pidx] = newForce;
        cerrLock.unlock();
      }
    }
  }
}




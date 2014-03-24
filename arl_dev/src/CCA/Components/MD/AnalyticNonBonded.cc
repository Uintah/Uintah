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

#include <CCA/Components/MD/AnalyticNonBonded.h>
#include <CCA/Components/MD/Forcefields/Forcefield.h>
#include <CCA/Components/MD/MDSystem.h>
#include <CCA/Components/MD/MDLabel.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Box.h>
#include <Core/Thread/Thread.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <iomanip>

#ifdef DEBUG
#include <Core/Util/FancyAssert.h>
#endif

using namespace Uintah;

extern SCIRun::Mutex cerrLock;
extern SCIRun::Mutex coutLock;

static DebugStream analytic_dbg("AnalyticNonbondedDbg", false);
static DebugStream analytic_cout("AnalyticNonbondedCout", false);

AnalyticNonBonded::AnalyticNonBonded()
{
	
}

AnalyticNonBonded::~AnalyticNonBonded()
{
	
}

AnalyticNonBonded::AnalyticNonBonded(MDSystem* system,
		const double r12,
		const double r6,
		const double cutoffRadius)
		:
				d_system(system), d_r12(r12), d_r6(r6), d_nonbondedRadius(cutoffRadius)
{
	d_nonBondedInteractionType = NonBonded::LJ12_6;
}

//-----------------------------------------------------------------------------
// Interface implementations
void AnalyticNonBonded::initialize(const ProcessorGroup* pg,
		const PatchSubset* patches,
		const MaterialSubset* materials,
		DataWarehouse* old_dw,
		DataWarehouse* new_dw)
		{
	// global sum reduction of "vdwEnergy" - van der Waals potential energy
	new_dw->put(sum_vartype(0.0), d_lb->nonbondedEnergyLabel);
	
	SoleVariable<double> dependency;
	new_dw->put(dependency, d_lb->nonbondedDependencyLabel);
}

void AnalyticNonBonded::setup(const ProcessorGroup* pg,
		const PatchSubset* patches,
		const MaterialSubset* materials,
		DataWarehouse* old_dw,
		DataWarehouse* new_dw)
		{
	
}

void AnalyticNonBonded::newCalculate(const ProcessorGroup* pg,
                                     const PatchSubset* patches,
                                     const MaterialSubset* materials,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw,
                                     SchedulerP& subscheduler,
                                     const LevelP& level) {
  Vector box = d_system->getBox();
  double cutoff2 = d_nonbondedRadius * d_nonbondedRadius;
  double nbEnergy = 0;
  Matrix3 stressTensor(0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0);

  int CUTOFF_CELLS = d_system->getNonbondedGhostCells();

  // TODO fixme [APH]
//  Forcefield* d_system->getForcefieldReference();

  size_t numPatches = patches->size();
  size_t numMaterials = materials->size();

  for (size_t patchIdx = 0; patchIdx < numPatches; ++patchIdx) {
    const Patch* currPatch = patches->get(patchIdx);
    for (size_t localMatIdx = 0; localMatIdx < numMaterials; ++localMatIdx) {
      int localMaterial = materials->get(localMatIdx);
      ParticleSubset* localAtoms = old_dw->getParticleSubset(localMaterial, currPatch);
      constParticleVariable<long64> localID;
      old_dw->get(localID, d_lb->pParticleIDLabel, localAtoms);
      constParticleVariable<Point> localPositions;
      old_dw->get(localPositions, d_lb->pXLabel, localAtoms);
      ParticleVariable<Vector> pForce;
      new_dw->allocateAndPut(pForce, d_lb->pNonbondedForceLabel_preReloc, localAtoms);
      for (size_t neighborMatIdx = 0; neighborMatIdx < numMaterials; ++neighborMatIdx) {
        int neighborMaterial = materials->get(neighborMatIdx);
        ParticleSubset* neighborAtoms = old_dw->getParticleSubset(neighborMaterial, currPatch, Ghost::AroundNodes, CUTOFF_CELLS, d_lb->pXLabel);
        constParticleVariable<long64> neighborID;
        old_dw->get(neighborID, d_lb->pParticleIDLabel, neighborAtoms);
        constParticleVariable<Point> neighborPositions;
        old_dw->get(neighborPositions, d_lb->pXLabel, neighborAtoms);
        // TODO fixme [APH]
//        NonbondedTwoBodyPotential* currentPotential = Forcefield->getNonbondedPotential(localMaterial, neighborMaterial);
        size_t localAtomCount = localAtoms->numParticles();
        size_t neighborAtomCount = neighborAtoms->numParticles();
        for (size_t localAtomIdx=0; localAtomIdx < localAtomCount; ++localAtomIdx ) {
          for (size_t neighborAtomIdx=0; neighborAtomIdx < neighborAtomCount; ++neighborAtomIdx) {
            if (localID[localAtomIdx] != neighborID[neighborAtomIdx]) { // Make sure we're not seeing the same atom
              SCIRun::Vector atomicDistanceVector = neighborPositions[neighborAtomIdx].asVector() - localPositions[localAtomIdx].asVector();
              if (atomicDistanceVector.length2() <= cutoff2) {
                double tempEnergy;
                SCIRun::Vector tempForce;
              // TODO fixme [APH]
//                currentPotential->fillEnergyAndForce(tempForce, tempEnergy, &atomicDistanceVector);
                nbEnergy += tempEnergy;
                pForce[localAtomIdx] += tempForce;
                stressTensor += OuterProduct(atomicDistanceVector,tempForce);
              }  // atomicDistanceVector.length2() <= cutoff2
            }  // localID[localAtomIdx] != neighborID[neighborAtomIdx]
          }  // loop over neighborAtomIdx
        }  // loop over localAtomIdx
      }  // loop over neighbor materials
    }  // loop over local materials
  }  // loop over patches
  new_dw->put(sum_vartype(0.5 * nbEnergy), d_lb->nonbondedEnergyLabel);
  new_dw->put(matrix_sum(0.5 * stressTensor), d_lb->nonbondedStressLabel);
  return;
}

void AnalyticNonBonded::calculate(const ProcessorGroup* pg,
		const PatchSubset* patches,
		const MaterialSubset* materials,
		DataWarehouse* old_dw,
		DataWarehouse* new_dw,
		SchedulerP& subscheduler,
		const LevelP& level)
		{
	Vector box = d_system->getBox();
	double cut_sq = d_nonbondedRadius * d_nonbondedRadius;
	double vdwEnergy = 0;
	int CUTOFF_RADIUS = d_system->getNonbondedGhostCells();
	
	// loop through all patches
	size_t numPatches = patches->size();
	size_t numMatls = materials->size();
	for (size_t p = 0; p < numPatches; ++p) {
		const Patch* patch = patches->get(p);
		for (size_t m = 0; m < numMatls; ++m) {
			int matl = materials->get(m);
			
			// get particles within bounds of current patch (interior, no ghost cells)
			ParticleSubset* local_pset = old_dw->getParticleSubset(matl, patch);
			
			// get particles within bounds of cutoff radius
			ParticleSubset* neighbor_pset = old_dw->getParticleSubset(matl, patch, Ghost::AroundNodes, CUTOFF_RADIUS, d_lb->pXLabel);
			
			// requires variables
			constParticleVariable<Point> px_local;
			constParticleVariable<Point> px_neighbors;
			constParticleVariable<Vector> pforce;
			constParticleVariable<double> penergy;
			constParticleVariable<long64> pid_local;
			constParticleVariable<long64> pid_neighbor;
			old_dw->get(px_local, d_lb->pXLabel, local_pset);
			old_dw->get(px_neighbors, d_lb->pXLabel, neighbor_pset);
			old_dw->get(penergy, d_lb->pEnergyLabel, local_pset);
			old_dw->get(pforce, d_lb->pNonbondedForceLabel, local_pset);
			old_dw->get(pid_local, d_lb->pParticleIDLabel, local_pset);
			old_dw->get(pid_neighbor, d_lb->pParticleIDLabel, neighbor_pset);
			
			// computes variables
			ParticleVariable<Vector> pforcenew;
			ParticleVariable<double> penergynew;
			new_dw->allocateAndPut(pforcenew, d_lb->pNonbondedForceLabel_preReloc, local_pset);
			new_dw->allocateAndPut(penergynew, d_lb->pEnergyLabel_preReloc, local_pset);
			
			// loop over all atoms in system, calculate the forces
			double r2, ir2, ir6, ir12, T6, T12;
			double forceTerm;
			Vector totalForce, atomForce;
			Vector atomicDistanceVector;
			
			size_t localAtoms = local_pset->numParticles();
			size_t neighborAtoms = neighbor_pset->numParticles();
			
			// loop over all local atoms
			for (size_t i = 0; i < localAtoms; ++i) {
				atomForce = Vector(0.0, 0.0, 0.0);
				
				// loop over the neighbors of atom "i"
				for (size_t j = 0; j < neighborAtoms; ++j) {
					
					// Ai != Aj
					if (pid_local[i] != pid_neighbor[j]) {
						
						// the vector distance between atom i and j
						atomicDistanceVector = px_local[i] - px_neighbors[j];
						
						// this is required for periodic boundary conditions
						atomicDistanceVector -= (atomicDistanceVector / box).vec_rint() * box;
						
						r2 = atomicDistanceVector.length2();
						
						// only add neighbor atoms within spherical cut-off around atom "i"
						if (r2 < cut_sq) {
							
							ir2 = 1.0 / r2;         // 1/r^2
							ir6 = ir2 * ir2 * ir2;  // 1/r^6
							ir12 = ir6 * ir6;       // 1/r^12
							T12 = d_r12 * ir12;
							T6 = d_r6 * ir6;
							penergynew[i] = T12 - T6;  // energy
							vdwEnergy += penergynew[i];  // count the energy
							forceTerm = (12.0 * T12 - 6.0 * T6) * ir2;  // the force term
							totalForce = forceTerm * atomicDistanceVector;
							
							// the contribution of force on atom i
							atomForce += totalForce;
							
						}  // end if (r2 < cut_sq)
					}  // end Ai != Aj
				}  // end neighbor loop for atom "j"
				
				// sum up contributions to force for atom i
				pforcenew[i] += atomForce;
				
			}  // end atom loop
			
			// this accounts for double energy from Aij <--> Aji
			vdwEnergy *= 0.50;
			
			if (analytic_dbg.active()) {
				cerrLock.lock();
				Vector forces(0.0, 0.0, 0.0);
				unsigned int numParticles = local_pset->numParticles();
				for (unsigned int i = 0; i < numParticles; ++i) {
					forces += pforcenew[i];
				}
				std::cout.setf(std::ios_base::scientific);
				std::cout << "Total Local Energy: " << std::setprecision(16) << vdwEnergy << std::endl;
				std::cout << "Local Force: [";
				std::cout << std::setw(16) << std::setprecision(8) << forces.x();
				std::cout << std::setw(16) << std::setprecision(8) << forces.y();
				std::cout << std::setprecision(8) << forces.z() << std::setw(4) << "]";
				std::cout << std::endl;
				std::cout.unsetf(std::ios_base::scientific);
				cerrLock.unlock();
			}
			
		}  // end materials loop
		
		coutLock.lock();
		std::cout.setf(std::ios_base::left);
		std::cout << std::setw(30) << Thread::self()->getThreadName();
		std::cout << "Uintah thread ID: " << Thread::self()->myid()
				<< "  Thread group: " << Thread::self()->getThreadGroup()
				<< "  Patch: " << patch->getID()
				<< "  VDW-Energy: " << vdwEnergy << std::endl;
		coutLock.unlock();
		
	}  // end patch loop
	
	// global reduction on vdwEnergy
	new_dw->put(sum_vartype(vdwEnergy), d_lb->nonbondedEnergyLabel);
	
}

void AnalyticNonBonded::finalize(const ProcessorGroup* pg,
		const PatchSubset* patches,
		const MaterialSubset* materials,
		DataWarehouse* old_dw,
		DataWarehouse* new_dw)
		{
// for now, do nothing
}

void AnalyticNonBonded::generateNeighborList(ParticleSubset* local_pset,
		ParticleSubset* neighbor_pset,
		constParticleVariable<Point>& px_local,
		constParticleVariable<Point>& px_neighbors,
		std::vector<std::vector<int> >& neighbors)
		{
	unsigned int localAtoms = local_pset->numParticles();
	unsigned int neighborAtoms = neighbor_pset->numParticles();
	
	double r2;
	Vector box = d_system->getBox();
	Vector reducedCoordinates;
	double cut_sq = d_nonbondedRadius * d_nonbondedRadius;
	
	for (size_t i = 0; i < localAtoms; ++i) {
		for (size_t j = 0; j < neighborAtoms; ++j) {
			if (i != j) {
				// the vector distance between atom i and j
				reducedCoordinates = px_local[i] - px_neighbors[j];
				
				// this is required for periodic boundary conditions
				reducedCoordinates -= (reducedCoordinates / box).vec_rint() * box;
				
				// eliminate atoms outside of cutoff radius, add those within as neighbors
				if ((fabs(reducedCoordinates[0]) < d_nonbondedRadius) && (fabs(reducedCoordinates[1]) < d_nonbondedRadius)
					&& (fabs(reducedCoordinates[2]) < d_nonbondedRadius)) {
					
					double reducedX = reducedCoordinates[0] * reducedCoordinates[0];
					double reducedY = reducedCoordinates[1] * reducedCoordinates[1];
					double reducedZ = reducedCoordinates[2] * reducedCoordinates[2];
					r2 = sqrt(reducedX + reducedY + reducedZ);
					// only add neighbor atoms within spherical cut-off around atom "i"
					if (r2 < cut_sq) {
						neighbors[i].push_back(j);
					}
				}
			}
		}
	}
}


/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Components/MPM/SerialMPM.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Components/MPM/Contact/ContactFactory.h>
#include <Uintah/Components/MPM/Fracture/FractureFactory.h>
#include <Uintah/Components/MPM/Fracture/Fracture.h>
#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Grid/NodeIterator.h> // Must be included after Region.h
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SoleVariable.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Exceptions/ParameterNotFound.h>

#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/MinMax.h>

#include <iostream>
#include <fstream>

#include "GeometrySpecification/Problem.h"

using namespace Uintah;
using namespace Uintah::MPM;

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Math::Min;
using SCICore::Math::Max;
using namespace std;


SerialMPM::SerialMPM( int MpiRank, int MpiProcesses ) :
  UintahParallelComponent( MpiRank, MpiProcesses )
{
   pDeformationMeasureLabel = new VarLabel("p.deformationMeasure",
			    ParticleVariable<Matrix3>::getTypeDescription());

   pStressLabel = new VarLabel( "p.stress",
			     ParticleVariable<Matrix3>::getTypeDescription() );

   pVolumeLabel = new VarLabel( "p.volume",
			     ParticleVariable<double>::getTypeDescription());

   pMassLabel = new VarLabel( "p.mass",
			ParticleVariable<double>::getTypeDescription() );

   pVelocityLabel = new VarLabel( "p.velocity", 
			     ParticleVariable<Vector>::getTypeDescription() );

   pExternalForceLabel = new VarLabel( "p.externalforce",
			     ParticleVariable<Vector>::getTypeDescription() );

   pXLabel = new VarLabel( "p.x", ParticleVariable<Point>::getTypeDescription(),
			     VarLabel::PositionVariable);

   //H.Tan:
   //  pSurfaceNormalLabel is used to define the surface normal of a boundary particle.
   //  For the interior particle, the p.surfaceNormal vector is set to (0,0,0)
   //  in this way we can distinguish boundary particles to interior particles
   //
   pSurfaceNormalLabel = new VarLabel( "p.surfaceNormal",
			     ParticleVariable<Vector>::getTypeDescription() );

   gAccelerationLabel = new VarLabel( "g.acceleration",
			      NCVariable<Vector>::getTypeDescription() );

   gMomExedAccelerationLabel = new VarLabel( "g.momexedacceleration",
			      NCVariable<Vector>::getTypeDescription() );

   gMassLabel = new VarLabel( "g.mass",
			      NCVariable<double>::getTypeDescription() );

   gVelocityLabel = new VarLabel( "g.velocity",
				  NCVariable<Vector>::getTypeDescription() );

   gMomExedVelocityLabel = new VarLabel( "g.momexedvelocity",
				NCVariable<Vector>::getTypeDescription() );

   gExternalForceLabel = new VarLabel( "g.externalforce",
			      NCVariable<Vector>::getTypeDescription() );

   gInternalForceLabel = new VarLabel( "g.internalforce",
			      NCVariable<Vector>::getTypeDescription() );

   gVelocityStarLabel = new VarLabel( "g.velocity_star",
			      NCVariable<Vector>::getTypeDescription() );

   gMomExedVelocityStarLabel = new VarLabel( "g.momexedvelocity_star",
			      NCVariable<Vector>::getTypeDescription() );

   cSelfContactLabel = new VarLabel( "c.selfContact",
			      CCVariable<bool>::getTypeDescription() );

   cSurfaceNormalLabel = new VarLabel( "c.surfaceNormalLabel",
			      CCVariable<Vector>::getTypeDescription() );


   // I'm not sure about this one:
   deltLabel = 
     new VarLabel( "delt", delt_vartype::getTypeDescription() );

}

SerialMPM::~SerialMPM()
{
}

void SerialMPM::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
			     SimulationStateP& sharedState)
{
   d_sharedState = sharedState;
   Problem prob_description;
   prob_description.preProcessor(prob_spec, grid, d_sharedState);

   cerr << "Number of velocity fields = " << d_sharedState->getNumVelFields()
	<< std::endl;

   d_contactModel = ContactFactory::create(prob_spec,sharedState);
   if (!d_contactModel)
     throw ParameterNotFound("No contact model");

   d_fractureModel = FractureFactory::create(prob_spec,sharedState);

   cerr << "SerialMPM::problemSetup not done\n";
}

void SerialMPM::scheduleInitialize(const LevelP& level,
				   SchedulerP& sched,
				   DataWarehouseP& dw)
{
   Level::const_regionIterator iter;

   for(iter=level->regionsBegin(); iter != level->regionsEnd(); iter++){

      const Region* region=*iter;
      {
	 Task* t = new Task("SerialMPM::initialize", region, dw, dw,
			    this, &SerialMPM::actuallyInitialize);
	 t->computes(dw, d_sharedState->get_delt_label());
	 sched->addTask(t);
      }
   }
}

void SerialMPM::scheduleComputeStableTimestep(const LevelP&,
					      SchedulerP&,
					      DataWarehouseP&)
{
   // Nothing to do here - delt is computed as a by-product of the
   // consitutive model
}

void SerialMPM::scheduleTimeAdvance(double /*t*/, double /*dt*/,
				    const LevelP&         level,
				          SchedulerP&     sched,
				          DataWarehouseP& old_dw, 
				          DataWarehouseP& new_dw)
{
   int numMatls = d_sharedState->getNumMatls();
   for(Level::const_regionIterator iter=level->regionsBegin();
       iter != level->regionsEnd(); iter++){

      const Region* region=*iter;

      if(d_fractureModel) {
	 /*
	  * labelSelfContactNodesAndCells
	  *   in(C.SURFACENORMAL,P.SURFACENORMAL)
	  *   operation(label the cell which has self-contact)
	  *   out(C.SELFCONTACTLABEL)
	  */
	 Task* t = new Task("Fracture::labelSelfContactCells",
			    region, old_dw, new_dw,
			    d_fractureModel,
			    &Fracture::labelSelfContactNodesAndCells);

	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( old_dw, cSurfaceNormalLabel, idx, region,
			 Ghost::None);
	    t->requires( old_dw, pSurfaceNormalLabel, idx, region,
			 Ghost::None);

	    t->computes( new_dw, cSelfContactLabel, idx, region );
	 }

	 sched->addTask(t);
      }

      {
	 /*
	  * interpolateParticlesToGrid
	  *   in(P.MASS, P.VELOCITY, P.NAT_X)
	  *   operation(interpolate the P.MASS and P.VEL to the grid
	  *             using P.NAT_X and some shape function evaluations)
	  * out(G.MASS, G.VELOCITY)
	  */
	 Task* t = new Task("SerialMPM::interpolateParticlesToGrid",
			    region, old_dw, new_dw,
			    this,&SerialMPM::interpolateParticlesToGrid);
	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires(old_dw, pMassLabel, idx, region,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, pVelocityLabel, idx, region,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, pExternalForceLabel, idx, region,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, pXLabel, idx, region,
			Ghost::AroundNodes, 1 );

	    t->computes(new_dw, gMassLabel, idx, region );
	    t->computes(new_dw, gVelocityLabel, idx, region );
	    t->computes(new_dw, gExternalForceLabel, idx, region );
	 }
		     
	 sched->addTask(t);
      }
      
      {
	 /* exMomInterpolated
	  *   in(G.MASS, G.VELOCITY)
	  *   operation(peform operations which will cause each of
	  *		  velocity fields to feel the influence of the
	  *		  the others according to specific rules)
	  *   out(G.VELOCITY)
	  */

	 Task* t = new Task("Contact::exMomInterpolated",
			    region, old_dw, new_dw,
			    d_contactModel,
			    &Contact::exMomInterpolated);
	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( new_dw, gMassLabel, idx, region,
			 Ghost::None);
	    t->requires( new_dw, gVelocityLabel, idx, region,
			 Ghost::None);

	    t->computes( new_dw, gMomExedVelocityLabel, idx, region );
	 }

	 sched->addTask(t);
      }
      
      {
	 /*
	  * computeStressTensor
	  *   in(G.VELOCITY, P.X, P.DEFORMATIONMEASURE)
	  *   operation(evaluate the gradient of G.VELOCITY at P.X, feed
	  *             this into a constitutive model, which will
	  *	           evaluate the stress and store it in the
	  *             DataWarehouse.  Each CM also computes the maximum
	  *             elastic wave speed for that material in the
	  *             region.  This is used in calculating delt.)
	  * out(P.DEFORMATIONMEASURE,P.STRESS)
	  */
	 Task* t = new Task("SerialMPM::computeStressTensor",
			    region, old_dw, new_dw,
			    this, &SerialMPM::computeStressTensor);
	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
	    if(mpm_matl){
	       ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
	       cm->addComputesAndRequires(t, mpm_matl, region, old_dw, new_dw);
	    }
	 }
	 sched->addTask(t);
      }

      if(d_fractureModel) {
	 /*
	  * updateSurfaceNormalOfBoundaryParticle
	  *   in(P.DEFORMATIONMEASURE)
	  *   operation(update the surface normal of each boundary particles)
	  * out(P.SURFACENORMAL)
	  */
	 Task* t = new Task("Fracture::updateSurfaceNormalOfBoundaryParticle",
			    region, old_dw, new_dw,
			    d_fractureModel,
			    &Fracture::updateSurfaceNormalOfBoundaryParticle);

	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( old_dw, pDeformationMeasureLabel, idx, region,
			 Ghost::None);
	    t->requires( old_dw, pSurfaceNormalLabel, idx, region,
			 Ghost::None);

	    t->computes( new_dw, pSurfaceNormalLabel, idx, region );
	 }

	 sched->addTask(t);
      }
      
      {
	 /*
	  * computeInternalForce
	  *   in(P.CONMOD, P.NAT_X, P.VOLUME)
	  *   operation(evaluate the divergence of the stress (stored in
	  *	       P.CONMOD) using P.NAT_X and the gradients of the
	  *             shape functions)
	  * out(G.F_INTERNAL)
	  */
	 Task* t = new Task("SerialMPM::computeInternalForce",
			    region, old_dw, new_dw,
			    this, &SerialMPM::computeInternalForce);
	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( new_dw, pStressLabel, idx, region,
			 Ghost::AroundNodes, 1);
	    t->requires( old_dw, pVolumeLabel, idx, region,
			 Ghost::AroundNodes, 1);

	    t->computes( new_dw, gInternalForceLabel, idx, region );
	 }

	 sched->addTask( t );
      }
      
      {
	 /*
	  * solveEquationsMotion
	  *   in(G.MASS, G.F_INTERNAL)
	  *   operation(acceleration = f/m)
	  *   out(G.ACCELERATION)
	  * 
	  */
	 Task* t = new Task("SerialMPM::solveEquationsMotion",
			    region, old_dw, new_dw,
			    this, &SerialMPM::solveEquationsMotion);
	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( new_dw, gMassLabel, idx, region,
			 Ghost::None);
	    t->requires( new_dw, gInternalForceLabel, idx, region,
			 Ghost::None);

	    t->computes( new_dw, gAccelerationLabel, idx, region);
	 }

	 sched->addTask(t);
      }
      
      {
	 /*
	  * integrateAcceleration
	  *   in(G.ACCELERATION, G.VELOCITY)
	  *   operation(v* = v + a*dt)
	  *   out(G.VELOCITY_STAR)
	  * 
	  */
	 Task* t = new Task("SerialMPM::integrateAcceleration",
			    region, old_dw, new_dw,
			    this, &SerialMPM::integrateAcceleration);
	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires(new_dw, gAccelerationLabel, idx, region,
			Ghost::None);
	    t->requires(new_dw, gMomExedVelocityLabel, idx, region,
			Ghost::None);
	    t->requires(old_dw, deltLabel );
		     
	    t->computes(new_dw, gVelocityStarLabel, idx, region );
	 }
		     
	 sched->addTask(t);
      }
      
      {
	 /* exMomIntegrated
	  *   in(G.MASS, G.VELOCITY_STAR, G.ACCELERATION)
	  *   operation(peform operations which will cause each of
	  *		  velocity fields to feel the influence of the
	  *		  the others according to specific rules)
	  *   out(G.VELOCITY_STAR, G.ACCELERATION)
	  */

	Task* t = new Task("Contact::exMomIntegrated",
			   region, old_dw, new_dw,
			   d_contactModel, &Contact::exMomIntegrated);
	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires(new_dw, gMassLabel, idx, region,
			Ghost::None);
	    t->requires(new_dw, gVelocityStarLabel, idx, region,
			Ghost::None);
	    t->requires(new_dw, gAccelerationLabel, idx, region,
			Ghost::None);

	    t->computes(new_dw, gMomExedVelocityStarLabel, idx, region);
	    t->computes(new_dw, gMomExedAccelerationLabel, idx, region);
	}

	sched->addTask(t);
      }

      if(d_fractureModel) {
	 /*
	  * updateNodeInformationInContactCells
	  *   in(C.SELFCONTACT,G.VELOCITY,G.ACCELERATION)
	  *   operation(update the node information including velocities and
	  *   accelerations in nodes of contact-cells)
	  *   out(G.VELOCITY,G.ACCELERATION)
	  */
	 Task* t = new Task("Fracture::updateSurfaceNormalOfBoundaryParticle",
			    region, old_dw, new_dw,
			    d_fractureModel,
			    &Fracture::updateSurfaceNormalOfBoundaryParticle);

	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( old_dw, cSelfContactLabel, idx, region,
			 Ghost::None);
	    t->requires( old_dw, gVelocityLabel, idx, region,
			 Ghost::None);
	    t->requires( old_dw, gAccelerationLabel, idx, region,
			 Ghost::None);

	    t->computes( new_dw, gVelocityLabel, idx, region );
	    t->computes( new_dw, gAccelerationLabel, idx, region );
	 }

	 sched->addTask(t);
      }
      
      {
	 /*
	  * interpolateToParticlesAndUpdate
	  *   in(G.ACCELERATION, G.VELOCITY_STAR, P.NAT_X)
	  *   operation(interpolate acceleration and v* to particles and
	  *             integrate these to get new particle velocity and
	  *             position)
	  * out(P.VELOCITY, P.X, P.NAT_X)
	  */
	 Task* t = new Task("SerialMPM::interpolateToParticlesAndUpdate",
			    region, old_dw, new_dw,
			    this, &SerialMPM::interpolateToParticlesAndUpdate);
	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires(new_dw, gMomExedAccelerationLabel, idx, region,
			Ghost::AroundCells, 1);
	    t->requires(new_dw, gMomExedVelocityStarLabel, idx, region,
			Ghost::AroundCells, 1);
	    t->requires(old_dw, pXLabel, idx, region,
			Ghost::None);
	    t->requires(old_dw, deltLabel );
	    t->computes(new_dw, pVelocityLabel, idx, region );
	    t->computes(new_dw, pXLabel, idx, region );
	 }

	 sched->addTask(t);
      }

      if(d_fractureModel) {
	 /*
	  * updateParticleInformationInContactCells
	  *   in(P.DEFORMATIONMEASURE)
	  *   operation(update the surface normal of each boundary particles)
	  * out(P.SURFACENORMAL)
	  */
	 Task* t = new Task("Fracture::updateSurfaceNormalOfBoundaryParticle",
			    region, old_dw, new_dw,
			    d_fractureModel,
			    &Fracture::updateSurfaceNormalOfBoundaryParticle);

	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( old_dw, pXLabel, idx, region,
			 Ghost::None);
	    t->requires( old_dw, pVelocityLabel, idx, region,
			 Ghost::None);
	    t->requires( old_dw, pExternalForceLabel, idx, region,
			 Ghost::None);
	    t->requires( old_dw, pDeformationMeasureLabel, idx, region,
			 Ghost::None);
	    t->requires( old_dw, pStressLabel, idx, region,
			 Ghost::None);

	    t->computes( new_dw, pXLabel, idx, region );
	    t->computes( new_dw, pVelocityLabel, idx, region );
	    t->computes( new_dw, pDeformationMeasureLabel, idx, region );
	    t->computes( new_dw, pStressLabel, idx, region );
	 }

	 sched->addTask(t);
      }

      if(d_fractureModel) {
	 /*
	  * crackGrow
	  *   in(P.STRESS)
	  *   operation(check the stress on each boudary particle to see
	  *             if the microcrack will grow.  If fracture occur,
	  *             more interior particles become boundary particles)
	  * out(P.SURFACENORMAL)
	  */
	 Task* t = new Task("Fracture::crackGrow",
			    region, old_dw, new_dw,
			    d_fractureModel,
			    &Fracture::crackGrow);

	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( new_dw, pStressLabel, idx, region,
			 Ghost::None);
	    t->requires( new_dw, pXLabel, idx, region,
			 Ghost::None);
	    t->requires( new_dw, pSurfaceNormalLabel, idx, region,
			 Ghost::None);

	    t->computes( new_dw, pSurfaceNormalLabel, idx, region );
	 }

	 sched->addTask(t);
      }
    }
}

void SerialMPM::actuallyInitialize(const ProcessorContext*,
				   const Region* region,
				   DataWarehouseP& /* old_dw */,
				   DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMatls();
  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    cerr << "numMatls = " << numMatls << '\n';
    if(mpm_matl){
       cerr << "dwindex=" << matl->getDWIndex() << '\n';
       particleIndex numParticles = mpm_matl->countParticles(region);

       mpm_matl->createParticles(numParticles, region, new_dw);
       mpm_matl->getConstitutiveModel()->initializeCMData(region,
						mpm_matl, new_dw);
       int vfindex = matl->getVFIndex();
       d_contactModel->initializeContact(region,vfindex,new_dw);
       
       if(d_fractureModel)
       d_fractureModel->initializeFracture(region,new_dw);
    }
  }
}


void SerialMPM::actuallyComputeStableTimestep(const ProcessorContext*,
					      const Region*,
					      DataWarehouseP&,
					      DataWarehouseP&)
{
}

void SerialMPM::interpolateParticlesToGrid(const ProcessorContext*,
					   const Region* region,
					   DataWarehouseP& old_dw,
					   DataWarehouseP& new_dw)
{
  // This needs the datawarehouse to allow indexing by material
  // for the particle data and velocity field by the grid data

  // Dd: making this compile... don't know if this correct...
  int numMatls = d_sharedState->getNumMatls();

  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int matlindex = matl->getDWIndex();
      int vfindex = matl->getVFIndex();
      // Create arrays for the particle data
      ParticleVariable<Point> px;
      ParticleVariable<double> pmass;
      ParticleVariable<Vector> pvelocity;
      ParticleVariable<Vector> pexternalforce;

      old_dw->get(px,             pXLabel, matlindex, region,
		  Ghost::AroundNodes, 1);
      old_dw->get(pmass,          pMassLabel, matlindex, region,
		  Ghost::AroundNodes, 1);
      old_dw->get(pvelocity,      pVelocityLabel, matlindex, region,
		  Ghost::AroundNodes, 1);
      old_dw->get(pexternalforce, pExternalForceLabel, matlindex, region,
		  Ghost::AroundNodes, 1);

      // Create arrays for the grid data
      NCVariable<double> gmass;
      NCVariable<Vector> gvelocity;
      NCVariable<Vector> externalforce;

      std::cerr << "allocating grid variables" << std::endl;
      new_dw->allocate(gmass,         gMassLabel, vfindex, region);
      new_dw->allocate(gvelocity,     gVelocityLabel, vfindex, region);
      new_dw->allocate(externalforce, gExternalForceLabel, vfindex, region);

      ParticleSubset* pset = px.getParticleSubset();
#if 0
      ASSERT(pset == pmass.getParticleSubset());
      ASSERT(pset == pvelocity.getParticleSubset());
#else
      ASSERT(pset->numParticles() == pmass.getParticleSubset()->numParticles());
      ASSERT(pset->numParticles() == pvelocity.getParticleSubset()->numParticles());
#endif

      // Interpolate particle data to Grid data.
      // This currently consists of the particle velocity and mass
      // Need to compute the lumped global mass matrix and velocity
      // Vector from the individual mass matrix and velocity vector (per cell).
      // GridMass * GridVelocity =  S^T*M_D*ParticleVelocity

      gmass.initialize(0);
      gvelocity.initialize(Vector(0,0,0));
      externalforce.initialize(Vector(0,0,0));
      for(ParticleSubset::iterator iter = pset->begin();
	  iter != pset->end(); iter++){
	 particleIndex idx = *iter;

	 // Get the node indices that surround the cell
	 IntVector ni[8];
	 double S[8];
	 if(!region->findCellAndWeights(px[idx], ni, S))
	    continue;
	 // Add each particles contribution to the local mass & velocity 
	 // Must use the node indices
	 for(int k = 0; k < 8; k++) {
	    if(region->containsNode(ni[k])){
	       gmass[ni[k]] += pmass[idx] * S[k];
	       gvelocity[ni[k]] += pvelocity[idx] * pmass[idx] * S[k];
	       externalforce[ni[k]] += pexternalforce[idx] * S[k];
	    }
	 }
      }

      for(NodeIterator iter = region->getNodeIterator(); !iter.done(); iter++){
	 if(gmass[*iter] != 0.0){
	    gvelocity[*iter] *= 1./gmass[*iter];
	 }
      }

      // Apply grid boundary conditions to the velocity
      // before storing the data
      for(int face = 0; face<6; face++){
	Region::FaceType f;
	switch (face) {
	case 0:
	    f = Region::xplus;
	    break;
	case 1:
	  f = Region::xminus;
	  break;
	case 2:
	  f = Region::yplus;
	  break;
	case 3:
	  f = Region::yminus;
	  break;
	case 4:
	  f = Region::zplus;
	  break;
	case 5:
	  f = Region::zminus;
	  break;
	}
	switch(region->getBCType(f)){
	case Region::None:
	     // Do nothing
	     break;
	case Region::Fixed:
	     gvelocity.fillFace(f,Vector(0.0,0.0,0.0));
	     break; 
	case Region::Symmetry:
	     gvelocity.fillFaceNormal(f);
	     break; 
	case Region::Neighbor:
	     // Do nothing
	     break;
	}
      }

      new_dw->put(gmass,         gMassLabel, vfindex, region);
      new_dw->put(gvelocity,     gVelocityLabel, vfindex, region);
      new_dw->put(externalforce, gExternalForceLabel, vfindex, region);
    }
  }
}

void SerialMPM::computeStressTensor(const ProcessorContext*,
				    const Region* region,
				    DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw)
{
   // This needs the datawarehouse to allow indexing by material
   // for both the particle and the grid data.
   for(int m = 0; m < d_sharedState->getNumMatls(); m++){
      Material* matl = d_sharedState->getMaterial(m);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      if(mpm_matl){
	 ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
	 cm->computeStressTensor(region, mpm_matl, old_dw, new_dw);
      }
   }
}

void SerialMPM::updateSurfaceNormalOfBoundaryParticle(const ProcessorContext*,
				    const Region* /*region*/,
				    DataWarehouseP& /*old_dw*/,
				    DataWarehouseP& /*new_dw*/)
{
  //Tan: not finished yet. 
}

void SerialMPM::computeInternalForce(const ProcessorContext*,
				     const Region* region,
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw)
{

  Vector dx = region->dCell();
  double oodx[3];
  oodx[0] = 1.0/dx.x();
  oodx[1] = 1.0/dx.y();
  oodx[2] = 1.0/dx.z();

  // This needs the datawarehouse to allow indexing by material
  // for the particle data and velocity field for the grid data.

  int numMatls = d_sharedState->getNumMatls();

  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int matlindex = matl->getDWIndex();
      int vfindex = matl->getVFIndex();
      // Create arrays for the particle position, volume
      // and the constitutive model
      ParticleVariable<Point>  px;
      ParticleVariable<double>  pvol;
      ParticleVariable<Matrix3> pstress;
      NCVariable<Vector>        internalforce;

      old_dw->get(px,      pXLabel, matlindex, region,
		  Ghost::AroundNodes, 1);
      old_dw->get(pvol,    pVolumeLabel, matlindex, region,
		  Ghost::AroundNodes, 1);
      new_dw->get(pstress, pStressLabel, matlindex, region,
		  Ghost::AroundNodes, 1);

      new_dw->allocate(internalforce, gInternalForceLabel, vfindex, region);
  
      ParticleSubset* pset = px.getParticleSubset();
#if 0
      ASSERT(pset == px.getParticleSubset());
      ASSERT(pset == pvol.getParticleSubset());
      ASSERT(pset == pstress.getParticleSubset());
#else
      ASSERT(pset->numParticles() == px.getParticleSubset()->numParticles());
      ASSERT(pset->numParticles() == pvol.getParticleSubset()->numParticles());
      ASSERT(pset->numParticles() == pstress.getParticleSubset()->numParticles());
#endif

      internalforce.initialize(Vector(0,0,0));

      for(ParticleSubset::iterator iter = pset->begin();
         iter != pset->end(); iter++){
         particleIndex idx = *iter;
  
         // Get the node indices that surround the cell
         IntVector ni[8];
         Vector d_S[8];
         if(!region->findCellAndShapeDerivatives(px[idx], ni, d_S))
  	   continue;

         for (int k = 0; k < 8; k++){
	    if(region->containsNode(ni[k])){
	       Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],d_S[k].z()*oodx[2]);
	       internalforce[ni[k]] -= (div * pstress[idx] * pvol[idx]);
	    }
         }
      }
      new_dw->put(internalforce, gInternalForceLabel, vfindex, region);
    }
  }
}

void SerialMPM::solveEquationsMotion(const ProcessorContext*,
				     const Region* region,
				     DataWarehouseP& /*old_dw*/,
				     DataWarehouseP& new_dw)
{
  Vector zero(0.,0.,0.);
  Vector gravity;

  // This needs the datawarehouse to allow indexing by velocity
  // field for the grid data

  int numMatls = d_sharedState->getNumMatls();

  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfindex = matl->getVFIndex();
      // Get required variables for this region
      NCVariable<double> mass;
      NCVariable<Vector> internalforce;
      NCVariable<Vector> externalforce;

#if 0
      if(vfindex==0){
	gravity=Vector(0.0,0.0,0.0);
      }
      else{
	gravity=Vector(0.0,0.0,-980.0);
      }
#endif

      new_dw->get(mass,          gMassLabel, vfindex, region, Ghost::None, 0);
      new_dw->get(internalforce, gInternalForceLabel, vfindex, region,
		  Ghost::None, 0);
      new_dw->get(externalforce, gExternalForceLabel, vfindex, region,
		  Ghost::None, 0);

      // Create variables for the results
      NCVariable<Vector> acceleration;
      new_dw->allocate(acceleration, gAccelerationLabel, vfindex, region);

      // Do the computation of a = F/m for nodes where m!=0.0
      for(NodeIterator iter = region->getNodeIterator(); !iter.done(); iter++){
	if(mass[*iter]>0.0){
	  acceleration[*iter] =
		 (internalforce[*iter] + externalforce[*iter])/ mass[*iter]
		 + gravity;
	}
	else{
	  acceleration[*iter] = zero;
	}
      }

      // Put the result in the datawarehouse
      new_dw->put(acceleration, gAccelerationLabel, vfindex, region);

    }
  }
}

void SerialMPM::integrateAcceleration(const ProcessorContext*,
				      const Region* region,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw)
{
  // This needs the datawarehouse to allow indexing by material

  int numMatls = d_sharedState->getNumMatls();

  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfindex = matl->getVFIndex();
      // Get required variables for this region
      NCVariable<Vector>        acceleration;
      NCVariable<Vector>        velocity;
      delt_vartype delt;

      new_dw->get(acceleration, gAccelerationLabel, vfindex, region,
		  Ghost::None, 0);
      new_dw->get(velocity, gMomExedVelocityLabel, vfindex, region,
		  Ghost::None, 0);

      old_dw->get((ReductionVariableBase&)delt, deltLabel);

      // Create variables for the results
      NCVariable<Vector> velocity_star;
      new_dw->allocate(velocity_star, gVelocityStarLabel, vfindex, region);

      // Do the computation

      for(NodeIterator iter = region->getNodeIterator(); !iter.done(); iter++){
	velocity_star[*iter] = velocity[*iter] + acceleration[*iter] * delt;
      }


      // Put the result in the datawarehouse
      new_dw->put( velocity_star, gVelocityStarLabel, vfindex, region );
    }
  }
}

void SerialMPM::interpolateToParticlesAndUpdate(const ProcessorContext*,
						const Region* region,
						DataWarehouseP& old_dw,
						DataWarehouseP& new_dw)
{
  // Performs the interpolation from the cell vertices of the grid
  // acceleration and velocity to the particles to update their
  // velocity and position respectively
  Vector vel(0.0,0.0,0.0);
  Vector acc(0.0,0.0,0.0);
  double ke=0;

  // This needs the datawarehouse to allow indexing by material

  int numMatls = d_sharedState->getNumMatls();

  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int matlindex = matl->getDWIndex();
      int vfindex = matl->getVFIndex();
      // Get the arrays of particle values to be changed
      ParticleVariable<Point> px;
      ParticleVariable<Vector> pvelocity;
      ParticleVariable<double> pmass;

      old_dw->get(px,        pXLabel, matlindex, region, Ghost::None, 0);
      old_dw->get(pvelocity, pVelocityLabel, matlindex, region, Ghost::None,0);
      old_dw->get(pmass,     pMassLabel, matlindex, region, Ghost::None, 0);

      // Get the arrays of grid data on which the new particle values depend
      NCVariable<Vector> gvelocity_star;
      NCVariable<Vector> gacceleration;
      delt_vartype delt;

      new_dw->get(gvelocity_star,gMomExedVelocityStarLabel, vfindex, region,
		  Ghost::AroundCells, 1);
      new_dw->get(gacceleration, gMomExedAccelerationLabel, vfindex, region,
		  Ghost::AroundCells, 1);

      // Apply grid boundary conditions to the velocity_star and
      // acceleration before interpolating back to the particles
      for(int face = 0; face<6; face++){
	// Dummy holder until this is resolved
	Region::FaceType f = Region::xplus;
	Region::BCType bctype = region->getBCType(f);
	switch(bctype){
	  case Region::None:
	     // Do nothing
	     break;
	  case Region::Fixed:
	     gvelocity_star.fillFace(f,Vector(0.0,0.0,0.0));
	     gacceleration.fillFace(f,Vector(0.0,0.0,0.0));
	     break;
	  case Region::Symmetry:
	     gvelocity_star.fillFaceNormal(f);
	     gacceleration.fillFaceNormal(f);
	     break;
	  case Region::Neighbor:
	     // Do nothing
	     break;
	}
      }

      old_dw->get(delt, deltLabel);

      ParticleSubset* pset = px.getParticleSubset();
      ASSERT(pset == pvelocity.getParticleSubset());

      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
	 particleIndex idx = *iter;

        // Get the node indices that surround the cell
	IntVector ni[8];
        double S[8];
        if(!region->findCellAndWeights(px[idx], ni, S))
	  continue;

        vel = Vector(0.0,0.0,0.0);
        acc = Vector(0.0,0.0,0.0);

        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < 8; k++) {
	   vel += gvelocity_star[ni[k]]  * S[k];
	   acc += gacceleration[ni[k]]   * S[k];
        }

        // Update the particle's position and velocity
        px[idx]        += vel * delt;
        pvelocity[idx] += acc * delt;
        ke += .5*pmass[idx]*pvelocity[idx].length2();

       // If we were storing particles in cellwise lists, this
       // is where we would update the lists so that each particle
       // is in the correct cells list

      }

#if 0
      static ofstream tmpout2("tmp2.out");
      tmpout2 << ts << " " << px[0] << std::endl;
#endif

      // Store the new result
      new_dw->put(px,        pXLabel, matlindex, region);
      new_dw->put(pvelocity, pVelocityLabel, matlindex, region);

      ParticleVariable<double> pvolume;
      ParticleVariable<Vector> pexternalforce;

      new_dw->put(pmass,          pMassLabel, matlindex, region);
      old_dw->get(pvolume,        pVolumeLabel, matlindex, region,
		  Ghost::None, 0);
      new_dw->put(pvolume,        pVolumeLabel, matlindex, region);
      old_dw->get(pexternalforce, pExternalForceLabel, matlindex, region,
		  Ghost::None, 0);
      new_dw->put(pexternalforce, pExternalForceLabel, matlindex, region);
    }
  }

  static ofstream tmpout("tmp.out");
  static int ts=0;
  tmpout << ts << " " << ke << std::endl;
  ts++;
}

void SerialMPM::crackGrow(const ProcessorContext*,
                          const Region* /*region*/,
                          DataWarehouseP& /*old_dw*/,
                          DataWarehouseP& /*new_dw*/)
{
}

// $Log$
// Revision 1.55  2000/05/15 19:02:44  tan
// A little change on initializedFracture Interface.
//
// Revision 1.54  2000/05/15 18:59:46  tan
// Initialized NCVariables and CCVaribles for Fracture.
//
// Revision 1.53  2000/05/12 01:45:17  tan
// Added call to initializeFracture in SerialMPM's actuallyInitailize.
//
// Revision 1.52  2000/05/11 20:10:12  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.51  2000/05/10 20:02:42  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made regions have a single uniform index space - still needs work
//
// Revision 1.50  2000/05/10 18:34:15  tan
// Added computations on self-contact cells for cracked surfaces.
//
// Revision 1.49  2000/05/10 05:01:33  tan
// linked to farcture model.
//
// Revision 1.48  2000/05/09 23:45:09  jas
// Fixed the application of grid boundary conditions.  It is probably slow
// as mud but hopefully the gist is right.
//
// Revision 1.47  2000/05/09 21:33:02  guilkey
// Added gravity to the acceleration.  Currently hardwired, I need a little
// help seeing how to get it out of the ProblemSpec.
//
// Revision 1.46  2000/05/09 03:27:55  jas
// Using the enums for boundary conditions hack.
//
// Revision 1.45  2000/05/08 18:46:16  guilkey
// Added call to initializeContact in SerialMPM's actuallyInitailize
//
// Revision 1.44  2000/05/08 17:16:51  tan
// Added grid VarLabel selfContactLabel for fracture simulation.
//
// Revision 1.43  2000/05/07 06:02:01  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.42  2000/05/04 23:40:00  tan
// Added fracture interface to general MPM.
//
// Revision 1.41  2000/05/04 19:10:52  guilkey
// Added code to apply boundary conditions.  This currently calls empty
// functions which will be implemented soon.
//
// Revision 1.40  2000/05/04 17:30:32  tan
//   Add surfaceNormal for boundary particle tracking.
//
// Revision 1.39  2000/05/03 23:52:44  guilkey
// Fixed some small errors in the MPM code to make it work
// and give what appear to be correct answers.
//
// Revision 1.38  2000/05/02 18:41:15  guilkey
// Added VarLabels to the MPM algorithm to comply with the
// immutable nature of the DataWarehouse. :)
//
// Revision 1.37  2000/05/02 17:54:21  sparker
// Implemented more of SerialMPM
//
// Revision 1.36  2000/05/02 06:07:08  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.35  2000/05/01 16:18:07  sparker
// Completed more of datawarehouse
// Initial more of MPM data
// Changed constitutive model for bar
//
// Revision 1.34  2000/04/28 21:08:15  jas
// Added exception to the creation of Contact factory if contact is not
// specified.
//
// Revision 1.33  2000/04/28 08:11:29  sparker
// ConstitutiveModelFactory should return null on failure
// MPMMaterial checks for failed constitutive model creation
// DWindex and VFindex are now initialized
// Fixed input file to match ConstitutiveModelFactory
//
// Revision 1.32  2000/04/28 07:35:26  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.31  2000/04/27 23:18:41  sparker
// Added problem initialization for MPM
//
// Revision 1.30  2000/04/27 21:39:27  jas
// Now creating contact via a factory.
//
// Revision 1.29  2000/04/26 06:48:12  sparker
// Streamlined namespaces
//
// Revision 1.28  2000/04/25 22:57:29  guilkey
// Fixed Contact stuff to include VarLabels, SimulationState, etc, and
// made more of it compile.
//
// Revision 1.27  2000/04/25 00:41:19  dav
// more changes to fix compilations
//
// Revision 1.26  2000/04/24 21:04:24  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.25  2000/04/24 15:16:58  sparker
// Fixed unresolved symbols
//
// Revision 1.24  2000/04/21 17:46:52  jas
// Inserted & for function pointers needed by Task constructor (required by
// gcc).  Moved around WONT_COMPILE_YET to get around allocate.
//
// Revision 1.23  2000/04/20 23:20:26  dav
// updates
//
// Revision 1.22  2000/04/20 22:13:36  dav
// making SerialMPM compile
//
// Revision 1.21  2000/04/20 18:56:16  sparker
// Updates to MPM
//
// Revision 1.20  2000/04/19 22:38:16  dav
// Make SerialMPM a UintahParallelComponent
//
// Revision 1.19  2000/04/19 22:30:00  dav
// somehow SerialMPM.cc had cvs diff stuff in it.  I have removed it.  I hope I removed the right stuff.
//
// Revision 1.18  2000/04/19 21:20:01  dav
// more MPI stuff
//
// Revision 1.17  2000/04/19 05:26:01  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.16  2000/04/14 02:16:33  jas
// Added variable to the problemSetup declaration.
//
// Revision 1.15  2000/04/14 02:13:57  jas
// problemSetup runs the preProcessor.
//
// Revision 1.14  2000/04/13 06:50:55  sparker
// More implementation to get this to work
//
// Revision 1.13  2000/04/12 22:59:03  sparker
// Working to make it compile
// Added xerces to link line
//
// Revision 1.12  2000/04/12 16:57:23  guilkey
// Converted the SerialMPM.cc to have multimaterial/multivelocity field
// capabilities.  Tried to guard all the functions against breaking the
// compilation, but then who really cares?  It's not like sus has compiled
// for more than 5 minutes in a row for two months.
//
// Revision 1.11  2000/03/23 20:42:16  sparker
// Added copy ctor to exception classes (for Linux/g++)
// Helped clean up move of ProblemSpec from Interface to Grid
//
// Revision 1.10  2000/03/21 02:13:56  dav
// commented out bad portions of code so I could check in a version that compiles
//
// Revision 1.9  2000/03/21 01:29:39  dav
// working to make MPM stuff compile successfully
//
// Revision 1.8  2000/03/20 17:17:05  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.7  2000/03/17 21:01:50  dav
// namespace mods
//
// Revision 1.6  2000/03/17 02:57:01  dav
// more namespace, cocoon, etc
//
// Revision 1.5  2000/03/16 01:11:23  guilkey
// To timeStep added tasks to do contact.
//
// Revision 1.4  2000/03/15 22:13:04  jas
// Added log and changed header file locations.
//

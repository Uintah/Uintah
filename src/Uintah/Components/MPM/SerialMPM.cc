/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Components/MPM/SerialMPM.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Components/MPM/Contact/ContactFactory.h>
#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
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

using namespace Uintah::MPM;

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Math::Max;
using namespace std;

SerialMPM::SerialMPM( int MpiRank, int MpiProcesses ) :
  UintahParallelComponent( MpiRank, MpiProcesses )
{
   pDeformationMeasureLabel = 
               new VarLabel("p.deformationMeasure",
			    ParticleVariable<Matrix3>::getTypeDescription());
   pStressLabel = 
               new VarLabel( "p.stress",
			     ParticleVariable<Matrix3>::getTypeDescription() );

   pVolumeLabel = 
               new VarLabel( "p.volume",
			     ParticleVariable<double>::getTypeDescription());
   pMassLabel = new VarLabel( "p.mass",
			      ParticleVariable<double>::getTypeDescription() );
   pVelocityLabel =
               new VarLabel( "p.velocity", 
			     ParticleVariable<Vector>::getTypeDescription() );
   pExternalForceLabel =
               new VarLabel( "p.externalforce",
			     ParticleVariable<Vector>::getTypeDescription() );
   pXLabel =   new VarLabel( "p.x",
			     ParticleVariable<Point>::getTypeDescription(),
			     VarLabel::PositionVariable);

   gAccelerationLabel =
                new VarLabel( "g.acceleration",
			      NCVariable<Vector>::getTypeDescription() );
   gMassLabel = new VarLabel( "g.mass",
			      NCVariable<double>::getTypeDescription() );
   gVelocityLabel = new VarLabel( "g.velocity",
				  NCVariable<Vector>::getTypeDescription() );
   gExternalForceLabel =
                new VarLabel( "g.externalforce",
			      NCVariable<Vector>::getTypeDescription() );
   gInternalForceLabel =
                new VarLabel( "g.internalforce",
			      NCVariable<Vector>::getTypeDescription() );
   gVelocityStarLabel =
                new VarLabel( "g.velocity_star",
			      NCVariable<Vector>::getTypeDescription() );


   // I'm not sure about this one:
   deltLabel = 
     new VarLabel( "delt",
		   ReductionVariable<double>::getTypeDescription() );

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
   cerr << "SerialMPM::problemSetup not done\n";
}

void SerialMPM::scheduleInitialize(const LevelP& level,
				   SchedulerP& sched,
				   DataWarehouseP& dw)
{
   for(Level::const_regionIterator iter=level->regionsBegin();
       iter != level->regionsEnd(); iter++){
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
				    const DataWarehouseP& old_dw, 
				          DataWarehouseP& new_dw)
{
   for(Level::const_regionIterator iter=level->regionsBegin();
       iter != level->regionsEnd(); iter++){
      const Region* region=*iter;
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
	 t->requires(old_dw, pMassLabel, region, 0 );
	 t->requires(old_dw, pVelocityLabel, region, 0 );
	 t->requires(old_dw, pExternalForceLabel, region, 0 );
	 t->requires(old_dw, pXLabel, region, 0 );

	 t->computes(new_dw, gMassLabel, region );
	 t->computes(new_dw, gVelocityLabel, region );
	 t->computes(new_dw, gExternalForceLabel, region );
		     
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
	 t->requires( new_dw, gMassLabel, region, 0 );
	 t->requires( new_dw, gVelocityLabel, region, 0 );

	 t->computes( new_dw, gVelocityLabel, region );

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
	 t->requires(new_dw, gVelocityLabel, region, 0);
	 /*
	   #warning
	   t->requires(old_dw, "p.cmdata", region, 0,
	   ParticleVariable<Uintah::Components::
	   ConstitutiveModel::CMData>::getTypeDescription());
	   */
	 t->requires(new_dw, pDeformationMeasureLabel, region, 0);

	 t->computes(new_dw, d_sharedState->get_delt_label());
	 t->computes(new_dw, pStressLabel, region);
	 t->computes(new_dw, pDeformationMeasureLabel, region);

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
	 t->requires( new_dw, pStressLabel, region, 0 );
	 t->requires( old_dw, pVolumeLabel, region, 0 );

	 t->computes( new_dw, gInternalForceLabel, region );

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
	 t->requires( new_dw, gMassLabel, region, 0 );
	 t->requires( new_dw, gInternalForceLabel, region, 0 );

	 t->computes( new_dw, gAccelerationLabel, region );

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
	 t->requires(new_dw, gAccelerationLabel, region, 0 );
	 t->requires(new_dw, gVelocityLabel, region, 0 );
	 t->requires(old_dw, deltLabel );
		     
	 t->computes(new_dw, gVelocityStarLabel, region );
		     
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
	t->requires(new_dw, gMassLabel, region, 0);
	t->requires(new_dw, gVelocityStarLabel, region, 0);
	t->requires(new_dw, gAccelerationLabel, region, 0);

	t->computes(new_dw, gVelocityStarLabel, region);
	t->computes(new_dw, gAccelerationLabel, region);

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
	 t->requires(new_dw, gAccelerationLabel, region, 0 );
	 t->requires(new_dw, gVelocityStarLabel, region, 0 );

	 t->requires(old_dw, pXLabel, region, 0 );
	 t->requires(old_dw, deltLabel );
	 t->computes(new_dw, pVelocityLabel, region );
	 t->computes(new_dw, pXLabel, region );

	 sched->addTask(t);
      }
    }
}

void SerialMPM::actuallyInitialize(const ProcessorContext*,
				   const Region* region,
				   const DataWarehouseP& /* old_dw */,
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
       mpm_matl->getConstitutiveModel()->initializeCMData(region, mpm_matl, new_dw);
    }
  }
}


void SerialMPM::actuallyComputeStableTimestep(const ProcessorContext*,
					      const Region*,
					      const DataWarehouseP&,
					      DataWarehouseP&)
{
}

void SerialMPM::interpolateParticlesToGrid(const ProcessorContext*,
					   const Region* region,
					   const DataWarehouseP& old_dw,
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
      ParticleVariable<Vector> px;
      ParticleVariable<double> pmass;
      ParticleVariable<Vector> pvelocity;
      ParticleVariable<Vector> pexternalforce;

      old_dw->get(px,             pXLabel, matlindex, region, 0);
      old_dw->get(pmass,          pMassLabel, matlindex, region, 0);
      old_dw->get(pvelocity,      pVelocityLabel, vfindex, region, 0);
      old_dw->get(pexternalforce, pExternalForceLabel, matlindex, region, 0);

      // Create arrays for the grid data
      NCVariable<double> gmass;
      NCVariable<Vector> gvelocity;
      NCVariable<Vector> externalforce;

      new_dw->allocate(gmass,         gMassLabel, vfindex, region);
      new_dw->allocate(gvelocity,     gVelocityLabel, vfindex, region);
      new_dw->allocate(externalforce, gExternalForceLabel, vfindex, region);

      ParticleSubset* pset = px.getParticleSubset();
      ASSERT(pset == pmass.getParticleSubset());
      ASSERT(pset == pvelocity.getParticleSubset());

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
	Array3Index ni[8];
	double S[8];
	if(!region->findCellAndWeights(px[idx], ni, S))
	    continue;
	// Add each particles contribution to the local mass & velocity 
	// Must use the node indices
	for(int k = 0; k < 8; k++) {
	    //if(region->contains(ni[k])){
		gmass[ni[k]] += pmass[idx] * S[k];
		gvelocity[ni[k]] += pvelocity[idx] * pmass[idx] * S[k];
		externalforce[ni[k]] += pexternalforce[idx] * S[k];
		//}
	}
      }

      for(NodeIterator iter = region->begin(); iter != region->end(); iter++){
	if(gmass[*iter] != 0.0){
	    gvelocity[*iter] *= 1./gmass[*iter];
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
				    const DataWarehouseP& old_dw,
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

void SerialMPM::computeInternalForce(const ProcessorContext*,
				     const Region* region,
				     const DataWarehouseP& old_dw,
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
      ParticleVariable<Vector>  px;
      ParticleVariable<double>  pvol;
      ParticleVariable<Matrix3> pstress;
      NCVariable<Vector>        internalforce;

      old_dw->get(px,      pXLabel, matlindex, region, 0);
      old_dw->get(pvol,    pVolumeLabel, matlindex, region, 0);
      old_dw->get(pstress, pStressLabel, matlindex, region, 0);

      new_dw->allocate(internalforce, gInternalForceLabel, vfindex, region);
  
      ParticleSubset* pset = px.getParticleSubset();
      ASSERT(pset == px.getParticleSubset());
      ASSERT(pset == pvol.getParticleSubset());
      ASSERT(pset == pstress.getParticleSubset());

      internalforce.initialize(Vector(0,0,0));

      for(ParticleSubset::iterator iter = pset->begin();
         iter != pset->end(); iter++){
         particleIndex idx = *iter;
  
         // Get the node indices that surround the cell
         Array3Index ni[8];
         Vector d_S[8];
         if(!region->findCellAndShapeDerivatives(px[idx], ni, d_S))
  	   continue;

         for (int k = 0; k < 8; k++){
  	   Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],d_S[k].z()*oodx[2]);
  	   internalforce[ni[k]] -= (div * pstress[idx] * pvol[idx]);
         }
      }
      new_dw->put(internalforce, gInternalForceLabel, vfindex, region);
    }
  }
}

void SerialMPM::solveEquationsMotion(const ProcessorContext*,
				     const Region* region,
				     const DataWarehouseP& /*old_dw*/,
				     DataWarehouseP& new_dw)
{
  Vector zero(0.,0.,0.);

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

      new_dw->get(mass,          gMassLabel, vfindex, region, 0);
      new_dw->get(internalforce, gInternalForceLabel, vfindex, region, 0);
      new_dw->get(externalforce, gExternalForceLabel, vfindex, region, 0);

      // Create variables for the results
      NCVariable<Vector> acceleration;
      new_dw->allocate(acceleration, gAccelerationLabel, vfindex, region);

      // Do the computation of a = F/m for nodes where m!=0.0
      for(NodeIterator  iter  = region->begin();
			iter != region->end(); iter++){
	if(mass[*iter]>0.0){
	  acceleration[*iter] =
		 (internalforce[*iter] + externalforce[*iter])/ mass[*iter];
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
				      const DataWarehouseP& old_dw,
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
      ReductionVariable<double> delt;

      new_dw->get(acceleration, gAccelerationLabel, vfindex, region, 0);
      new_dw->get(velocity, gVelocityLabel, vfindex, region, 0);

      old_dw->get(delt, deltLabel);

      // Create variables for the results
      NCVariable<Vector> velocity_star;
      new_dw->allocate(velocity_star, gVelocityStarLabel, vfindex, region);

      // Do the computation

      for(NodeIterator  iter  = region->begin();
			iter != region->end(); iter++) {
	velocity_star[*iter] = velocity[*iter] + acceleration[*iter] * delt;
      }


      // Put the result in the datawarehouse
      new_dw->put( velocity_star, gVelocityStarLabel, vfindex, region );
    }
  }
}

void SerialMPM::interpolateToParticlesAndUpdate(const ProcessorContext*,
						const Region* region,
						const DataWarehouseP& old_dw,
						DataWarehouseP& new_dw)
{
  // Performs the interpolation from the cell vertices of the grid
  // acceleration and velocity to the particles to update their
  // velocity and position respectively
  Vector vel(0.0,0.0,0.0);
  Vector acc(0.0,0.0,0.0);

  // This needs the datawarehouse to allow indexing by material

  int numMatls = d_sharedState->getNumMatls();

  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int matlindex = matl->getDWIndex();
      int vfindex = matl->getVFIndex();
      // Get the arrays of particle values to be changed
      ParticleVariable<Vector> px;
      ParticleVariable<Vector> pvelocity;

      old_dw->get(px,        pXLabel, matlindex, region, 0);
      old_dw->get(pvelocity, pVelocityLabel, matlindex, region, 0);

      // Get the arrays of grid data on which the new particle values depend
      NCVariable<Vector> gvelocity_star;
      NCVariable<Vector> gacceleration;
      ReductionVariable<double> delt;

      new_dw->get(gvelocity_star, gVelocityStarLabel, vfindex, region, 0);
      new_dw->get(gacceleration,  gAccelerationLabel, vfindex, region, 0);

      old_dw->get(delt, deltLabel);

      ParticleSubset* pset = px.getParticleSubset();
      ASSERT(pset == pvelocity.getParticleSubset());

      double ke=0;
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
	 particleIndex idx = *iter;

        // Get the node indices that surround the cell
        Array3Index ni[8];
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
        ke += pvelocity[idx].length2();

       // If we were storing particles in cellwise lists, this
       // is where we would update the lists so that each particle
       // is in the correct cells list

      }

      static ofstream tmpout("tmp.out");
      static int ts=0;
      tmpout << ts << " " << ke << std::endl;
    
      static ofstream tmpout2("tmp2.out");
      tmpout2 << ts << " " << px[5] << std::endl;
      ts++;

      // Store the new result
      new_dw->put(px,        pXLabel, matlindex, region);
      new_dw->put(pvelocity, pVelocityLabel, matlindex, region);

      ParticleVariable<double> pmass;
      ParticleVariable<double> pvolume;
      ParticleVariable<Vector> pexternalforce;

      old_dw->get(pmass,          pMassLabel, matlindex, region, 0);
      new_dw->put(pmass,          pMassLabel, matlindex, region);
      old_dw->get(pvolume,        pVolumeLabel, matlindex, region, 0);
      new_dw->put(pvolume,        pVolumeLabel, matlindex, region);
      old_dw->get(pexternalforce, pExternalForceLabel, matlindex, region, 0);
      new_dw->put(pexternalforce, pExternalForceLabel, matlindex, region);
    }
  }
}

// $Log$
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

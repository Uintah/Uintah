/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Uintah/Components/MPM/Contact/Contact.h>
#include <Uintah/Components/MPM/SerialMPM.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>

#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Grid/NodeIterator.h> // Must be included after Region.h
#include <Uintah/Grid/SoleVariable.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Interface/Scheduler.h>

#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/MinMax.h>

#include <iostream>
#include <fstream>

#include "GeometrySpecification/Problem.h"

namespace Uintah {
namespace Components {

using Uintah::Interface::Scheduler;

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Math::Max;
using std::cerr;
using std::string;
using std::ofstream;
using Uintah::Grid::Level;
using Uintah::Grid::ParticleSet;
using Uintah::Grid::ParticleSubset;
using Uintah::Grid::ParticleVariable;
using Uintah::Grid::Task;
using Uintah::Grid::SoleVariable;
using Uintah::Interface::ProblemSpec;
using Uintah::Grid::Region;
using Uintah::Grid::NodeIterator;
using Uintah::Grid::NCVariable;

SerialMPM::SerialMPM()
{
}

SerialMPM::~SerialMPM()
{
}

void SerialMPM::problemSetup(const ProblemSpecP&, GridP& grid,
			     DataWarehouseP& dw)
{

  Problem prob_description;
  prob_description.preProcessor(prob_spec,grid);  
#if 0
    for(Level::const_regionIterator iter=level->regionsBegin();
	iter != level->regionsEnd(); iter++){
	const Region* region=*iter;
	ParticleSet* pset = new ParticleSet();
	ParticleSubset* psubset = new ParticleSubset(pset);
	ParticleVariable<Vector> px(psubset);
	dw->put(px, "p.x", region, 0);
	ParticleVariable<double> pvolume(psubset);
	dw->put(pvolume, "p.volume", region, 0);
	ParticleVariable<double> pmass(psubset);
	dw->put(pmass, "p.mass", region, 0);
	ParticleVariable<Vector> pvel(psubset);
	dw->put(pvel, "p.velocity", region, 0);
	ParticleVariable<Matrix3> pstress(psubset);
	dw->put(pstress, "p.stress", region, 0);
	ParticleVariable<Matrix3> pdeformationMeasure(psubset);
	dw->put(pdeformationMeasure, "p.deformationMeasure", region, 0);
	ParticleVariable<Vector> pexternalforce(psubset);
	dw->put(pexternalforce, "p.externalforce", region, 0);
	cerr << "Creating particles for region\n";
	prob_description.createParticles(region, dw);
    }
#endif
    cerr << "SerialMPM::problemSetup not done\n";
}

void SerialMPM::scheduleStableTimestep(const LevelP& level,
				      SchedulerP& sched, DataWarehouseP& dw)
{
    for(Level::const_regionIterator iter=level->regionsBegin();
	iter != level->regionsEnd(); iter++){
	const Region* region=*iter;

	Task* t = new Task("SerialMPM::computeStableTimestep", region, dw, dw,
			   this, SerialMPM::actuallyComputeStableTimestep);
	t->requires(dw, "params", ProblemSpec::getTypeDescription());
	t->requires(dw, "MaxWaveSpeed",
				SoleVariable<double>::getTypeDescription());
	t->computes(dw, "delt", SoleVariable<double>::getTypeDescription());
	sched->addTask(t);
    }
}

void SerialMPM::scheduleTimeAdvance(double t, double dt,
				    const LevelP& level, SchedulerP& sched,
				    const DataWarehouseP& old_dw, DataWarehouseP& new_dw)
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
			       this, SerialMPM::interpolateParticlesToGrid);
	    t->requires(old_dw, "p.mass", region, 0,
			ParticleVariable<double>::getTypeDescription());
	    t->requires(old_dw, "p.velocity", region, 0,
			ParticleVariable<Vector>::getTypeDescription());
	    t->requires(old_dw, "p.externalforce", region, 0,
			ParticleVariable<Vector>::getTypeDescription());
	    t->requires(old_dw, "p.x", region, 0,
			ParticleVariable<Point>::getTypeDescription());
	    t->computes(new_dw, "g.mass", region, 0,
			NCVariable<double>::getTypeDescription());
	    t->computes(new_dw, "g.velocity", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->computes(new_dw, "g.externalforce", region, 0,
			NCVariable<Vector>::getTypeDescription());
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
#if 0
	    Task* t = new Task("Contact::exMomInterpolated",
			       region, old_dw, new_dw,
			       this, Contact::dav); // Contact::exMomInterpolated);
	    t->requires(new_dw, "g.mass", region, 0,
			NCVariable<double>::getTypeDescription());
	    t->requires(new_dw, "g.velocity", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->computes(new_dw, "g.velocity", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    sched->addTask(t);
#endif
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
			       this, SerialMPM::computeStressTensor);
	    t->requires(new_dw, "g.velocity", region, 0,
			NCVariable<Vector>::getTypeDescription());
/*
#warning
	    t->requires(old_dw, "p.cmdata", region, 0,
			ParticleVariable<Uintah::Components::
                            ConstitutiveModel::CMData>::getTypeDescription());
*/
	    t->requires(new_dw, "p.deformationMeasure", region, 0,
			ParticleVariable<Matrix3>::getTypeDescription());
	    t->requires(old_dw, "delt",
			SoleVariable<double>::getTypeDescription());
	    t->computes(new_dw, "p.stress", region, 0,
			ParticleVariable<Matrix3>::getTypeDescription());
	    t->computes(new_dw, "p.deformationMeasure", region, 0,
			ParticleVariable<Matrix3>::getTypeDescription());
	    t->computes(new_dw, "MaxWaveSpeed",
			SoleVariable<double>::getTypeDescription());
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
			       this, SerialMPM::computeInternalForce);
	    t->requires(new_dw, "p.stress", region, 0,
			ParticleVariable<Matrix3>::getTypeDescription());
	    t->requires(old_dw, "p.volume", region, 0,
			ParticleVariable<double>::getTypeDescription());
	    t->computes(new_dw, "g.internalforce", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    sched->addTask(t);
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
			       this, SerialMPM::solveEquationsMotion);
	    t->requires(new_dw, "g.mass", region, 0,
			NCVariable<double>::getTypeDescription());
	    t->requires(new_dw, "g.internalforce", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->computes(new_dw, "g.acceleration", region, 0,
			NCVariable<Vector>::getTypeDescription());
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
			       this, SerialMPM::integrateAcceleration);
	    t->requires(new_dw, "g.acceleration", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->requires(new_dw, "g.velocity", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->requires(old_dw, "delt",
			SoleVariable<double>::getTypeDescription());
	    t->computes(new_dw, "g.velocity_star", region, 0,
			NCVariable<Vector>::getTypeDescription());
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
/*
#warning
	    Task* t = new Task("Contact::exMomIntegrated",
				region, old_dw, new_dw,
				this, Contact::exMomIntegrated);
	    t->requires(new_dw, "g.mass", region, 0,
			NCVariable<double>::getTypeDescription());
	    t->requires(new_dw, "g.velocity_star", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->requires(new_dw, "g.acceleration", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->computes(new_dw, "g.velocity_star", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->computes(new_dw, "g.acceleration", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    sched->addTask(t);
*/
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
			      this, SerialMPM::interpolateToParticlesAndUpdate);
	    t->requires(new_dw, "g.acceleration", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->requires(new_dw, "g.velocity_star", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->requires(old_dw, "p.x", region, 0,
			ParticleVariable<Point>::getTypeDescription());
	    t->requires(old_dw, "delt",
			SoleVariable<double>::getTypeDescription());
	    t->computes(new_dw, "p.velocity", region, 0,
			ParticleVariable<Vector>::getTypeDescription());
	    t->computes(new_dw, "p.x", region, 0,
			ParticleVariable<Point>::getTypeDescription());
	    sched->addTask(t);
	}
    }
}

void SerialMPM::actuallyComputeStableTimestep(const ProcessorContext*,
					      const Region* region,
					      const DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw)
{
    using SCICore::Math::Min;

    SoleVariable<double> MaxWaveSpeed;
    new_dw->get(MaxWaveSpeed, "MaxWaveSpeed");

    Vector dCell = region->dCell();
    double width = Min(dCell.x(), dCell.y(), dCell.z());
    double delt = 0.5*width/MaxWaveSpeed;
/*
 DataWarehouse needs a Min function implemented
    new_dw->put(SoleVariable<double>(delt), "delt", DataWarehouse::Min);
*/
}

void SerialMPM::interpolateParticlesToGrid(const ProcessorContext*,
					   const Region* region,
					   const DataWarehouseP& old_dw,
					   DataWarehouseP& new_dw)
{
#if 0  // This needs the datawarehouse to allow indexing by material
       // for the particle data and velocity field by the grid data

  for(int m = 0; m < numMatls; m++){
    Material* matl = materials[m];
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int matlindex = matl->getDWIndex();
      int vfindex = matl->getVFIndex();
      // Create arrays for the particle data
      ParticleVariable<Vector> px;
      old_dw->get(px, "p.x", matlindex, region, 0);
      ParticleVariable<double> pmass;
      old_dw->get(pmass, "p.mass", matlindex, region, 0);
      ParticleVariable<Vector> pvelocity;
      old_dw->get(pvelocity, "p.velocity", matlindex, region, 0);
      ParticleVariable<Vector> pexternalforce;
      old_dw->get(pexternalforce, "p.externalforce", matlindex, region, 0);

      // Create arrays for the grid data
      NCVariable<double> gmass;
      new_dw->allocate(gmass, "g.mass", vfindex, region, 0);
      NCVariable<Vector> gvelocity;
      new_dw->allocate(gvelocity, "g.velocity", vfindex, region, 0);
      NCVariable<Vector> externalforce;
      new_dw->allocate(externalforce, "g.externalforce", vfindex, region, 0);

      ParticleSubset* pset = px.getParticleSubset(matlindex);
      ASSERT(pset == pmass.getParticleSubset(matlindex));
      ASSERT(pset == pvelocity.getParticleSubset(matlindex));

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
	Uintah::Grid::particleIndex idx = *iter;

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

      for(NodeIterator iter = region->begin();
	iter != region->end(); iter++){
	if(gmass[*iter] != 0.0){
	    gvelocity[*iter] *= 1./gmass[*iter];
	}
      }

      new_dw->put(gmass, "g.mass", vfindex, region, 0);
      new_dw->put(gvelocity, "g.velocity", vfindex, region, 0);
      new_dw->put(externalforce, "g.externalforce", vfindex, region, 0);
    }
  }
#endif
}


void SerialMPM::computeStressTensor(const ProcessorContext*,
				    const Region* region,
				    const DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw)
{
#if 0 // This needs the datawarehouse to allow indexing by material
      // for both the particle and the grid data.
    for(int m = 0; m < numMatls; m++){
        Material* matl = materials[m];
        MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
        if(mpm_matl){
            ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
            cm->computeStressTensor(region, mpm_matl, old_dw, new_dw);
	}
    }
#endif
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

#if 0  // This needs the datawarehouse to allow indexing by material
       // for the particle data and velocity field for the grid data.

  for(int m = 0; m < numMatls; m++){
    Material* matl = materials[m];
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int matlindex = matl->getDWIndex();
      int vfindex = matl->getVFIndex();
      // Create arrays for the particle position, volume
      // and the constitutive model
      ParticleVariable<Vector> px;
      old_dw->get(px, "p.x", matlindex, region, 0);
      ParticleVariable<double> pvol;
      old_dw->get(pvol, "p.volume", matlindex, region, 0);
      ParticleVariable<Matrix3> pstress;
      old_dw->get(pstress, "p.stress", matlindex, region, 0);

      NCVariable<Vector> internalforce;
      new_dw->allocate(internalforce, "g.internalforce", vfindex, region, 0);
  
      ParticleSubset* pset = px.getParticleSubset(matlindex);
      ASSERT(pset == px.getParticleSubset(matlindex));
      ASSERT(pset == pvol.getParticleSubset(matlindex));
      ASSERT(pset == pstress.getParticleSubset(matlindex));

      internalforce.initialize(Vector(0,0,0));

      for(ParticleSubset::iterator iter = pset->begin();
         iter != pset->end(); iter++){
         Uintah::Grid::particleIndex idx = *iter;
  
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

      new_dw->put(internalforce, "g.internalforce", vfindex, region, 0);
    }
  }
#endif
}

void SerialMPM::solveEquationsMotion(const ProcessorContext*,
				     const Region* region,
				     const DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw)
{
    Vector zero(0.,0.,0.);

#if 0  // This needs the datawarehouse to allow indexing by velocity
       // field for the grid data

  for(int m = 0; m < numMatls; m++){
    Material* matl = materials[m];
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfindex = matl->getVFIndex();
      // Get required variables for this region
      NCVariable<double> mass;
      new_dw->get(mass, "g.mass", vfindex, region, 0);
      NCVariable<Vector> internalforce;
      new_dw->get(internalforce, "g.internalforce", vfindex, region, 0);
      NCVariable<Vector> externalforce;
      new_dw->get(externalforce, "g.externalforce", vfindex, region, 0);

      // Create variables for the results
      NCVariable<Vector> acceleration;
      new_dw->allocate(acceleration, "g.acceleration", vfindex, region, 0);

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
      new_dw->put(acceleration, "g.acceleration", vfindex, region, 0);

    }
  }
#endif
}

void SerialMPM::integrateAcceleration(const ProcessorContext*,
				      const Region* region,
				      const DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw)
{
#if 0  // This needs the datawarehouse to allow indexing by material

  for(int m = 0; m < numMatls; m++){
    Material* matl = materials[m];
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfindex = matl->getVFIndex();
      // Get required variables for this region
      NCVariable<Vector> acceleration;
      new_dw->get(acceleration, "g.acceleration", vfindex, region, 0);
      NCVariable<Vector> velocity;
      new_dw->get(velocity, "g.velocity", vfindex, region, 0);
      SoleVariable<double> delt;
      old_dw->get(delt, "delt");

      // Create variables for the results
      NCVariable<Vector> velocity_star;
      new_dw->allocate(velocity_star, "g.velocity_star", vfindex, region, 0);

      // Do the computation

      for(NodeIterator  iter  = region->begin();
			iter != region->end(); iter++)
	velocity_star[*iter] = velocity[*iter] + acceleration[*iter] * delt;

      // Put the result in the datawarehouse
      new_dw->put(velocity_star, "g.velocity_star", vfindex, region, 0);
    }
  }
#endif
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

#if 0  // This needs the datawarehouse to allow indexing by material

  for(int m = 0; m < numMatls; m++){
    Material* matl = materials[m];
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int matlindex = matl->getDWIndex();
      int vfindex = matl->getVFIndex();
      // Get the arrays of particle values to be changed
      ParticleVariable<Vector> px;
      old_dw->get(px, "p.x", matlindex, region, 0);
      ParticleVariable<Vector> pvelocity;
      old_dw->get(pvelocity, "p.velocity", matlindex, region, 0);

      // Get the arrays of grid data on which the new particle values depend
      NCVariable<Vector> gvelocity_star;
      new_dw->get(gvelocity_star, "g.velocity_star", vfindex, region, 0);
      NCVariable<Vector> gacceleration;
      new_dw->get(gacceleration, "g.acceleration", vfindex, region, 0);
      SoleVariable<double> delt;
      old_dw->get(delt, "delt");

      ParticleSubset* pset = px.getParticleSubset(matlindex);
      ASSERT(pset == pvelocity.getParticleSubset(matlindex));

      double ke=0;
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        Uintah::Grid::particleIndex idx = *iter;

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
      new_dw->put(px, "p.x", matlindex, region, 0);
      new_dw->put(pvelocity, "p.velocity", matlindex, region, 0);

      ParticleVariable<double> pmass;
      old_dw->get(pmass, "p.mass", matlindex, region, 0);
      new_dw->put(pmass, "p.mass", matlindex, region, 0);
      ParticleVariable<double> pvolume;
      old_dw->get(pvolume, "p.volume", matlindex, region, 0);
      new_dw->put(pvolume, "p.volume", matlindex, region, 0);
      ParticleVariable<Vector> pexternalforce;
      old_dw->get(pexternalforce, "p.externalforce", matlindex, region, 0);
      new_dw->put(pexternalforce, "p.externalforce", matlindex, region, 0);
    }
  }
#endif
}

} // end namespace Components
} // end namespace Uintah

// $Log$
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

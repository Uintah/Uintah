
#include <Uintah/Components/MPM/SerialMPM.h>

#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Components/MPM/Matrix3.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/ProblemSpec.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Grid/SoleVariable.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Components/MPM/CompMooneyRivlin.h> // TEMPORARY
#include <SCICore/Geometry/Vector.h>
using SCICore::Geometry::Vector;
#include <SCICore/Geometry/Point.h>
using SCICore::Geometry::Point;
#include <SCICore/Math/MinMax.h>
using SCICore::Math::Max;
#include <iostream>
using std::cerr;
using std::string;
#include <fstream>
using std::ofstream;

#include "Problem.h"


SerialMPM::SerialMPM()
{
}

SerialMPM::~SerialMPM()
{
}

void SerialMPM::problemSetup(const ProblemSpecP&, GridP& grid,
			     DataWarehouseP& dw)
{
    string infile("in.mpm");
    Problem Enigma;
    Enigma.preProcessor(infile);
    LevelP level = grid->getLevel(0);
    double bnds[7];
    Enigma.getBnds(bnds);
    Point lower(bnds[1], bnds[3], bnds[5]);
    Point upper(bnds[2], bnds[4], bnds[6]);
    double dx[4];
    Enigma.getDx(dx);
    Vector diag = upper-lower;
    int nx = (int)(diag.x()/dx[1]+0.5);
    int ny = (int)(diag.y()/dx[2]+0.5);
    int nz = (int)(diag.z()/dx[3]+0.5);
    
    level->addRegion(lower, upper, nx, ny, nz);
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
	ParticleVariable<Vector> pexternalforce(psubset);
	dw->put(pexternalforce, "p.externalforce", region, 0);
	ParticleVariable<CompMooneyRivlin> pconmod(psubset);
	dw->put(pconmod, "p.conmod", region, 0);
	cerr << "Creating particles for region\n";
	Enigma.createParticles(region, dw);
    }
    cerr << "SerialMPM::problemSetup not done\n";
}

void SerialMPM::computeStableTimestep(const LevelP& level,
				      SchedulerP& sched, DataWarehouseP& dw)
{
    for(Level::const_regionIterator iter=level->regionsBegin();
	iter != level->regionsEnd(); iter++){
	const Region* region=*iter;

	Task* t = new Task("SerialMPM::computeStableTimestep", region, dw, dw,
			   this, SerialMPM::actuallyComputeStableTimestep);
	t->requires(dw, "velocity", region, 0,
		    ParticleVariable<Vector>::getTypeDescription());
	t->requires(dw, "params", ProblemSpec::getTypeDescription());
	t->computes(dw, "delt", SoleVariable<double>::getTypeDescription());
	sched->addTask(t);
    }
}

void SerialMPM::timeStep(double t, double dt,
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
	    /*
	     * computeStressTensor
	     *   in(G.VELOCITY, P.NAT_X, P.CONMOD)
	     *   operation(evaluate the gradient of G.VELOCITY at P.NAT_X, feed
	     *             this into P.CONMOD, which will evaluate and store
	     *             the stress)
	     * out(P.CONMOD)
	     */
	    Task* t = new Task("SerialMPM::computeStressTensor",
			       region, old_dw, new_dw,
			       this, SerialMPM::computeStressTensor);
	    t->requires(new_dw, "g.velocity", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->requires(old_dw, "p.conmod", region, 0,
			ParticleVariable<CompMooneyRivlin>::getTypeDescription());
	    t->requires(old_dw, "delt",
			SoleVariable<double>::getTypeDescription());
	    t->computes(new_dw, "p.conmod", region, 0,
			ParticleVariable<CompMooneyRivlin>::getTypeDescription());
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
	    t->requires(new_dw, "p.conmod", region, 0,
			ParticleVariable<CompMooneyRivlin>::getTypeDescription());
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

#define HEConstant1 100000.0
#define HEConstant2 20000.0
#define HEConstant3 70000.0
#define HEConstant4 1320000.0

inline double getLambda()
{
    double C1 = HEConstant1;
    double C2 = HEConstant2;
    double C4 = HEConstant4;
  
    double PR = (2.*C1 + 5.*C2 + 2.*C4)/(4.*C4 + 5.*C1 + 11.*C2);
    double mu = 2.*(C1 + C2);
    double lambda = 2.*mu*(1.+PR)/(3.*(1.-2.*PR)) - (2./3.)*mu;

    return lambda;
}

inline double getMu()
{
  double mu = 2.*(HEConstant1 + HEConstant2);

  return mu;
}

void SerialMPM::actuallyComputeStableTimestep(const ProcessorContext*,
					      const Region* region,
					      const DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw)
{
    ParticleVariable<double> pmass;
    new_dw->get(pmass, "p.mass", region, 0);
    ParticleVariable<double> pvolume;
    new_dw->get(pvolume, "p.volume", region, 0);
    ParticleSubset* pset = pmass.getParticleSubset();
    ASSERT(pset == pvolume.getParticleSubset());

    double c_dil = 0;
    double c_rot = 0;
    for(ParticleSubset::iterator iter = pset->begin();
	iter != pset->end(); iter++){
	ParticleSet::index idx = *iter;
	double lambda = getLambda();
	double mu = getMu();
	double density = pmass[idx]/pvolume[idx];
	c_dil = Max(c_dil, sqrt((lambda + 2.*mu)/density));
	c_rot = Max(c_rot, sqrt(mu/density));
    }
    double c = Max(c_rot, c_dil);
    Vector dCell = region->dCell();
    double width = Max(dCell.x(), dCell.y(), dCell.z());
    double delt = 0.5*width/c;
    new_dw->put(SoleVariable<double>(delt), "delt");
}

void SerialMPM::interpolateParticlesToGrid(const ProcessorContext*,
					   const Region* region,
					   const DataWarehouseP& old_dw,
					   DataWarehouseP& new_dw)
{
    // Create arrays for the particle data
    ParticleVariable<Vector> px;
    old_dw->get(px, "p.x", region, 0);
    ParticleVariable<double> pmass;
    old_dw->get(pmass, "p.mass", region, 0);
    ParticleVariable<Vector> pvelocity;
    old_dw->get(pvelocity, "p.velocity", region, 0);
    ParticleVariable<Vector> pexternalforce;
    old_dw->get(pexternalforce, "p.externalforce", region, 0);

    // Create arrays for the grid data
    NCVariable<double> gmass;
    new_dw->allocate(gmass, "g.mass", region, 0);
    NCVariable<Vector> gvelocity;
    new_dw->allocate(gvelocity, "g.velocity", region, 0);
    NCVariable<Vector> externalforce;
    new_dw->allocate(externalforce, "g.externalforce", region, 0);

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
	ParticleSet::index idx = *iter;

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

    new_dw->put(gmass, "g.mass", region, 0);
    new_dw->put(gvelocity, "g.velocity", region, 0);
    new_dw->put(externalforce, "g.externalforce", region, 0);
}

void SerialMPM::computeStressTensor(const ProcessorContext*,
				    const Region* region,
				    const DataWarehouseP& old_dw,
				    DataWarehouseP& new_dw)
{
    Matrix3 velGrad;
    Vector dx = region->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();

    // Create arrays for the particle position
    // and the constitutive model
    ParticleVariable<Vector> px;
    old_dw->get(px, "p.x", region, 0);
    ParticleVariable<CompMooneyRivlin> pconmod;
    old_dw->get(pconmod, "p.conmod", region, 0);

    NCVariable<Vector> gvelocity;
    new_dw->get(gvelocity, "g.velocity", region, 0);
    SoleVariable<double> delt;
    old_dw->get(delt, "delt");

    ParticleSubset* pset = px.getParticleSubset();
    ASSERT(pset == px.getParticleSubset());
    ASSERT(pset == pconmod.getParticleSubset());

    for(ParticleSubset::iterator iter = pset->begin();
       iter != pset->end(); iter++){
       ParticleSet::index idx = *iter;

       velGrad.set(0.0);
       // Get the node indices that surround the cell
       Array3Index ni[8];
       Vector d_S[8];
       if(!region->findCellAndShapeDerivatives(px[idx], ni, d_S))
	   continue;

        for(int k = 0; k < 8; k++) {
         // While this reflects the smpm version, I think it is
         // slightly wrong, but should give the same answer. JG 1/24/00
	    //	    if(region->contains(ni[k])){
	    Vector& gvel = gvelocity[ni[k]];
		for (int j = 0; j<3; j++){
		    for (int i = 0; i<3; i++) {
			velGrad(i+1,j+1)+=gvel(j) * d_S[k](i) * oodx[i];
		    }
		}
		//}
        }

	// Compute the stress tensor at each particle location.
        pconmod[idx].computeStressTensor(velGrad,delt);

    }

    new_dw->put(pconmod, "p.conmod", region, 0);

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

    // Create arrays for the particle position, volume
    // and the constitutive model
    ParticleVariable<Vector> px;
    old_dw->get(px, "p.x", region, 0);
    ParticleVariable<double> pvol;
    old_dw->get(pvol, "p.volume", region, 0);
    ParticleVariable<CompMooneyRivlin> pconmod;
    old_dw->get(pconmod, "p.conmod", region, 0);

    NCVariable<Vector> internalforce;
    new_dw->allocate(internalforce, "g.internalforce", region, 0);

    ParticleSubset* pset = px.getParticleSubset();
    ASSERT(pset == px.getParticleSubset());
    ASSERT(pset == pvol.getParticleSubset());
    ASSERT(pset == pconmod.getParticleSubset());

    internalforce.initialize(Vector(0,0,0));

    for(ParticleSubset::iterator iter = pset->begin();
       iter != pset->end(); iter++){
       ParticleSet::index idx = *iter;

       // Get the node indices that surround the cell
       Array3Index ni[8];
       Vector d_S[8];
       if(!region->findCellAndShapeDerivatives(px[idx], ni, d_S))
	   continue;

       for (int k = 0; k < 8; k++){
	   Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],d_S[k].z()*oodx[2]);
	   //if(region->contains(ni[k])){
	       internalforce[ni[k]] -=
		   (div * pconmod[idx].getStressTensor() * pvol[idx]);
	       //}
       }
    }

    new_dw->put(internalforce, "g.internalforce", region, 0);
}

void SerialMPM::solveEquationsMotion(const ProcessorContext*,
				     const Region* region,
				     const DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw)
{
    Vector zero(0.,0.,0.);

    // Get required variables for this region
    NCVariable<double> mass;
    new_dw->get(mass, "g.mass", region, 0);
    NCVariable<Vector> internalforce;
    new_dw->get(internalforce, "g.internalforce", region, 0);
    NCVariable<Vector> externalforce;
    new_dw->get(externalforce, "g.externalforce", region, 0);

    // Create variables for the results
    NCVariable<Vector> acceleration;
    new_dw->allocate(acceleration, "g.acceleration", region, 0);

    // Do the computation of a = F/m for nodes where m!=0.0
    for(NodeIterator iter = region->begin();
        iter != region->end(); iter++){
	if(mass[*iter]>0.0){
	  acceleration[*iter] = (internalforce[*iter] + externalforce[*iter])/
						mass[*iter];
	}
	else{
	  acceleration[*iter] = zero;
	}
    }

    // Put the result in the datawarehouse
    new_dw->put(acceleration, "g.acceleration", region, 0);

}

void SerialMPM::integrateAcceleration(const ProcessorContext*,
				      const Region* region,
				      const DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw)
{
    // Get required variables for this region
    NCVariable<Vector> acceleration;
    new_dw->get(acceleration, "g.acceleration", region, 0);
    NCVariable<Vector> velocity;
    new_dw->get(velocity, "g.velocity", region, 0);
    SoleVariable<double> delt;
    old_dw->get(delt, "delt");

    // Create variables for the results
    NCVariable<Vector> velocity_star;
    new_dw->allocate(velocity_star, "g.velocity_star", region, 0);

    // Do the computation

    for(NodeIterator iter = region->begin();
	iter != region->end(); iter++)
	velocity_star[*iter] = velocity[*iter] + acceleration[*iter] * delt;

    // Put the result in the datawarehouse
    new_dw->put(velocity_star, "g.velocity_star", region, 0);
}

void SerialMPM::interpolateToParticlesAndUpdate(const ProcessorContext*,
						const Region* region,
						const DataWarehouseP& old_dw,
						DataWarehouseP& new_dw)
{
    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively

    // Get the arrays of particle values to be changed
    ParticleVariable<Vector> px;
    old_dw->get(px, "p.x", region, 0);
    ParticleVariable<Vector> pvelocity;
    old_dw->get(pvelocity, "p.velocity", region, 0);

    // Get the arrays of grid data on which the new particle values depend
    NCVariable<Vector> gvelocity_star;
    new_dw->get(gvelocity_star, "g.velocity_star", region, 0);
    NCVariable<Vector> gacceleration;
    new_dw->get(gacceleration, "g.acceleration", region, 0);
    SoleVariable<double> delt;
    old_dw->get(delt, "delt");

    ParticleSubset* pset = px.getParticleSubset();
    ASSERT(pset == pvelocity.getParticleSubset());

    Vector vel(0.0,0.0,0.0);
    Vector acc(0.0,0.0,0.0);

    double ke=0;
    for(ParticleSubset::iterator iter = pset->begin();
        iter != pset->end(); iter++){
      ParticleSet::index idx = *iter;

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
    new_dw->put(px, "p.x", region, 0);
    new_dw->put(pvelocity, "p.velocity", region, 0);

    ParticleVariable<double> pmass;
    old_dw->get(pmass, "p.mass", region, 0);
    new_dw->put(pmass, "p.mass", region, 0);
    ParticleVariable<double> pvolume;
    old_dw->get(pvolume, "p.volume", region, 0);
    new_dw->put(pvolume, "p.volume", region, 0);
    ParticleVariable<Vector> pexternalforce;
    old_dw->get(pexternalforce, "p.externalforce", region, 0);
    new_dw->put(pexternalforce, "p.externalforce", region, 0);
}

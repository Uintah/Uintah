/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/MPM/SerialMPM.h>
#include <Uintah/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Components/MPM/Contact/ContactFactory.h>
#include <Uintah/Components/MPM/Fracture/FractureFactory.h>
#include <Uintah/Components/MPM/Fracture/Fracture.h>
#include <Uintah/Components/MPM/ThermalContact/ThermalContact.h>
#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/NodeIterator.h> // Must be included after Patch.h
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SoleVariable.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Exceptions/ParameterNotFound.h>

#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/MinMax.h>

#include <iostream>
#include <fstream>

#include "GeometrySpecification/Problem.h"
#include <Uintah/Components/MPM/MPMLabel.h>

using namespace Uintah;
using namespace Uintah::MPM;

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Geometry::Dot;
using SCICore::Math::Min;
using SCICore::Math::Max;
using namespace std;


SerialMPM::SerialMPM( int MpiRank, int MpiProcesses ) :
  UintahParallelComponent( MpiRank, MpiProcesses )
{
   d_heatConductionInvolved = false;
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


   if (d_heatConductionInvolved) {
     d_thermalContactModel = new ThermalContact;
   }
  
   cerr << "SerialMPM::problemSetup not done\n";
}

void SerialMPM::scheduleInitialize(const LevelP& level,
				   SchedulerP& sched,
				   DataWarehouseP& dw)
{
   Level::const_patchIterator iter;

   for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++){

      const Patch* patch=*iter;
      {
	 Task* t = scinew Task("SerialMPM::actuallyInitialize", patch, dw, dw,
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
   int fieldIndependentVariable = 0;

   int numMatls = d_sharedState->getNumMatls();

   const MPMLabel* lb = MPMLabel::getLabels();

   for(Level::const_patchIterator iter=level->patchesBegin();
       iter != level->patchesEnd(); iter++){

      const Patch* patch=*iter;

      if(d_fractureModel) {
	 /*
	  * labelSelfContactNodesAndCells
	  *   in(C.SURFACENORMAL,P.SURFACENORMAL)
	  *   operation(label the nodes and cells that has self-contact)
	  *   out(C.SELFCONTACTLABEL)
	  */
	 Task* t = scinew Task("Fracture::labelSelfContactNodesAndCells",
			    patch, old_dw, new_dw,
			    d_fractureModel,
			    &Fracture::labelSelfContactNodesAndCells);

	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( old_dw, lb->pSurfaceNormalLabel, idx, patch,
			 Ghost::None);
	 }

         t->requires( old_dw, lb->cSurfaceNormalLabel,
                      d_sharedState->getMaterial(fieldIndependentVariable)
                                   ->getDWIndex(), patch, Ghost::None);

         t->computes( new_dw, lb->cSelfContactLabel,
                      d_sharedState->getMaterial(fieldIndependentVariable)
                                   ->getDWIndex(), patch );

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
	 Task* t = scinew Task("SerialMPM::interpolateParticlesToGrid",
			    patch, old_dw, new_dw,
			    this,&SerialMPM::interpolateParticlesToGrid);
	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires(old_dw, lb->pMassLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, lb->pVelocityLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, lb->pExternalForceLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, lb->pXLabel, idx, patch,
			Ghost::AroundNodes, 1 );

	    t->computes(new_dw, lb->gMassLabel, idx, patch );
	    t->computes(new_dw, lb->gVelocityLabel, idx, patch );
	    t->computes(new_dw, lb->gExternalForceLabel, idx, patch );

            if (d_heatConductionInvolved) {
              t->requires(old_dw, lb->pTemperatureLabel, idx, patch,
			Ghost::AroundNodes, 1 );
              t->computes(new_dw, lb->gTemperatureLabel, idx, patch );
            }
	 }
		     
	 sched->addTask(t);
      }

      if (d_heatConductionInvolved) {
	 /* computeHeatExchange
	  *   in(G.MASS, G.TEMPERATURE, G.EXTERNAL_HEAT_RATE)
	  *   operation(peform heat exchange which will cause each of
	  *		velocity fields to exchange heat according to 
	  *             the temperature differences)
	  *   out(G.EXTERNAL_HEAT_RATE)
	  */

	 Task* t = scinew Task("ThermalContact::computeHeatExchange",
			    patch, old_dw, new_dw,
			    d_thermalContactModel,
			    &ThermalContact::computeHeatExchange);

	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
	    if(mpm_matl){
               d_thermalContactModel->addComputesAndRequires(
		 t, mpm_matl, patch, old_dw, new_dw);
	    }

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

	 Task* t = scinew Task("Contact::exMomInterpolated",
			    patch, old_dw, new_dw,
			    d_contactModel,
			    &Contact::exMomInterpolated);

	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
	    if(mpm_matl){
               d_contactModel->addComputesAndRequiresInterpolated(
					t, mpm_matl, patch, old_dw, new_dw);
	    }

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
	  *             patch.  This is used in calculating delt.)
	  * out(P.DEFORMATIONMEASURE,P.STRESS)
	  */
	 Task* t = scinew Task("SerialMPM::computeStressTensor",
			    patch, old_dw, new_dw,
			    this, &SerialMPM::computeStressTensor);
	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
	    if(mpm_matl){
	       ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
	       cm->addComputesAndRequires(t, mpm_matl, patch, old_dw, new_dw);
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
	 Task* t = scinew Task("Fracture::updateSurfaceNormalOfBoundaryParticle",
			    patch, old_dw, new_dw,
			    d_fractureModel,
			    &Fracture::updateSurfaceNormalOfBoundaryParticle);

	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( old_dw, lb->pDeformationMeasureLabel, idx, patch,
			 Ghost::None);
	    t->requires( old_dw, lb->pSurfaceNormalLabel, idx, patch,
			 Ghost::None);

	    t->computes( new_dw, lb->pSurfaceNormalLabel, idx, patch );
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
	 Task* t = scinew Task("SerialMPM::computeInternalForce",
			    patch, old_dw, new_dw,
			    this, &SerialMPM::computeInternalForce);
	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( new_dw, lb->pStressLabel, idx, patch,
			 Ghost::AroundNodes, 1);
	    t->requires( new_dw, lb->pVolumeLabel, idx, patch,
			 Ghost::AroundNodes, 1);

	    t->computes( new_dw, lb->gInternalForceLabel, idx, patch );
	 }

	 sched->addTask( t );
      }
      
      if (d_heatConductionInvolved) {
	 /*
	  * computeInternalHeatRate
	  *   in(P.X, P.VOLUME, P.TEMPERATUREGRADIENT)
	  *   operation(evaluate the grid internal heat rate using 
	  *   P.TEMPERATUREGRADIENT and the gradients of the
	  *   shape functions)
	  * out(G.INTERNALHEATRATE)
	  */

	 Task* t = new Task("SerialMPM::computeInternalHeatRate",
			    patch, old_dw, new_dw,
			    this, &SerialMPM::computeInternalHeatRate);

	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();

	    t->requires(old_dw, lb->pXLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires(old_dw, lb->pVolumeLabel, idx, patch,
			Ghost::AroundNodes, 1 );
	    t->requires( new_dw, lb->pTemperatureGradientLabel, idx, patch,
			 Ghost::AroundNodes, 1);

	    t->computes( new_dw, lb->gInternalHeatRateLabel, idx, patch );
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
	 Task* t = scinew Task("SerialMPM::solveEquationsMotion",
			    patch, old_dw, new_dw,
			    this, &SerialMPM::solveEquationsMotion);
	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( new_dw, lb->gMassLabel, idx, patch,
			 Ghost::None);
	    t->requires( new_dw, lb->gInternalForceLabel, idx, patch,
			 Ghost::None);

	    t->computes( new_dw, lb->gAccelerationLabel, idx, patch);
	 }

	 sched->addTask(t);
      }

      if (d_heatConductionInvolved) {
	 /*
	  * solveHeatEquations
	  *   in(G.MASS, G.INTERNALHEATRATE, G.EXTERNALHEATRATE)
	  *   out(G.TEMPERATURERATE)
	  * 
	  */
	 Task* t = new Task("SerialMPM::solveHeatEquations",
			    patch, old_dw, new_dw,
			    this, &SerialMPM::solveHeatEquations);
	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( new_dw, lb->gMassLabel, idx, patch,
			 Ghost::None);
	    t->requires( new_dw, lb->gInternalHeatRateLabel, idx, patch,
			 Ghost::None);
	    t->requires( new_dw, lb->gExternalHeatRateLabel, idx, patch,
			 Ghost::None);

	    t->computes( new_dw, lb->gTemperatureRateLabel, idx, patch);
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
	 Task* t = scinew Task("SerialMPM::integrateAcceleration",
			    patch, old_dw, new_dw,
			    this, &SerialMPM::integrateAcceleration);
	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires(new_dw, lb->gAccelerationLabel, idx, patch,
			Ghost::None);
	    t->requires(new_dw, lb->gMomExedVelocityLabel, idx, patch,
			Ghost::None);
	    t->requires(old_dw, d_sharedState->get_delt_label() );
		     
	    t->computes(new_dw, lb->gVelocityStarLabel, idx, patch );
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

	Task* t = scinew Task("Contact::exMomIntegrated",
			   patch, old_dw, new_dw,
			   d_contactModel, &Contact::exMomIntegrated);
	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
	    int idx = matl->getDWIndex();
            d_contactModel->addComputesAndRequiresIntegrated(
				t, mpm_matl, patch, old_dw, new_dw);
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
	 Task* t = scinew Task("Fracture::updateNodeInformationInContactCells",
			    patch, old_dw, new_dw,
			    d_fractureModel,
			    &Fracture::updateNodeInformationInContactCells);

	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( old_dw, lb->gVelocityLabel, idx, patch,
			 Ghost::None);
	    t->requires( old_dw, lb->gAccelerationLabel, idx, patch,
			 Ghost::None);

	    t->computes( new_dw, lb->gVelocityLabel, idx, patch );
	    t->computes( new_dw, lb->gAccelerationLabel, idx, patch );
	 }

         t->requires( old_dw,
                      lb->gSelfContactLabel,
                      d_sharedState->getMaterial(fieldIndependentVariable)
                                   ->getDWIndex(),
                      patch,
	              Ghost::None);

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
	 Task* t = scinew Task("SerialMPM::interpolateToParticlesAndUpdate",
			    patch, old_dw, new_dw,
			    this, &SerialMPM::interpolateToParticlesAndUpdate);
	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires(new_dw, lb->gMomExedAccelerationLabel, idx, patch,
			Ghost::AroundCells, 1);
	    t->requires(new_dw, lb->gMomExedVelocityStarLabel, idx, patch,
			Ghost::AroundCells, 1);
	    t->requires(old_dw, lb->pXLabel, idx, patch,
			Ghost::None);
			
	    t->requires(old_dw, lb->pMassLabel, idx, patch, Ghost::None);
	    t->requires(old_dw, lb->pExternalForceLabel, idx, patch, Ghost::None);
	    t->requires(old_dw, d_sharedState->get_delt_label() );
	    t->computes(new_dw, lb->pVelocityLabel, idx, patch );
	    t->computes(new_dw, lb->pXLabel, idx, patch );
	    t->computes(new_dw, lb->pMassLabel, idx, patch);
	    t->computes(new_dw, lb->pExternalForceLabel, idx, patch);
	    t->computes(new_dw, lb->KineticEnergyLabel);

	    if(d_heatConductionInvolved) {
	      t->requires(new_dw, lb->gTemperatureLabel, idx, patch,
			Ghost::None);
              t->computes(new_dw, lb->pTemperatureRateLabel, idx, patch);
              t->computes(new_dw, lb->pTemperatureLabel, idx, patch);
              t->computes(new_dw, lb->pTemperatureGradientLabel, idx, patch);
	    }
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
	 Task* t = scinew Task("Fracture::updateParticleInformationInContactCells",
			    patch, old_dw, new_dw,
			    d_fractureModel,
			    &Fracture::updateParticleInformationInContactCells);

	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( old_dw, lb->pXLabel, idx, patch,
			 Ghost::None);
	    t->requires( old_dw, lb->pVelocityLabel, idx, patch,
			 Ghost::None);
	    t->requires( old_dw, lb->pExternalForceLabel, idx, patch,
			 Ghost::None);
	    t->requires( old_dw, lb->pDeformationMeasureLabel, idx, patch,
			 Ghost::None);
	    t->requires( old_dw, lb->pStressLabel, idx, patch,
			 Ghost::None);

	    t->computes( new_dw, lb->pXLabel, idx, patch );
	    t->computes( new_dw, lb->pVelocityLabel, idx, patch );
	    t->computes( new_dw, lb->pDeformationMeasureLabel, idx, patch );
	    t->computes( new_dw, lb->pStressLabel, idx, patch );
	 }

         t->requires( old_dw,
                      lb->cSelfContactLabel,
                      d_sharedState->getMaterial(fieldIndependentVariable)
                                   ->getDWIndex(),
                      patch,
	              Ghost::None);

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
	 Task* t = scinew Task("Fracture::crackGrow",
			    patch, old_dw, new_dw,
			    d_fractureModel,
			    &Fracture::crackGrow);

	 for(int m = 0; m < numMatls; m++){
	    Material* matl = d_sharedState->getMaterial(m);
	    int idx = matl->getDWIndex();
	    t->requires( new_dw, lb->pStressLabel, idx, patch,
			 Ghost::None);
	    t->requires( new_dw, lb->pXLabel, idx, patch,
			 Ghost::None);
	    t->requires( new_dw, lb->pSurfaceNormalLabel, idx, patch,
			 Ghost::None);

	    t->computes( new_dw, lb->pSurfaceNormalLabel, idx, patch );
	 }

	 sched->addTask(t);
      }

    }

   if(d_fractureModel) {
      new_dw->pleaseSave(lb->pDeformationMeasureLabel, numMatls);
   }
   new_dw->pleaseSave(lb->pVolumeLabel, numMatls);
   new_dw->pleaseSave(lb->pExternalForceLabel, numMatls);
   new_dw->pleaseSave(lb->gVelocityLabel, numMatls);
   new_dw->pleaseSave(lb->pXLabel, numMatls);
   new_dw->pleaseSaveIntegrated(lb->StrainEnergyLabel);
   new_dw->pleaseSaveIntegrated(lb->KineticEnergyLabel);
}

void SerialMPM::actuallyInitialize(const ProcessorContext*,
				   const Patch* patch,
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
       particleIndex numParticles = mpm_matl->countParticles(patch);

       mpm_matl->createParticles(numParticles, patch, new_dw);
       mpm_matl->getConstitutiveModel()->initializeCMData(patch,
						mpm_matl, new_dw);
       int vfindex = matl->getVFIndex();

       d_contactModel->initializeContact(patch,vfindex,new_dw);
       
       if(d_fractureModel) {
	 d_fractureModel->initializeFracture( patch, new_dw );
       }
    }
  }
}


void SerialMPM::actuallyComputeStableTimestep(const ProcessorContext*,
					      const Patch*,
					      DataWarehouseP&,
					      DataWarehouseP&)
{
}

void SerialMPM::interpolateParticlesToGrid(const ProcessorContext*,
					   const Patch* patch,
					   DataWarehouseP& old_dw,
					   DataWarehouseP& new_dw)
{
  // This needs the datawarehouse to allow indexing by material
  // for the particle data and velocity field by the grid data

  // Dd: making this compile... don't know if this correct...
  int numMatls = d_sharedState->getNumMatls();
  const MPMLabel* lb = MPMLabel::getLabels();

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
      ParticleVariable<double> pTemperature;

      old_dw->get(px,             lb->pXLabel, matlindex, patch,
		  Ghost::AroundNodes, 1);
      old_dw->get(pmass,          lb->pMassLabel, matlindex, patch,
		  Ghost::AroundNodes, 1);
      old_dw->get(pvelocity,      lb->pVelocityLabel, matlindex, patch,
		  Ghost::AroundNodes, 1);
      old_dw->get(pexternalforce, lb->pExternalForceLabel, matlindex, patch,
		  Ghost::AroundNodes, 1);

      // Create arrays for the grid data
      NCVariable<double> gmass;
      NCVariable<Vector> gvelocity;
      NCVariable<Vector> externalforce;
      NCVariable<double> gTemperature;

//      std::cerr << "allocating grid variables" << std::endl;
      new_dw->allocate(gmass,         lb->gMassLabel, vfindex, patch);
      new_dw->allocate(gvelocity,     lb->gVelocityLabel, vfindex, patch);
      new_dw->allocate(externalforce, lb->gExternalForceLabel, vfindex, patch);

      if (d_heatConductionInvolved) {
        ParticleVariable<double> pTemperature;
        NCVariable<double> gTemperature;
        old_dw->get(pTemperature, lb->pTemperatureLabel, matlindex, patch,
		  Ghost::AroundNodes, 1);
        new_dw->allocate(gTemperature, lb->gTemperatureLabel, vfindex, patch);
      }

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
	 if(!patch->findCellAndWeights(px[idx], ni, S))
	    continue;
	 // Add each particles contribution to the local mass & velocity 
	 // Must use the node indices
	 for(int k = 0; k < 8; k++) {
	    if(patch->containsNode(ni[k])){
	       gmass[ni[k]] += pmass[idx] * S[k];
	       gvelocity[ni[k]] += pvelocity[idx] * pmass[idx] * S[k];
	       externalforce[ni[k]] += pexternalforce[idx] * S[k];
	       
  	       if (d_heatConductionInvolved) {
    	         gTemperature[ni[k]] += pTemperature[idx] * pmass[idx] * S[k];
  	       }
	    }
	 }
      }

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	 if(gmass[*iter] != 0.0){
	    gvelocity[*iter] *= 1./gmass[*iter];
	    if (d_heatConductionInvolved) {
    	      gTemperature[*iter] /= gmass[*iter];
	    }
	 }
      }
#if 0

      // Apply grid boundary conditions to the velocity
      // before storing the data
      for(int face = 0; face<6; face++){
	Patch::FaceType f=(Patch::FaceType)face;

#if 0
	switch(patch->getBCType(f)){
	case Patch::None:
	     // Do nothing
	     break;
	case Patch::Fixed:
	     gvelocity.fillFace(f,Vector(0.0,0.0,0.0));
	     break; 
	case Patch::Symmetry:
	     gvelocity.fillFaceNormal(f);
	     break; 
	case Patch::Neighbor:
	     // Do nothing
	     break;
	}
#endif
	gvelocity.fillFace(f,Vector(0.0,0.0,0.0));
      }
#endif

      new_dw->put(gmass,         lb->gMassLabel, vfindex, patch);
      new_dw->put(gvelocity,     lb->gVelocityLabel, vfindex, patch);
      new_dw->put(externalforce, lb->gExternalForceLabel, vfindex, patch);
      if (d_heatConductionInvolved) {
        new_dw->put(gTemperature, lb->gTemperatureLabel, vfindex, patch);
      }
    }
  }
}

void SerialMPM::computeStressTensor(const ProcessorContext*,
				    const Patch* patch,
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
	 cm->computeStressTensor(patch, mpm_matl, old_dw, new_dw);
      }
   }
}

void SerialMPM::updateSurfaceNormalOfBoundaryParticle(const ProcessorContext*,
				    const Patch* /*patch*/,
				    DataWarehouseP& /*old_dw*/,
				    DataWarehouseP& /*new_dw*/)
{
  //Tan: not finished yet. 
}

void SerialMPM::computeInternalForce(const ProcessorContext*,
				     const Patch* patch,
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw)
{

  Vector dx = patch->dCell();
  double oodx[3];
  oodx[0] = 1.0/dx.x();
  oodx[1] = 1.0/dx.y();
  oodx[2] = 1.0/dx.z();

  // This needs the datawarehouse to allow indexing by material
  // for the particle data and velocity field for the grid data.

  int numMatls = d_sharedState->getNumMatls();

  const MPMLabel* lb = MPMLabel::getLabels();

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

      old_dw->get(px,      lb->pXLabel, matlindex, patch,
		  Ghost::AroundNodes, 1);
      new_dw->get(pvol,    lb->pVolumeLabel, matlindex, patch,
		  Ghost::AroundNodes, 1);
      new_dw->get(pstress, lb->pStressLabel, matlindex, patch,
		  Ghost::AroundNodes, 1);

      new_dw->allocate(internalforce, lb->gInternalForceLabel, vfindex, patch);
  
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
         if(!patch->findCellAndShapeDerivatives(px[idx], ni, d_S))
  	   continue;

         for (int k = 0; k < 8; k++){
	  if(patch->containsNode(ni[k])){
	   Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],d_S[k].z()*oodx[2]);
	   internalforce[ni[k]] -= (div * pstress[idx] * pvol[idx]);
	  }
         }
      }
      new_dw->put(internalforce, lb->gInternalForceLabel, vfindex, patch);
    }
  }
}


void SerialMPM::computeInternalHeatRate(
                                     const ProcessorContext*,
				     const Patch* patch,
				     DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw)
{

  Vector dx = patch->dCell();
  double oodx[3];
  oodx[0] = 1.0/dx.x();
  oodx[1] = 1.0/dx.y();
  oodx[2] = 1.0/dx.z();

  int numMatls = d_sharedState->getNumMatls();
  const MPMLabel* lb = MPMLabel::getLabels();

  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      double thermalConductivity = mpm_matl->getThermalConductivity();
      int matlindex = matl->getDWIndex();
      int vfindex = matl->getVFIndex();
      ParticleVariable<Point>  px;
      ParticleVariable<double> pvol;
      ParticleVariable<Vector> pTemperatureGradient;
      NCVariable<double>       internalHeatRate;

      old_dw->get(px,      lb->pXLabel, matlindex, patch,
		  Ghost::AroundNodes, 1);
      old_dw->get(pvol,    lb->pVolumeLabel, matlindex, patch,
		  Ghost::AroundNodes, 1);
      old_dw->get(pTemperatureGradient, lb->pTemperatureGradientLabel, matlindex, patch,
		  Ghost::AroundNodes, 1);

      new_dw->allocate(internalHeatRate, lb->gInternalHeatRateLabel, vfindex, patch);
  
      ParticleSubset* pset = px.getParticleSubset();

      ASSERT(pset->numParticles() == px.getParticleSubset()->numParticles());
      ASSERT(pset->numParticles() == pvol.getParticleSubset()->numParticles());
      ASSERT(pset->numParticles() == pTemperatureGradient.getParticleSubset()->numParticles());

      internalHeatRate.initialize(0.);

      for(ParticleSubset::iterator iter = pset->begin();
         iter != pset->end(); iter++){
         particleIndex idx = *iter;
  
         // Get the node indices that surround the cell
         IntVector ni[8];
         Vector d_S[8];
         if(!patch->findCellAndShapeDerivatives(px[idx], ni, d_S))
  	   continue;

         for (int k = 0; k < 8; k++){
	  if(patch->containsNode(ni[k])){
           Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],d_S[k].z()*oodx[2]);
	   internalHeatRate[ni[k]] -= Dot( div, pTemperatureGradient[idx] ) * 
	                              pvol[idx] * thermalConductivity;
	  }
         }
      }
      new_dw->put(internalHeatRate, lb->gInternalHeatRateLabel, vfindex, patch);
    }
  }
}


void SerialMPM::solveEquationsMotion(const ProcessorContext*,
				     const Patch* patch,
				     DataWarehouseP& /*old_dw*/,
				     DataWarehouseP& new_dw)
{
  Vector zero(0.,0.,0.);

  // This needs the datawarehouse to allow indexing by velocity
  // field for the grid data

  int numMatls = d_sharedState->getNumMatls();

  const MPMLabel* lb = MPMLabel::getLabels();

  // Gravity
  Vector gravity = d_sharedState->getGravity();

  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfindex = matl->getVFIndex();
      // Get required variables for this patch
      NCVariable<double> mass;
      NCVariable<Vector> internalforce;
      NCVariable<Vector> externalforce;

      new_dw->get(mass,          lb->gMassLabel, vfindex, patch, Ghost::None, 0);
      new_dw->get(internalforce, lb->gInternalForceLabel, vfindex, patch,
		  Ghost::None, 0);
      new_dw->get(externalforce, lb->gExternalForceLabel, vfindex, patch,
		  Ghost::None, 0);

      // Create variables for the results
      NCVariable<Vector> acceleration;
      new_dw->allocate(acceleration, lb->gAccelerationLabel, vfindex, patch);

      // Do the computation of a = F/m for nodes where m!=0.0
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
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

      new_dw->put(acceleration, lb->gAccelerationLabel, vfindex, patch);

    }
  }
}


void SerialMPM::solveHeatEquations(const ProcessorContext*,
				     const Patch* patch,
				     DataWarehouseP& /*old_dw*/,
				     DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMatls();

  const MPMLabel* lb = MPMLabel::getLabels();

  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfindex = matl->getVFIndex();
      
      double specificHeat = mpm_matl->getSpecificHeat();
      
      // Get required variables for this patch
      NCVariable<double> mass,internalHeatRate,externalHeatRate;

      new_dw->get(mass, lb->gMassLabel, vfindex, patch, Ghost::None, 0);
      new_dw->get(internalHeatRate, lb->gInternalHeatRateLabel, 
                  vfindex, patch, Ghost::None, 0);
      new_dw->get(externalHeatRate, lb->gExternalHeatRateLabel, 
                  vfindex, patch, Ghost::None, 0);

      // Create variables for the results
      NCVariable<double> temperatureRate;
      new_dw->allocate(temperatureRate, lb->gTemperatureRateLabel, vfindex, patch);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	if(mass[*iter]>0.0){
	  temperatureRate[*iter] =
		 (internalHeatRate[*iter] + externalHeatRate[*iter])
		 /mass[*iter] /specificHeat;
	}
	else{
	  temperatureRate[*iter] = 0;
	}
      }

      // Put the result in the datawarehouse
      new_dw->put(temperatureRate, lb->gTemperatureRateLabel, vfindex, patch);

    }
  }
}


void SerialMPM::integrateAcceleration(const ProcessorContext*,
				      const Patch* patch,
				      DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw)
{
  // This needs the datawarehouse to allow indexing by material

  int numMatls = d_sharedState->getNumMatls();

  const MPMLabel* lb = MPMLabel::getLabels();

  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int vfindex = matl->getVFIndex();
      // Get required variables for this patch
      NCVariable<Vector>        acceleration;
      NCVariable<Vector>        velocity;
      delt_vartype delT;
      sum_vartype strainEnergy;
      new_dw->get(strainEnergy, lb->StrainEnergyLabel);

      cout << strainEnergy << endl;

      new_dw->get(acceleration, lb->gAccelerationLabel, vfindex, patch,
		  Ghost::None, 0);
      new_dw->get(velocity, lb->gMomExedVelocityLabel, vfindex, patch,
		  Ghost::None, 0);

      old_dw->get(delT, lb->delTLabel);

      // Create variables for the results
      NCVariable<Vector> velocity_star;
      new_dw->allocate(velocity_star, lb->gVelocityStarLabel, vfindex, patch);

      // Do the computation

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	velocity_star[*iter] = velocity[*iter] + acceleration[*iter] * delT;
      }


      // Put the result in the datawarehouse
      new_dw->put( velocity_star, lb->gVelocityStarLabel, vfindex, patch );
    }
  }
}

void SerialMPM::interpolateToParticlesAndUpdate(const ProcessorContext*,
						const Patch* patch,
						DataWarehouseP& old_dw,
						DataWarehouseP& new_dw)
{
  // Performs the interpolation from the cell vertices of the grid
  // acceleration and velocity to the particles to update their
  // velocity and position respectively

  Vector dx = patch->dCell();
  double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};
  /* tan: oodx used for shape-function-gradient calculation.
          shape-function-gradient will be used for temperature 
          gradient calculation.  */

  Vector vel(0.0,0.0,0.0);
  Vector acc(0.0,0.0,0.0);
  
  double tempRate = 0; /* tan: tempRate stands for "temperature variation
                               time rate", used for heat conduction.  */
  double ke=0;
  int numPTotal = 0;

  // This needs the datawarehouse to allow indexing by material

  int numMatls = d_sharedState->getNumMatls();

  const MPMLabel* lb = MPMLabel::getLabels();

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
      
      ParticleVariable<double> pTemperature; //for heat conduction
      ParticleVariable<Vector> pTemperatureGradient; //for heat conduction
      ParticleVariable<double> pTemperatureRate; //for heat conduction
      NCVariable<double> gTemperatureRate; //for heat conduction
      NCVariable<double> gTemperature; //for heat conduction

      old_dw->get(px,        lb->pXLabel, matlindex, patch, Ghost::None, 0);
      old_dw->get(pvelocity, lb->pVelocityLabel, matlindex, patch, Ghost::None,0);
      old_dw->get(pmass,     lb->pMassLabel, matlindex, patch, Ghost::None, 0);

      // Get the arrays of grid data on which the new particle values depend
      NCVariable<Vector> gvelocity_star;
      NCVariable<Vector> gacceleration;
      delt_vartype delT;

      new_dw->get(gvelocity_star,lb->gMomExedVelocityStarLabel, vfindex, patch,
		  Ghost::AroundCells, 1);
      new_dw->get(gacceleration, lb->gMomExedAccelerationLabel, vfindex, patch,
		  Ghost::AroundCells, 1);
		  
      if(d_heatConductionInvolved) {
        old_dw->get(pTemperature, lb->pTemperatureLabel, matlindex, patch, 
                  Ghost::None, 0);
        new_dw->allocate(pTemperatureRate, lb->pTemperatureRateLabel, vfindex, patch);
        new_dw->allocate(pTemperatureGradient, lb->pTemperatureGradientLabel, vfindex, patch);
        new_dw->get(gTemperatureRate, lb->gTemperatureRateLabel, vfindex, patch,
		  Ghost::AroundCells, 1);
      }

#if 0
      // Apply grid boundary conditions to the velocity_star and
      // acceleration before interpolating back to the particles
      for(int face = 0; face<6; face++){
	Patch::FaceType f=(Patch::FaceType)face;
#if 0
	// Dummy holder until this is resolved
	Patch::FaceType f = Patch::xplus;
	Patch::BCType bctype = patch->getBCType(f);
	switch(bctype){
	  case Patch::None:
	     // Do nothing
	     break;
	  case Patch::Fixed:
	     gvelocity_star.fillFace(f,Vector(0.0,0.0,0.0));
	     gacceleration.fillFace(f,Vector(0.0,0.0,0.0));
	     break;
	  case Patch::Symmetry:
	     gvelocity_star.fillFaceNormal(f);
	     gacceleration.fillFaceNormal(f);
	     break;
	  case Patch::Neighbor:
	     // Do nothing
	     break;
	}
#endif
	gvelocity_star.fillFace(f,Vector(0.0,0.0,0.0));
	gacceleration.fillFace(f,Vector(0.0,0.0,0.0));
      }
#endif

      old_dw->get(delT, lb->delTLabel);

      ParticleSubset* pset = px.getParticleSubset();
      ASSERT(pset == pvelocity.getParticleSubset());

      numPTotal += pset->numParticles();

      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
	 particleIndex idx = *iter;

        // Get the node indices that surround the cell
	IntVector ni[8];
        double S[8];
        Vector d_S[8];

        if(!patch->findCellAndWeights(px[idx], ni, S))
	  continue;
        if(!patch->findCellAndShapeDerivatives(px[idx], ni, d_S))
          continue;

        vel = Vector(0.0,0.0,0.0);
        acc = Vector(0.0,0.0,0.0);

        if(d_heatConductionInvolved) {
          pTemperatureGradient[idx] = Vector(0.0,0.0,0.0);
          tempRate = 0;
        }

        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < 8; k++) {
	   vel += gvelocity_star[ni[k]]  * S[k];
	   acc += gacceleration[ni[k]]   * S[k];
	   
           if(d_heatConductionInvolved) {
             tempRate = gTemperatureRate[ni[k]] * S[k];
             for (int j = 0; j<3; j++){
               pTemperatureGradient[idx](j+1) += 
                 gTemperature[ni[k]] * d_S[k](j) * oodx[j];
             }
           }
        }

        // Update the particle's position and velocity
        px[idx]        += vel * delT;
        pvelocity[idx] += acc * delT;
        if(d_heatConductionInvolved) {
          pTemperatureRate[idx] = tempRate;
          pTemperature[idx] += tempRate * delT;
        }
        
        ke += .5*pmass[idx]*pvelocity[idx].length2();
      }

      // Store the new result
      new_dw->put(px,        lb->pXLabel, matlindex, patch);
      new_dw->put(pvelocity, lb->pVelocityLabel, matlindex, patch);

      ParticleVariable<Vector> pexternalforce;

      new_dw->put(pmass,          lb->pMassLabel, matlindex, patch);
      old_dw->get(pexternalforce, lb->pExternalForceLabel, matlindex, patch,
		  Ghost::None, 0);
      new_dw->put(pexternalforce, lb->pExternalForceLabel, matlindex, patch);

      new_dw->put(sum_vartype(ke), lb->KineticEnergyLabel);

      if(d_heatConductionInvolved) {
        new_dw->put(pTemperatureRate, lb->pTemperatureRateLabel, matlindex, patch);
        new_dw->put(pTemperature, lb->pTemperatureLabel, matlindex, patch);
        new_dw->put(pTemperatureGradient, lb->pTemperatureGradientLabel, matlindex, patch);
      }

    }
  }

  static int ts=0;
#if 0
  // Code to dump out tecplot files
  int freq = 10;

  if (( ts % freq) == 0) {
   char fnum[5];
   string filename;
   int stepnum=ts/freq;
   sprintf(fnum,"%04d",stepnum);
   string partroot("partout");

   filename = partroot+fnum;
   ofstream partfile(filename.c_str());

   partfile << "TITLE = \"Time Step # " << ts <<"\"," << endl;
   partfile << "VARIABLES = X,Y,Z,U,V,W,MATL" << endl;
   partfile << "ZONE T=\"PARTICLES\", I= " << numPTotal;
   partfile <<", F=POINT" << endl;

   for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int matlindex = matl->getDWIndex();
      int vfindex = matl->getVFIndex();
      // Get the arrays of particle values to be changed
      ParticleVariable<Point> px;
      old_dw->get(px, lb->XLabel, matlindex, patch, Ghost::None, 0);
      ParticleVariable<Vector> pv;
      old_dw->get(pv, lb->pVelocityLabel, matlindex, patch, Ghost::None, 0);
      ParticleVariable<double> pmass;
      old_dw->get(pmass,lb-> pMassLabel, matlindex, patch, Ghost::None, 0);

      ParticleSubset* pset = px.getParticleSubset();

      cout << "Time step # " << ts << endl;

      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
	 particleIndex idx = *iter;

        partfile << px[idx].x() <<" "<< px[idx].y() <<" "<< px[idx].z() <<" "<<
		    pv[idx].x() <<" "<< pv[idx].y() <<" "<< pv[idx].z() <<" "<<
		    vfindex << endl;

      }
    }
   }
  }
#endif

  static ofstream tmpout("tmp.out");
//  tmpout << ts << " " << ke << " " << se << std::endl;
  tmpout << ts << " " << ke << std::endl;
  ts++;
}

void SerialMPM::crackGrow(const ProcessorContext*,
                          const Patch* /*patch*/,
                          DataWarehouseP& /*old_dw*/,
                          DataWarehouseP& /*new_dw*/)
{
}

// $Log$
// Revision 1.79  2000/06/03 05:25:44  sparker
// Added a new for pSurfLabel (was uninitialized)
// Uncommented pleaseSaveIntegrated
// Minor cleanups of reduction variable use
// Removed a few warnings
//
// Revision 1.78  2000/06/01 23:12:04  guilkey
// Code to store integrated quantities in the DW and save them in
// an archive of sorts.  Also added the "computes" in the right tasks.
//
// Revision 1.77  2000/05/31 21:20:50  tan
// ThermalContact::computeHeatExchange() linked to scheduleTimeAdvance to handle
// thermal contact.
//
// Revision 1.76  2000/05/31 18:30:21  tan
// Create linkage to ThermalContact model.
//
// Revision 1.75  2000/05/31 17:40:57  tan
// Particle temperature gradient computations included in
// interpolateToParticlesAndUpdate().
//
// Revision 1.74  2000/05/31 16:10:17  tan
// Heat conduction computations included in interpolateParticlesToGrid().
//
// Revision 1.73  2000/05/31 00:34:43  tan
// temp to tempRate
//
// Revision 1.72  2000/05/30 21:03:22  dav
// delt to delT
//
// Revision 1.71  2000/05/30 20:18:58  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.70  2000/05/30 18:17:26  dav
// removed the include varLaberl
//
// Revision 1.69  2000/05/30 17:07:34  dav
// Removed commented out labels.  Other MPI fixes.  Changed delt to delT so I would stop thinking of it as just delta.
//
// Revision 1.68  2000/05/30 04:26:17  tan
// Heat conduction algorithm integrated into scheduleTimeAdvance().
//
// Revision 1.67  2000/05/26 23:07:03  tan
// Rewrite interpolateToParticlesAndUpdate to include heat conduction.
//
// Revision 1.66  2000/05/26 21:37:30  jas
// Labels are now created and accessed using Singleton class MPMLabel.
//
// Revision 1.65  2000/05/26 17:14:56  tan
// Added solveHeatEquations on grid.
//
// Revision 1.64  2000/05/26 02:27:48  tan
// Added computeHeatRateGeneratedByInternalHeatFlux() for thermal field
// computation.
//
// Revision 1.63  2000/05/25 23:03:11  guilkey
// Implemented calls to addComputesAndRequires for the Contact
// funtions (addComputesAndRequiresInterpolated and
// addComputesAndRequiresIntegrated)
//
// Revision 1.62  2000/05/25 22:06:34  tan
// A boolean variable d_heatConductionInvolved is set to true when
// heat conduction considered in the simulation.
//
// Revision 1.61  2000/05/23 02:25:45  tan
// Put all velocity-field independent variables on material
// index of 0.
//
// Revision 1.60  2000/05/18 18:50:25  jas
// Now using the gravity from the input file.
//
// Revision 1.59  2000/05/18 16:36:37  guilkey
// Numerous small changes including:
//   1.  Moved carry forward of particle volume to the cons. models.
//   2.  Added more data to the tecplot files (commented out)
//   3.  Computing strain energy on all particles at each time step,
//       in addition to the kinetic energy.
//   4.  Now printing out two files for diagnostics, one with energies
//       and one with center of mass.
//
// Revision 1.58  2000/05/16 00:40:51  guilkey
// Added code to do boundary conditions, print out tecplot files, and a
// few other things.  Most of this is now commented out.
//
// Revision 1.57  2000/05/15 20:03:22  dav
// couple of cleanups
//
// Revision 1.56  2000/05/15 19:39:37  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
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
// Made patches have a single uniform index space - still needs work
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

/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <CCA/Components/MPM/Materials/ConstitutiveModel/TransIsoHypoFrictional.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/MiscMath.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MinMax.h>

//#include <iostream>

using namespace std;
using namespace Uintah;

// _________________transversely isotropic hypoelastic material with frictional strength [Michael Homel's]
TransIsoHypoFrictional::TransIsoHypoFrictional(ProblemSpecP& ps, MPMFlags* Mflag) :
		  				ConstitutiveModel(Mflag)
{

	// Private Class Variables:
	one_third      = 1.0/3.0;
	//two_third      = 2.0/3.0;
	//four_third     = 4.0/3.0;
	//sqrt_two       = sqrt(2.0);
	//one_sqrt_two   = 1.0/sqrt_two;
	//sqrt_three     = sqrt(3.0);
	//one_sqrt_three = 1.0/sqrt_three;
	//one_ninth      = 1.0/9.0;
	//one_sixth      = 1.0/6.0;
	//pi  = 3.141592653589793238462;
	//pi_fourth = 0.25*pi;
	//pi_half = 0.5*pi;
	Identity.Identity();
	Zero.set(0.0);
	d_useModifiedEOS = false;

	//______________________material properties
	ps->require("E_t", d_cm.E_t);      // transverse modulus
	ps->require("Y_t", d_cm.Y_t);      // transverse cohesive strength
	ps->require("E_a", d_cm.E_a);      // axial modulus
	ps->require("nu_at", d_cm.nu_at);  // axial-transverse poisson ratio
	ps->require("nu_t", d_cm.nu_t);    // transverse poisson ratio
	ps->require("G_at", d_cm.G_at);    // axial-transverse shear modulus
	ps->require("n_fiber", d_cm.n_fiber);    // unit vector in fiber direction
	ps->require("mu_fiber", d_cm.mu_fiber);  // inter-fiber friction coefficient
	ps->require("crimp_stretch",d_cm.crimp_stretch);  // stretch to uncrimp fibers
	ps->require("crimp_ratio",d_cm.crimp_ratio);      // ratio of uncrimped to initial tensile modulus.
	ps->require("phi_0",d_cm.phi_0);   // initial micro porosity in fiber bundle
	ps->require("alpha",d_cm.alpha);   // exponent in porosity scaling of transverse moduli: (1-phi)^alpha
	ps->require("bulk",d_cm.bulk);   // nominal bulk modulus for MPM-ICE cell-centered compressibility calculations

	//ps->get("useModifiedEOS",d_useModifiedEOS);//no negative pressure for solids
  pCrimpLabel = VarLabel::create("p.crimp",
     ParticleVariable<double>::getTypeDescription());
  pCrimpLabel_preReloc = VarLabel::create("p.crimp+",
     ParticleVariable<double>::getTypeDescription());
  pStretchLabel = VarLabel::create("p.stretch",
     ParticleVariable<double>::getTypeDescription());
  pStretchLabel_preReloc = VarLabel::create("p.stretch+",
     ParticleVariable<double>::getTypeDescription());
  pPorosityLabel = VarLabel::create("p.porosity",
     ParticleVariable<double>::getTypeDescription());
  pPorosityLabel_preReloc = VarLabel::create("p.porosity+",
     ParticleVariable<double>::getTypeDescription());
}

TransIsoHypoFrictional::~TransIsoHypoFrictional()
// _______________________DESTRUCTOR
{
  VarLabel::destroy(pCrimpLabel);
  VarLabel::destroy(pCrimpLabel_preReloc);
  VarLabel::destroy(pStretchLabel);
  VarLabel::destroy(pStretchLabel_preReloc);
  VarLabel::destroy(pPorosityLabel);
  VarLabel::destroy(pPorosityLabel_preReloc);

}

//adds problem specification values to checkpoint data for restart
void TransIsoHypoFrictional::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
	ProblemSpecP cm_ps = ps;
	if (output_cm_tag) {
		cm_ps = ps->appendChild("constitutive_model");
		cm_ps->setAttribute("type","TransIsoHypoFrictional");
	}

	//______________________material properties
	cm_ps->appendElement("E_t", d_cm.E_t);      // transverse modulus
	cm_ps->appendElement("Y_t", d_cm.Y_t);      // transverse cohesive strength
	cm_ps->appendElement("E_a", d_cm.E_a);      // axial modulus
	cm_ps->appendElement("nu_at", d_cm.nu_at);  // axial-transverse poisson ratio
	cm_ps->appendElement("nu_t", d_cm.nu_t);    // transverse poisson ratio
	cm_ps->appendElement("G_at", d_cm.G_at);    // axial-transverse shear modulus
	cm_ps->appendElement("n_fiber", d_cm.n_fiber);      // unit vector in fiber direction
	cm_ps->appendElement("mu_fiber", d_cm.mu_fiber);    // inter-fiber friction coefficient
	cm_ps->appendElement("crimp_stretch",d_cm.crimp_stretch);  // stretch to uncrimp fibers
	cm_ps->appendElement("crimp_ratio",d_cm.crimp_ratio);      // ratio of uncrimped to initial tensile modulus.
	cm_ps->appendElement("phi_0",d_cm.phi_0);   // initial micro porosity in fiber bundle
	cm_ps->appendElement("alpha",d_cm.alpha);   // exponent in porosity scaling of transverse moduli: (1-phi)^alpha
	cm_ps->appendElement("bulk",d_cm.bulk);   // nominal bulk modulus for MPM-ICE cell-centered compressibility calculations
}

TransIsoHypoFrictional* TransIsoHypoFrictional::clone()
{
	return scinew TransIsoHypoFrictional(*this);
}


void TransIsoHypoFrictional::initializeCMData(const Patch* patch,
		const MPMMaterial* matl,
		DataWarehouse* new_dw)
// _____________________STRESS FREE REFERENCE CONFIG
{
	// Initialize the variables shared by all constitutive models
	// This method is defined in the ConstitutiveModel base class.
	initSharedDataForExplicit(patch, matl, new_dw);
	//
	// Allocates memory for internal state variables at beginning of run.
	//
	// Get the particles in the current patch
	ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

	// Put stuff in here to initialize each particle's
	// constitutive model parameters and deformationMeasure
	ParticleVariable<double>
	pStretch,  // fiber stretch
	pPorosity, // current porosity estimate based on incompressible solid
	pCrimp;    // modulus scale factor based on stretch

	// these are standard labels:
	// These have new labels unique to this model;
	new_dw->allocateAndPut(pStretch,        pStretchLabel,        pset);
	new_dw->allocateAndPut(pPorosity,       pPorosityLabel,       pset);
	new_dw->allocateAndPut(pCrimp,          pCrimpLabel,          pset);

	for(ParticleSubset::iterator iter = pset->begin();
			iter != pset->end();iter++){
		pStretch[*iter] = 1.0;
		pPorosity[*iter] = d_cm.phi_0;
		pCrimp[*iter] = computeCrimp(1.0);
	}

	computeStableTimestep(patch, matl, new_dw);
}

void TransIsoHypoFrictional::addParticleState(std::vector<const VarLabel*>& from,
		std::vector<const VarLabel*>& to)
//______________________________KEEPS TRACK OF THE PARTICLES AND THE RELATED VARIABLES
//______________________________(EACH CM ADD ITS OWN STATE VARS)
//______________________________AS PARTICLES MOVE FROM PATCH TO PATCH
{
	// Add the local particle state data for this constitutive model.
	from.push_back(lb->pFiberDirLabel);
	from.push_back(pStretchLabel);
	from.push_back(pPorosityLabel);
	from.push_back(pCrimpLabel);
	to.push_back(lb->pFiberDirLabel_preReloc);
	to.push_back(pStretchLabel_preReloc);
	to.push_back(pPorosityLabel_preReloc);
	to.push_back(pCrimpLabel_preReloc);
}

void TransIsoHypoFrictional::computeStableTimestep(const Patch* patch,
		const MPMMaterial* matl,
		DataWarehouse* new_dw)
//__________________________TIME STEP DEPENDS ON:
//__________________________CELL SPACING, VEL OF PARTICLE, MATERIAL WAVE SPEED @ EACH PARTICLE
//__________________________REDUCTION OVER ALL dT'S FROM EVERY PATCH PERFORMED
//__________________________(USE THE SMALLEST dT)
{
	// This is only called for the initial timestep - all other timesteps
	// are computed as a side-effect of computeStressTensor
	Vector dx = patch->dCell();
	int dwi = matl->getDWIndex();

	// Retrieve the array of constitutive parameters
	ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
	constParticleVariable<double> pmass, pvolume;
	constParticleVariable<Vector> pvelocity;

	new_dw->get(pmass,     lb->pMassLabel, pset);
	new_dw->get(pvolume,   lb->pVolumeLabel, pset);
	new_dw->get(pvelocity, lb->pVelocityLabel, pset);

	double c_dil = 0.0;
	Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

	// __________________________________________Compute wave speed at each particle, store the maximum

	// Using conservative upper bound moduli to estimate timestep.
	double M_t = (d_cm.E_t*(-d_cm.E_a + d_cm.E_t*d_cm.nu_at*d_cm.nu_at))/((1 + d_cm.nu_t)*(d_cm.E_a*(-1 + d_cm.nu_t) + 2*d_cm.E_t*d_cm.nu_at*d_cm.nu_at));
	double M_a = (d_cm.E_a*d_cm.E_a*(-1 + d_cm.nu_t))/(d_cm.E_a*(-1 + d_cm.nu_t) + 2*d_cm.E_t*d_cm.nu_at*d_cm.nu_at);
	// ----------------------
	double M = max( M_t, M_a );

	for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
		particleIndex idx = *iter;

		// this is valid only for F=Identity
		c_dil = sqrt(M*pvolume[idx]/pmass[idx]);

		WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
				Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
				Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
	}
	WaveSpeed = dx/WaveSpeed;
	double delT_new = WaveSpeed.minComponent();
	new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

Vector TransIsoHypoFrictional::getInitialFiberDir()
{
	return d_cm.n_fiber;
}


void TransIsoHypoFrictional::computeStressTensor(const PatchSubset* patches,
		const MPMMaterial* matl,
		DataWarehouse* old_dw,
		DataWarehouse* new_dw)
//___________________________________COMPUTES THE STRESS ON ALL THE PARTICLES
//__________________________________ IN A GIVEN PATCH FOR A GIVEN MATERIAL
//___________________________________CALLED ONCE PER TIME STEP
//___________________________________CONTAINS A COPY OF computeStableTimestep
{
	for(int pp=0;pp<patches->size();pp++)
	{
		const Patch* patch = patches->get(pp);

                double se=0.;

		delt_vartype delT;

		double J_new;               // volumetric stretch at end of step
		double c_dil=0.0;
		double E_a, G_at, nu_at, E_t, nu_t; // Scaled material parameters
		double h1, h2, h3, h4, h5;  // Tensor coefficients in constructing 4-th order stiffness
		double p2D;                 // compressive mean stress in transverse plane
		double maxShear;            // magnitude of max shar based on frictional slip
		double inPlane_J2;          // in-plane deviatoric stress J2 norm
		double inPlane_rootJ2;      // in-plane deviatoric stress magnitude
		double fiberShear_J2;       // out-of-plane deviatoric stress J2 norm
		double fiberShear_rootJ2;   // out-of-plane deviatoric stress magnitude

		Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
		Vector deformed_fiber_vector;  // deformed fiber vector (not normalized)
		Vector n;                      // deformed fiber direction (unit vector)

		Matrix3 D;                // Symmetric part of velocity gradient.
		Matrix3 tensorR, tensorU; // rotation and stretch for polar decomposition of F
		Matrix3 stress_old;        // Cauchy stress at start of step
		Matrix3 stress_trial;      // Trial cauchy stress
		Matrix3 stress_inPlane;    // Portion of trial stress in transverse plane
		Matrix3 stress_inPlaneDev; // Isotropic portion of in-plane trial stress
		Matrix3 stress_inPlaneIso; // Deviatoric portion of in-plane trial stress
		Matrix3 stress_outOfPlane; // Out-of-plane portion of trial stress
		Matrix3 stress_axial;      // Axial normal component of trial stress
		Matrix3 stress_fiberShear; // trasnverse-axial shear component of trial stress.
		Matrix3 defGrad;          // deformation gradient (copy since the particleVariable has limited operations)

		Vector dx = patch->dCell();
		int dwi = matl->getDWIndex();
		ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

		// note the particle variables are arrays and are accessed with a particle index
		// like p_velocity[*iter]

		// Particle variables, stored between timesteps, but unchanged in this routine
		constParticleVariable<double> p_stretch;   // stretch in fiber direction
		constParticleVariable<double> p_crimp;     // scaling for axial moduli based on fiber stretch.
		constParticleVariable<double> p_porosity;  // Current porosity assuming incompressible matrix.
		constParticleVariable<double> p_volume_new;
		constParticleVariable<double> p_mass;
		constParticleVariable<Vector> p_velocity;
		constParticleVariable<Vector> p_fiberDir;
		constParticleVariable<Matrix3> p_deformationGradient_new;
		constParticleVariable<Matrix3> p_deformationGradient;
		constParticleVariable<Matrix3> p_velGrad;
		constParticleVariable<Matrix3> p_stress_old;

		// Stored between timesteps but may be modified
		ParticleVariable<double> p_dTdt;
		ParticleVariable<double> p_q;             // artificial viscosity
		ParticleVariable<double> p_stretch_new;   // stretch in fiber direction
		ParticleVariable<double> p_crimp_new;     // scaling for axial moduli based on fiber stretch.
		ParticleVariable<double> p_porosity_new;  // Current porosity assuming incompressible matrix.
		ParticleVariable<Vector> p_fiberDir_new;

		ParticleVariable<Matrix3> p_stress_new;   // stress at end of step


		// Start of step labeled variables.
		old_dw->get(delT,                      lb->delTLabel,               getLevel(patches));
		old_dw->get(p_stretch,                     pStretchLabel,           pset);
		old_dw->get(p_crimp,                       pCrimpLabel,             pset);
		old_dw->get(p_porosity,                    pPorosityLabel,          pset);

		old_dw->get(p_velocity,                lb->pVelocityLabel,          pset);
		old_dw->get(p_fiberDir,                lb->pFiberDirLabel,          pset);

		old_dw->get(p_deformationGradient,     lb->pDeformationMeasureLabel,pset);
		old_dw->get(p_stress_old,              lb->pStressLabel,            pset); //initializeCMData()

    old_dw->get(p_mass,                    lb->pMassLabel,              pset);
		new_dw->get(p_volume_new,              lb->pVolumeLabel_preReloc,   pset);
		new_dw->get(p_velGrad,                 lb->pVelGradLabel_preReloc,  pset);
		new_dw->get(p_deformationGradient_new, lb->pDeformationMeasureLabel_preReloc, pset);

		// End of step labeled variables.
		new_dw->allocateAndPut(p_dTdt,        lb->pdTdtLabel,               pset);
		new_dw->allocateAndPut(p_q,           lb->p_qLabel_preReloc,        pset);
		new_dw->allocateAndPut(p_stretch_new,     pStretchLabel_preReloc,   pset);
		new_dw->allocateAndPut(p_crimp_new,       pCrimpLabel_preReloc,     pset);
		new_dw->allocateAndPut(p_porosity_new,    pPorosityLabel_preReloc,  pset);
		new_dw->allocateAndPut(p_fiberDir_new,lb->pFiberDirLabel_preReloc,  pset);
		new_dw->allocateAndPut(p_stress_new,  lb->pStressLabel_preReloc,    pset);

		// Loop over the particles of the current patch to update particle
		// stress at the end of the current timestep along with all other
		// required data such plastic strain, elastic strain, cap position, etc.

		for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++)
		{
			particleIndex idx = *iter;  //patch index

      // A parameter to consider the thermal effects of the plastic work which
      // is not coded in the current source code.
      p_dTdt[idx] = 0.0;

			// "rate of deformatino tensor" D = sym(L)
			D=(p_velGrad[idx] + p_velGrad[idx].Transpose())*0.5;

			// Deformed fiber vector at end of step
			p_fiberDir_new[idx] = (p_deformationGradient_new[idx])*d_cm.n_fiber;

			// Fiber stretch at end of step (assumes vector is initially a unit vector)
			p_stretch_new[idx] = p_fiberDir_new[idx].length();
			p_crimp_new[idx] = computeCrimp(p_stretch_new[idx]); // fiber stretch scale factor

			// Unit direction in deformed fiber direction at start of step:
			n = p_fiberDir[idx]/p_stretch[idx];

			// Volumetric part of the deformation
			J_new = p_deformationGradient_new[idx].Determinant();

			// Porosity is computed assuming incompressible matrix, 0<phi<1
			p_porosity_new[idx] = computePorosity(J_new);

			// Use polar decomposition to compute the rotation and stretch tensors at start of step
			p_deformationGradient[idx].polarDecompositionRMB(tensorU, tensorR);

			// Compute the unrotated symmetric part of the velocity gradient
			D = (tensorR.Transpose())*(D*tensorR);

			// Compute the unrotated stress at the start of the current timestep
			// here we're creating a new variable for the unrotated stress because
			// the p_stress_old is const.
			stress_old = (tensorR.Transpose())*(p_stress_old[idx]*tensorR);

			// Compute the unrotated deformed fiber direction at the start of the current time step
			n = (tensorR.Transpose())*n;

			// Scale elastic moduli:
			// --------------------
			double crimpScale = 0.5*( p_crimp[idx] + p_crimp_new[idx] );
			double porosity = 0.5*( p_porosity[idx] + p_porosity_new[idx] );
			double porosityScale = Pow(1.0-porosity, d_cm.alpha);

			E_a = crimpScale*(1.0-porosity)*d_cm.E_a;
      G_at = crimpScale*porosityScale*d_cm.G_at;
      nu_at = d_cm.nu_at;						  //Old: nu_at = crimpScale*porosityScale*d_cm.nu_at;
      E_t = porosityScale*d_cm.E_t;
			nu_t = d_cm.nu_t;								//Old: nu_t = porosityScale*d_cm.nu_t;

			// basis tensor coefficients for 4-th order tangent stiffness calculation.
			h1 = (E_a*E_a*(-1. + nu_t))/(E_a*(-1. + nu_t) + 2.*E_t*nu_at*nu_at);
			h2 = -((E_t*(E_a*nu_t + E_t*nu_at*nu_at))/((1. + nu_t)*(E_a*(-1. + nu_t) + 2.*E_t*nu_at*nu_at)));
			h3 = (E_t*E_a*nu_at)/(E_a - E_a*nu_t - 2.*E_t*nu_at*nu_at);
			h4 = E_t/(1. + nu_t);
			h5 = 2.*G_at;

			// Compute trial stress:
			// --------------------
			// stress_trial_ij = stress_old_ij + (h1*b1[n]_ijpq + h2*b2[n]_ijpq + h3*b3[n]_ijpq + h4*b4[n]_ijpq + h5*b5[n]_ijpq ):D_pq
			stress_trial = stress_old;
			for(int i=0; i<3; i++){
				for(int j=0; j<3; j++){
					for(int p=0; p<3; p++){
						for(int q=0; q<3; q++){
							stress_trial(i,j) += ( h1*B1(n,i,j,p,q) + h2*B2(n,i,j,p,q) +
									h3*B3(n,i,j,p,q) + h4*B4(n,i,j,p,q) +
									h5*B5(n,i,j,p,q) )*D(p,q)*delT;
						}
					}
				}
			}

			// Additive decomposition of trial stress:
			// --------------------------------------
			// stress_trial = stress_inPlaneIso + stress_inPlaneDev + stress_axial + stress_fiberShear

			// In-plane stress = B4:stress_trial
			for(int i=0; i<3; i++){
				for(int j=0; j<3; j++){
					stress_inPlane(i,j) = 0.0; // this is uneccesary if Matrix3 init to 0l
					for(int p=0; p<3; p++){
						for(int q=0; q<3; q++){
							stress_inPlane(i,j) += B4(n,i,j,p,q)*stress_trial(p,q);
						}
					}
				}
			}

			// In-plane isotropic stress = (1/2)*B2:stress_trial
			for(int i=0; i<3; i++){
				for(int j=0; j<3; j++){
					stress_inPlaneIso(i,j) = 0.0; // this is uneccesary if Matrix3 init to 0l
					for(int p=0; p<3; p++){
						for(int q=0; q<3; q++){
							stress_inPlaneIso(i,j) += 0.5*B2(n,i,j,p,q)*stress_trial(p,q);
						}
					}
				}
			}

			// In-plane deviatoric stress:
			stress_inPlaneDev = stress_inPlane - stress_inPlaneIso;

			// Out of plane stress:
			stress_outOfPlane = stress_trial - stress_inPlane;

			// In-plane isotropic stress = (1/2)*B2:stress_trial
			for(int i=0; i<3; i++){
				for(int j=0; j<3; j++){
					stress_axial(i,j) = 0.0; // this is uneccesary if Matrix3 init to 0l
					for(int p=0; p<3; p++){
						for(int q=0; q<3; q++){
							stress_axial(i,j) += B1(n,i,j,p,q)*stress_trial(p,q);
						}
					}
				}
			}

			// Out of plane shear stress:
			stress_fiberShear = stress_outOfPlane - stress_axial;


			// Enforce yield criteria
			// ----------------------

			// Mean stress in the transverse plane:
			p2D = -0.5*stress_inPlaneIso.Trace();

			// Cut-off tensile 2-D stresses to avoid overshoot in unloading.
			if( p2D < 0.0 || p_porosity[idx] > d_cm.phi_0 )
			{
				stress_inPlaneIso = 0.0*stress_inPlaneIso;
				p2D = 0.0;
			}

			// We could modify the tensile cutoff based on the cohesive
			// strength, to allow some p_min = -Y_t/mu_fiber, but this
			// makes it complicated to define the porosity cutoff
			// for distension.  For now we do the simple thing and
			// just add a cohesive strength.

			// maximum shear stress allowed on any slip plane.
			maxShear = max( 1.0e-12 , d_cm.mu_fiber*p2D + d_cm.Y_t );

			// Enforce pressure-dependent strength in transverse plane:
			inPlane_J2 = 0.5*stress_inPlaneDev.Contract( stress_inPlaneDev );
			if( inPlane_J2 < 1.e-16 )
			{
				inPlane_J2 = 0.0;
			}
			inPlane_rootJ2 = sqrt(inPlane_J2);
			if( inPlane_rootJ2 > maxShear )
			{
				stress_inPlaneDev = stress_inPlaneDev*( maxShear / inPlane_rootJ2 );
			}

			// Enforce pressure-dependent strength out of transverse plane:
			fiberShear_J2 = 0.5*stress_fiberShear.Contract( stress_fiberShear );
			if( fiberShear_J2 < 1.e-16 )
			{
				fiberShear_J2 = 0.0;
			}
			fiberShear_rootJ2 = sqrt(fiberShear_J2);
			if(  fiberShear_rootJ2 > maxShear )
			{
				stress_fiberShear = stress_fiberShear*( maxShear / fiberShear_rootJ2 );
			}

			// Reassemble stress tensor
			// ------------------------
			Matrix3 stress_new = stress_inPlaneIso + stress_inPlaneDev + stress_fiberShear + stress_axial;

			// Compute the averaged stress
			Matrix3 AvgStress = (stress_new + stress_old)*0.5;
			// Compute the strain energy increment associated with the particle
			double e = (D(0,0)*AvgStress(0,0) +
					D(1,1)*AvgStress(1,1) +
					D(2,2)*AvgStress(2,2) +
					2.0*(D(0,1)*AvgStress(0,1) +
							D(0,2)*AvgStress(0,2) +
							D(1,2)*AvgStress(1,2))) * p_volume_new[idx]*delT;

			// Accumulate the total strain energy
			// MH! FIXME the initialization of se needs to be fixed as it is currently reset to 0
			se += e;


			// Re-rotate the stress based on the deformation gradient at the end of the step.
			// -----------------------------------------------------------------------------
			p_deformationGradient_new[idx].polarDecompositionRMB(tensorU, tensorR);
			p_stress_new[idx] = (tensorR*stress_new)*(tensorR.Transpose());

			// Update wavespeed, timestep, and artificial viscosity
			// -----------------------------------------------------

			// Use conservative upper bound limits since moduli can change greatly in a single step
			// e.g if the crimp variable goes from .01 to 1.
			double M_t = (d_cm.E_t*(-d_cm.E_a + d_cm.E_t*d_cm.nu_at*d_cm.nu_at))/((1 + d_cm.nu_t)*(d_cm.E_a*(-1 + d_cm.nu_t) + 2*d_cm.E_t*d_cm.nu_at*d_cm.nu_at));
			double M_a = (d_cm.E_a*d_cm.E_a*(-1 + d_cm.nu_t))/(d_cm.E_a*(-1 + d_cm.nu_t) + 2*d_cm.E_t*d_cm.nu_at*d_cm.nu_at);
			// ----------------------
			double M = max( M_t, M_a );

			double rho_cur = p_mass[idx]/p_volume_new[idx];
			       c_dil = sqrt(M/rho_cur);

			WaveSpeed=Vector(Max(c_dil+fabs(p_velocity[idx].x()),WaveSpeed.x()),
					Max(c_dil+fabs(p_velocity[idx].y()),WaveSpeed.y()),
					Max(c_dil+fabs(p_velocity[idx].z()),WaveSpeed.z()));

			double bulk = (d_cm.E_a*d_cm.E_t)/(d_cm.E_t - 4.*d_cm.E_t*d_cm.nu_at - 2.*d_cm.E_a*(-1. + d_cm.nu_t));

			// Compute artificial viscosity term
			if (flag->d_artificial_viscosity) {
				double dx_ave = (dx.x() + dx.y() + dx.z())*one_third;
				double c_bulk = sqrt(bulk/rho_cur);
				p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur, dx_ave);
			} else {
				p_q[idx] = 0.;
			}



			// Update internal energy
			// ----------------------

		}  // end loop over particles
		// Compute the stable timestep based on maximum value of "wave speed + particle velocity"
		WaveSpeed = dx/WaveSpeed; // Variable now holds critical timestep (not speed)
		double delT_new = WaveSpeed.minComponent();

		// Put the stable timestep and total strain enrgy
		new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
		if (flag->d_reductionVars->accStrainEnergy ||
				flag->d_reductionVars->strainEnergy) {
			new_dw->put(sum_vartype(se),        lb->StrainEnergyLabel);
		}  // end loop over particles
	} // end loop over patches
} // end computeStress function.

// estimates porosity assuming no matrix compression.
double TransIsoHypoFrictional::computePorosity(const double& J)
{
	double phi0 = d_cm.phi_0;
	double porosity = min( 1.0, max( 0.0, 1.0 - (1.0-phi0)/J) );
	return porosity;
}

// computes crimp scale factor based on fiber stretch
double TransIsoHypoFrictional::computeCrimp(const double& stretch)
{
	double crimpRatio = d_cm.crimp_ratio;     // ratio of tensile moduli to crimped
	double crimpStretch = d_cm.crimp_stretch; // crimpStretch should be > 0
	double crimp = (1.0/crimpRatio) +
			(1.0-1.0/crimpRatio)*smoothStep(stretch, 1.0, crimpStretch);
	return crimp;
}

// cubic blending function
double TransIsoHypoFrictional::smoothStep(const double& x,
		const double& xmin,
		const double& xmax)
{
	double eta = (x-xmin)/(xmax-xmin);
	double step = 0.0;
	if(x>xmin){
		step = (x>xmax) ? 1.0 : 3.0*eta*eta - 2.0*eta*eta*eta;
	}
	return step;
}

// Basis function for transverse isotropy B1(n)_ijpq
double TransIsoHypoFrictional::B1(const Vector& n, // fiber direction
		const int& i,    // index
		const int& j,    // index
		const int& p,    // index
		const int& q)    // index
{
	double B1_ijpq = n[i]*n[j]*n[p]*n[q];
	return B1_ijpq;
}

// Basis function for transverse isotropy B2(n)_ijpq
double TransIsoHypoFrictional::B2(const Vector& n, // fiber direction
		const int& i,    // index
		const int& j,    // index
		const int& p,    // index
		const int& q)    // index
{
	double B2_ijpq = Identity(i,j)*Identity(p,q) -
			( n[i]*n[j]*Identity(p,q) + Identity(i,j)*n[p]*n[q] ) +
			n[i]*n[j]*n[p]*n[q];
	return B2_ijpq;
}

// Basis function for transverse isotropy B3(n)_ijpq
double TransIsoHypoFrictional::B3(const Vector& n, // fiber direction
		const int& i,    // index
		const int& j,    // index
		const int& p,    // index
		const int& q)    // index
{
	double B3_ijpq = ( n[i]*n[j]*Identity(p,q) + Identity(i,j)*n[p]*n[q] ) -
			2.0*n[i]*n[j]*n[p]*n[q];
	return B3_ijpq;
}

// Basis function for transverse isotropy B4(n)_ijpq
double TransIsoHypoFrictional::B4(const Vector& n, // fiber direction
		const int& i,    // index
		const int& j,    // index
		const int& p,    // index
		const int& q)    // index
{
	double B4_ijpq = 0.5*( Identity(i,p)*Identity(j,q) + Identity(i,q)*Identity(j,p) ) -
			0.5*( Identity(i,q)*n[j]*n[p] + n[i]*Identity(j,p)*n[q] + n[i]*Identity(j,q)*n[p] + Identity(i,p)*n[j]*n[q] ) +
			n[i]*n[j]*n[p]*n[q];
	return B4_ijpq;
}

// Basis function for transverse isotropy B5(n)_ijpq
double TransIsoHypoFrictional::B5(const Vector& n, // fiber direction
		const int& i,    // index
		const int& j,    // index
		const int& p,    // index
		const int& q)    // index
{
	double B5_ijpq = 0.5*( Identity(i,p)*n[j]*n[q] + Identity(i,q)*n[j]*n[p] + n[i]*Identity(j,q)*n[p] + n[i]*Identity(j,p)*n[q] ) -
			2.0*n[i]*n[p]*n[j]*n[q];
	return B5_ijpq;
}


void TransIsoHypoFrictional::carryForward(const PatchSubset* patches,
		const MPMMaterial* matl,
		DataWarehouse* old_dw,
		DataWarehouse* new_dw)
//___________________________________________________________used with RigidMPM
{
	for(int p=0;p<patches->size();p++){
		const Patch* patch = patches->get(p);
		int dwi = matl->getDWIndex();
		ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

		// Carry forward the data common to all constitutive models
		// when using RigidMPM.
		// This method is defined in the ConstitutiveModel base class.
		carryForwardSharedData(pset, old_dw, new_dw, matl);

		// TODO Finish this.
		//     		// Stored between timesteps but may be modified

		ParticleVariable<double> p_dTdt;
		ParticleVariable<double> p_q;            // artificial viscosity

		constParticleVariable<double> p_stretch;   // stretch in fiber direction
		constParticleVariable<double> p_crimp;     // scaling for axial moduli based on fiber stretch.
		constParticleVariable<double> p_porosity;  // Current porosity assuming incompressible matrix.
		constParticleVariable<Vector> p_fiberDir;

		old_dw->get(p_stretch,        pStretchLabel,       pset);
		old_dw->get(p_crimp,          pCrimpLabel,         pset);
		old_dw->get(p_porosity,       pPorosityLabel,      pset);
		old_dw->get(p_fiberDir,       lb->pFiberDirLabel,  pset);

		ParticleVariable<double> p_stretch_new;   // stretch in fiber direction
		ParticleVariable<double> p_crimp_new;     // scaling for axial moduli based on fiber stretch.
		ParticleVariable<double> p_porosity_new;  // Current porosity assuming incompressible matrix.
		ParticleVariable<Vector> p_fiberDir_new;

		new_dw->allocateAndPut(p_stretch_new,   pStretchLabel_preReloc,      pset);
		new_dw->allocateAndPut(p_crimp_new,     pCrimpLabel_preReloc,        pset);
		new_dw->allocateAndPut(p_porosity_new,  pPorosityLabel_preReloc,     pset);
		new_dw->allocateAndPut(p_fiberDir_new,  lb->pFiberDirLabel_preReloc, pset);

		for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++){
			particleIndex idx = *iter;
			p_stretch_new[idx] = p_stretch[idx];
			p_crimp_new[idx] = p_crimp[idx];
			p_porosity_new[idx] = p_porosity[idx];
			p_fiberDir_new[idx] = p_fiberDir[idx];
		}
		new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());

		if (flag->d_reductionVars->accStrainEnergy ||
				flag->d_reductionVars->strainEnergy) {
			new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
		}
	}
}

 void TransIsoHypoFrictional::addInitialComputesAndRequires(Task* task,
                                                     const MPMMaterial* matl,
                                                     const PatchSet*) const
 {
   const MaterialSubset* matlset = matl->thisMaterial();
   task->computes(pStretchLabel,              matlset);
   task->computes(pCrimpLabel,                matlset);
   task->computes(pPorosityLabel,             matlset);
   task->computes(lb->pStressLabel_preReloc,  matlset);
 }

 void TransIsoHypoFrictional::addComputesAndRequires(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patches) const
   //___________TELLS THE SCHEDULER WHAT DATA
   //___________NEEDS TO BE AVAILABLE AT THE TIME computeStressTensor IS CALLED
 {
   // Add the computes and requires that are common to all explicit
   // constitutive models.  The method is defined in the ConstitutiveModel
   // base class.
   const MaterialSubset* matlset = matl->thisMaterial();
   addSharedCRForExplicit(task, matlset, patches);

   // Other constitutive model and input dependent computes and requires
   Ghost::GhostType  gnone = Ghost::None;

   task->requires(Task::OldDW, lb->pFiberDirLabel, matlset,gnone);
   task->requires(Task::OldDW, pStretchLabel,      matlset,gnone);
   task->requires(Task::OldDW, pCrimpLabel,        matlset,gnone);
   task->requires(Task::OldDW, pPorosityLabel,     matlset,gnone);

   task->computes(lb->pFiberDirLabel_preReloc, matlset);
   task->computes(pStretchLabel_preReloc,      matlset);
   task->computes(pCrimpLabel_preReloc,        matlset);
   task->computes(pPorosityLabel_preReloc,     matlset);
 }

 void TransIsoHypoFrictional::addComputesAndRequires(Task* ,
                                            const MPMMaterial* ,
                                            const PatchSet* ,
                                            const bool ) const
   //_________________________________________here this one's empty
 {
 }


 // The "CM" versions use the pressure-volume relationship of the CNH model
 double TransIsoHypoFrictional::computeRhoMicroCM(double pressure,
                                         const double p_ref,
                                         const MPMMaterial* matl,
                                         double temperature,
                                         double rho_guess)
 {
	 double rho_orig = matl->getInitialDensity();

		// Particle data are not available for MPM-ICE cell-centered compressibility calculations
		// so instead we use a user-specified nominal bulk modulus
   double Bulk = d_cm.bulk;

   double p_gauge = pressure - p_ref;
   double rho_cur;

   if(d_useModifiedEOS && p_gauge < 0.0) {
     double A = p_ref;           // MODIFIED EOS
     double n = p_ref/Bulk;
     rho_cur = rho_orig*pow(pressure/A,n);
   } else {                      // STANDARD EOS
     rho_cur = rho_orig*(p_gauge/Bulk + sqrt((p_gauge/Bulk)*(p_gauge/Bulk) +1));
   }
   return rho_cur;
 }

void TransIsoHypoFrictional::computePressEOSCM(const double rho_cur,double& pressure,
		const double p_ref,
		double& dp_drho, double& tmp,
		const MPMMaterial* matl,
		double temperature)
{
	//  This is for cell centered MPM-ICE calculations, where particle data are not available
	//  so we have to estimate the response based on initial conditions.

	double rho_orig = matl->getInitialDensity();

	// Compute bulk modulus based on initial material properties.
	// --------------------------------------------------------
	//	double J = rho_orig/rho_cur; // Volumetric part of the deformation
	//	double stretch = 1.0;        // Assume unity fiber stretch since particle data are not available;
	//	double Bulk = computeBulkModulus(J,stretch);

	// User specified bulk modulus.
	// --------------------------------------------------------
	// Particle data are not available for MPM-ICE cell-centered compressibility calculations
	// so instead we use a user-specified nominal bulk modulus
	double Bulk = d_cm.bulk;

	if(d_useModifiedEOS && rho_cur < rho_orig){
		double A = p_ref;           // MODIFIED EOS
		double n = Bulk/p_ref;
		pressure = A*pow(rho_cur/rho_orig,n);
		dp_drho  = (Bulk/rho_orig)*pow(rho_cur/rho_orig,n-1);
		tmp      = dp_drho;         // speed of sound squared
	} else {                      // STANDARD EOS
		double p_g = .5*Bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
		pressure   = p_ref + p_g;
		dp_drho    = .5*Bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
		tmp        = Bulk/rho_cur;  // speed of sound squared
	}
}

double TransIsoHypoFrictional::computeBulkModulus(const double& J,      // volumetric stretch
		const double& stretch // fiber stretch
)
{
	// This currently isn't used since the bulk modulus and compressibility are
	// only called by MPM ICE routines that use cell-centered, not particle data.

	// stretch scaling of axial fiber stiffness
	double crimp = computeCrimp(stretch);
	double porosity = computePorosity(J); // Current porosity estimate

	// Scale elastic moduli:
	// Axial moduli
	double E_a = crimp*(1-porosity)*d_cm.E_a;
	double nu_at = crimp*d_cm.nu_at;

	// Transverse: moduli
	double E_t = pow(1.-porosity,d_cm.alpha)*d_cm.E_t;
	double nu_t = d_cm.nu_t;
	double Bulk = (E_a*E_t)/(E_t - 4.*E_t*nu_at - 2.*E_a*(-1. + nu_t));

	return Bulk;
}

double TransIsoHypoFrictional::getCompressibility()
{

	// Initial bulk modulus
	// double bulk = computeBulkModulus(1.0, 1.0);
	// User-prescribed value.
	double bulk = d_cm.bulk;
	return 1.0/bulk;
}


namespace Uintah {

#if 0
static MPI_Datatype makeMPI_CMData()
{
	ASSERTEQ(sizeof(TransIsoHypoFrictional::StateData), sizeof(double)*0);
	MPI_Datatype mpitype;
	Uintah::MPI::Type_vector(1, 0, 0, MPI_DOUBLE, &mpitype);
	Uintah::MPI::Type_commit(&mpitype);
	return mpitype;
}

const TypeDescription* fun_getTypeDescription(TransIsoHypoFrictional::StateData*)
{
	static TypeDescription* td = 0;
	if(!td){
		td = scinew TypeDescription(TypeDescription::Other,
				"TransIsoHypoFrictional::StateData", true, &makeMPI_CMData);
	}
	return td;
}
#endif
} // End namespace Uintah


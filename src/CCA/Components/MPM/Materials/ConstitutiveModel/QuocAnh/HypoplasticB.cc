/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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
 //#include <CCA/Components/MPM/ConstitutiveModel/HypoplasticB.h> // Uintah 1.6
 //#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h> // Uintah 1.6

#include <CCA/Components/MPM/Materials/ConstitutiveModel/QuocAnh/HypoplasticB.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>

#include <CCA/Ports/DataWarehouse.h>

#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>

//#include <Core/Labels/MPMLabel.h> // Uintah 1.6
#include <CCA/Components/MPM/Core/MPMLabel.h>

#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h>
#include <Core/ProblemSpec/ProblemSpec.h>

//#include <Core/Containers/StaticArray.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>

#include <sci_defs/uintah_defs.h>

#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

extern "C" {

#if defined( FORTRAN_UNDERSCORE_END )
#  define DMMCHK dmmchk_
#  define DIAMM_CALC diamm_calc_
#  define DMMRXV dmmrxv_
#elif defined( FORTRAN_UNDERSCORE_LINUX )
#  define DMMCHK dmmchk_
#  define DMMRXV dmmrxv_
#  define DIAMM_CALC diamm_calc__
#else // NONE
#  define DMMCHK dmmchk
#  define DIAMM_CALC diamm_calc
#  define DMMRXV dmmrxv
#endif

	void DMMRXV(double UI[], double UJ[], double UK[], int &nx, char* namea[],
		char* keya[], double rinit[], double rdim[], int iadvct[],
		int itype[]);
}

/*
double ElAreaB[20000];
double coordXB[1][20000];
double coordYB[1][20000];
//double coordZ[1][20000];
double dlocMB[1][20000];
double dnonlocMB[1][20000];
int iNLB;
*/

using namespace std; using namespace Uintah;

HypoplasticB::HypoplasticB(ProblemSpecP& ps, MPMFlags* Mflag)
	: ConstitutiveModel(Mflag)
{
	d_NBASICINPUTS = 16;
	d_NMGDC = 0;

	// Total number of properties
	d_NDMMPROP = d_NBASICINPUTS + d_NMGDC;

	// pre-initialize all of the user inputs to zero.
	for (int i = 0; i < d_NDMMPROP; i++) {
		UI[i] = 0.;
	}
	// Read model parameters from the input file
	getInputParameters(ps);

	// Check that model parameters are valid and allow model to change if needed

	// DMMCHK(UI,UI,&UI[d_NBASICINPUTS]);
	CheckModel(UI);
	//Create VarLabels for GeoModel internal state variables (ISVs)
	int nx;

	// DMMRXV( UI, UI, UI, nx, namea, keya, rinit, rdim, iadvct, itype );

	nx = 17;

	for (int i = 0; i < nx; i++)
	{
		rinit[i] = UI[i];
	}

	d_NINSV = nx;
	//  cout << "d_NINSV = " << d_NINSV << endl;

	initializeLocalMPMLabels();
}

#if 0
HypoplasticB::HypoplasticB(const HypoplasticB* cm) : ConstitutiveModel(cm)
{
	for (int i = 0; i < d_NDMMPROP; i++) {
		UI[i] = cm->UI[i];
	}

	//Create VarLabels for Diamm internal state variables (ISVs)
	initializeLocalMPMLabels();
}
#endif

HypoplasticB::~HypoplasticB()
{
	for (unsigned int i = 0; i < ISVLabels.size(); i++) {
		VarLabel::destroy(ISVLabels[i]);
	}
}

void HypoplasticB::outputProblemSpec(ProblemSpecP& ps, bool output_cm_tag)
{
	ProblemSpecP cm_ps = ps;
	if (output_cm_tag) {
		cm_ps = ps->appendChild("constitutive_model");
		cm_ps->setAttribute("type", "HypoplasticB");
	}

	cm_ps->appendElement("ei0_B", UI[0]);
	cm_ps->appendElement("ed0_B", UI[1]);
	cm_ps->appendElement("ec0_B", UI[2]);
	cm_ps->appendElement("phic_B", UI[3]);
	cm_ps->appendElement("hs_B", UI[4]);
	cm_ps->appendElement("beta_B", UI[5]);
	cm_ps->appendElement("n_B", UI[6]);
	cm_ps->appendElement("alpha_B", UI[7]);
	cm_ps->appendElement("E_B", UI[8]);
	cm_ps->appendElement("v_B", UI[9]);
	cm_ps->appendElement("epocz_B", UI[10]);
	cm_ps->appendElement("nl_B", UI[11]);
	cm_ps->appendElement("lchar_B", UI[12]);
	cm_ps->appendElement("mean_pressure", UI[13]);
	cm_ps->appendElement("phase", UI[14]);
	cm_ps->appendElement("phase_change", UI[15]);
	cm_ps->appendElement("Volumerate", UI[16]);
}

HypoplasticB* HypoplasticB::clone()
{
	return scinew HypoplasticB(*this);
}

void HypoplasticB::initializeCMData(const Patch* patch,
	const MPMMaterial* matl,
	DataWarehouse* new_dw)
{
	// Initialize the variables shared by all constitutive models
	// This method is defined in the ConstitutiveModel base class.
	initSharedDataForExplicit(patch, matl, new_dw);

	ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

	/*
	// PVoidRatio
	ParticleVariable<double>  pVoidRatio;
	new_dw->allocateAndPut(pVoidRatio, lb->pVoidRatioLabel, pset);

	ParticleSubset::iterator iter = pset->begin();
	for (; iter != pset->end(); iter++) {
		particleIndex idx = *iter;

		pVoidRatio[idx] = 0.0;

	}
	*/

	//StaticArray<ParticleVariable<double> > ISVs(d_NINSV+1);
	std::vector<ParticleVariable<double> > ISVs(d_NINSV + 1);

	cout << "In initializeCMData" << endl;
	for (int i = 0; i < d_NINSV; i++) {
		new_dw->allocateAndPut(ISVs[i], ISVLabels[i], pset);
		ParticleSubset::iterator iter = pset->begin();
		for (; iter != pset->end(); iter++) {
			ISVs[i][*iter] = rinit[i];
		}
	}

	computeStableTimestep(patch, matl, new_dw);
}

void HypoplasticB::addParticleState(std::vector<const VarLabel*>& from,
	std::vector<const VarLabel*>& to)
{
	// Add the local particle state data for this constitutive model.
	for (int i = 0; i < d_NINSV; i++) {
		from.push_back(ISVLabels[i]);
		to.push_back(ISVLabels_preReloc[i]);
	}
}

void HypoplasticB::computeStableTimestep(const Patch* patch,
	const MPMMaterial* matl,
	DataWarehouse* new_dw)
{
	// This is only called for the initial timestep - all other timesteps
	// are computed as a side-effect of computeStressTensor
	Vector dx = patch->dCell();
	int dwi = matl->getDWIndex();
	ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
	constParticleVariable<double> pmass, pvolume;
	constParticleVariable<Vector> pvelocity;

	new_dw->get(pmass, lb->pMassLabel, pset);
	new_dw->get(pvolume, lb->pVolumeLabel, pset);
	new_dw->get(pvelocity, lb->pVelocityLabel, pset);

	double c_dil = 0.0;
	Vector WaveSpeed(1.e-12, 1.e-12, 1.e-12);
	double bulk = (UI[8]) / (3 * (1 - 2 * UI[9]));
	double G = (UI[8]) / (2 * (1 + UI[9]));

	for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++) {
		particleIndex idx = *iter;

		// Compute wave speed at each particle, store the maximum
		c_dil = sqrt((bulk + 4.*G / 3.)*pvolume[idx] / pmass[idx]);
		WaveSpeed = Vector(Max(c_dil + fabs(pvelocity[idx].x()), WaveSpeed.x()),
			Max(c_dil + fabs(pvelocity[idx].y()), WaveSpeed.y()),
			Max(c_dil + fabs(pvelocity[idx].z()), WaveSpeed.z()));
	}

	WaveSpeed = dx / WaveSpeed;
	double delT_new = WaveSpeed.minComponent();
	new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void HypoplasticB::computeStressTensor(const PatchSubset* patches,
	const MPMMaterial* matl,
	DataWarehouse* old_dw,
	DataWarehouse* new_dw)
{
	double rho_orig = matl->getInitialDensity();
	for (int p = 0; p < patches->size(); p++) {
		double se = 0.0;
		const Patch* patch = patches->get(p);

		Matrix3 Identity; Identity.Identity();
		double c_dil = 0.0;
		Vector WaveSpeed(1.e-12, 1.e-12, 1.e-12);
		Vector dx = patch->dCell();

		int dwi = matl->getDWIndex();
		// Create array for the particle position
		ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
		constParticleVariable<Matrix3> deformationGradient, pstress;
		ParticleVariable<Matrix3> pstress_new;
		constParticleVariable<Matrix3> deformationGradient_new, velGrad;
		constParticleVariable<double> pmass, pvolume, ptemperature;
		ParticleVariable<double> pvolume_new;
		constParticleVariable<Vector> pvelocity;
		constParticleVariable<Point> px;
		delt_vartype delT;
		old_dw->get(delT, lb->delTLabel, getLevel(patches));

		old_dw->get(px, lb->pXLabel, pset);
		old_dw->get(pstress, lb->pStressLabel, pset);
		old_dw->get(pmass, lb->pMassLabel, pset);
		old_dw->get(pvolume, lb->pVolumeLabel, pset);
		old_dw->get(pvelocity, lb->pVelocityLabel, pset);
		old_dw->get(ptemperature, lb->pTemperatureLabel, pset);
		old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);

		std::vector<constParticleVariable<double> > ISVs(d_NINSV + 1);


		for (int i = 0; i < d_NINSV; i++) {
			old_dw->get(ISVs[i], ISVLabels[i], pset);
		}

		ParticleVariable<double> pdTdt, p_q;
		//ParticleVariable<double> pVoidRatio;

		new_dw->allocateAndPut(pstress_new, lb->pStressLabel_preReloc, pset);
		new_dw->allocateAndPut(pdTdt, lb->pdTdtLabel, pset);
		// void ratio
		//new_dw->allocateAndPut(pVoidRatio, lb->pVoidRatioLabel, pset);

		new_dw->allocateAndPut(p_q, lb->p_qLabel_preReloc, pset);
		new_dw->get(deformationGradient_new,
			lb->pDeformationMeasureLabel_preReloc, pset);

		new_dw->getModifiable(pvolume_new, lb->pVolumeLabel_preReloc, pset);

		new_dw->get(velGrad, lb->pVelGradLabel_preReloc, pset);

		std::vector<ParticleVariable<double> > ISVs_new(d_NINSV + 1);
		for (int i = 0; i < d_NINSV; i++) {
			new_dw->allocateAndPut(ISVs_new[i], ISVLabels_preReloc[i], pset);
		}	

		for (ParticleSubset::iterator iter = pset->begin();
			iter != pset->end(); iter++) {
			particleIndex idx = *iter;

			// Assign zero internal heating by default - modify if necessary.
			pdTdt[idx] = 0.0;

			// Calculate rate of deformation D, and deviatoric rate DPrime,
			Matrix3 D = (velGrad[idx] + velGrad[idx].Transpose())*.5;

			// get the volumetric part of the deformation
			//double J = deformationGradient_new[idx].Determinant();

			// This is wrong because it is for current configuraition but avoid the negative jacobian
			double J = exp(delT * velGrad[idx].Trace());

			// Check 1: Look at Jacobian
			if (!(J > 0.0)) {
				cerr << getpid();
				constParticleVariable<long64> pParticleID;
				old_dw->get(pParticleID, lb->pParticleIDLabel, pset);
				cerr << "**ERROR** Negative Jacobian of deformation gradient"
					<< " in particle " << pParticleID[idx] << endl;
				cerr << "l = " << velGrad[idx] << endl;
				cerr << "F_old = " << deformationGradient[idx] << endl;
				cerr << "F_new = " << deformationGradient_new[idx] << endl;
				cerr << "J_hypo = " << J << endl;
				throw InternalError("Negative Jacobian", __FILE__, __LINE__);
			}

			// Compute the local sound speed
			//double rho_cur = rho_orig / J;

			// NEED TO FIND R
			Matrix3 tensorR, tensorU;

			// Look into using Rebecca's PD algorithm
			deformationGradient_new[idx].polarDecompositionRMB(tensorU, tensorR);

			// This is the previous timestep Cauchy stress
			// unrotated tensorSig=R^T*pstress*R
			Matrix3 tensorSig = (tensorR.Transpose())*(pstress[idx]*tensorR);

			// Load into 1-D array for the fortran code
			double sigarg[6];
			sigarg[0] = tensorSig(0, 0);
			sigarg[1] = tensorSig(1, 1);
			sigarg[2] = tensorSig(2, 2);
			sigarg[3] = tensorSig(0, 1);
			sigarg[4] = tensorSig(1, 2);
			sigarg[5] = tensorSig(2, 0);

			// UNROTATE D: S=R^T*D*R
			D = (tensorR.Transpose())*(D*tensorR);

			// Load into 1-D array for the fortran code
			double Darray[6];
			Darray[0] = D(0, 0);
			Darray[1] = D(1, 1);
			Darray[2] = D(2, 2);
			Darray[3] = D(0, 1);
			Darray[4] = D(1, 2);
			Darray[5] = D(2, 0);
			double svarg[d_NINSV];
			double USM = 9e99;
			double dt = delT;
			int nblk = 1;

			// Load ISVs into a 1D array for fortran code
			for (int i = 0; i < d_NINSV; i++) {
				svarg[i] = ISVs[i][idx];
			}

			pvolume_new[idx] = pvolume[idx] * exp(delT * velGrad[idx].Trace());

			// Ratio of current volume over initial volume (jacobian)
			svarg[16] = pvolume_new[idx] * rho_orig / pmass[idx];

			// Compute the local sound speed
			double rho_cur = rho_orig / svarg[16];

			//calculateStress for each particle
			CalculateStress(nblk, d_NINSV, dt, UI, sigarg, Darray, svarg, USM);

			//pVoidRatio[idx] = svarg[10];


			// Unload ISVs from 1D array into ISVs_new
			for (int i = 0; i < d_NINSV; i++) {
				ISVs_new[i][idx] = svarg[i];
			}

			// This is the Cauchy stress, still unrotated
			tensorSig(0, 0) = sigarg[0];
			tensorSig(1, 1) = sigarg[1];
			tensorSig(2, 2) = sigarg[2];
			tensorSig(0, 1) = sigarg[3];
			tensorSig(1, 0) = sigarg[3];
			tensorSig(2, 1) = sigarg[4];
			tensorSig(1, 2) = sigarg[4];
			tensorSig(2, 0) = sigarg[5];
			tensorSig(0, 2) = sigarg[5];

			// ROTATE pstress_new: S=R*tensorSig*R^T
			pstress_new[idx] = (tensorR*tensorSig)*(tensorR.Transpose());

			//cerr << pstress_new[idx] << endl;

			c_dil = sqrt(USM / rho_cur);

			// Compute the strain energy for all the particles
			Matrix3 AvgStress = (pstress_new[idx] + pstress[idx])*.5;

			double e = (D(0, 0)*AvgStress(0, 0) +
				D(1, 1)*AvgStress(1, 1) +
				D(2, 2)*AvgStress(2, 2) +
				2.*(D(0, 1)*AvgStress(0, 1) +
					D(0, 2)*AvgStress(0, 2) +
					D(1, 2)*AvgStress(1, 2))) * pvolume_new[idx] * delT;
			se += e;

			// Compute wave speed at each particle, store the maximum
			Vector pvelocity_idx = pvelocity[idx];
			WaveSpeed = Vector(Max(c_dil + fabs(pvelocity_idx.x()), WaveSpeed.x()),
				Max(c_dil + fabs(pvelocity_idx.y()), WaveSpeed.y()),
				Max(c_dil + fabs(pvelocity_idx.z()), WaveSpeed.z()));

			// Compute artificial viscosity term
			if (flag->d_artificial_viscosity) {
				double dx_ave = (dx.x() + dx.y() + dx.z()) / 3.0;
				double c_bulk = sqrt(((UI[8]) / (3 * (1 - 2 * UI[9]))) / rho_cur);
				p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur, dx_ave);
			}
			else {
				p_q[idx] = 0.;
			}
		}  // end loop over particles
	   
	
		WaveSpeed = dx / WaveSpeed;
		double delT_new = WaveSpeed.minComponent();
		new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
		if (flag->d_reductionVars->accStrainEnergy ||
			flag->d_reductionVars->strainEnergy) {
			new_dw->put(sum_vartype(se), lb->StrainEnergyLabel);
		}
	}
}

void HypoplasticB::carryForward(const PatchSubset* patches,
	const MPMMaterial* matl,
	DataWarehouse* old_dw,
	DataWarehouse* new_dw)
{
	for (int p = 0; p < patches->size(); p++) {
		const Patch* patch = patches->get(p);
		int dwi = matl->getDWIndex();
		ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

		// Carry forward the data common to all constitutive models
		// when using RigidMPM.
		// This method is defined in the ConstitutiveModel base class.
		carryForwardSharedData(pset, old_dw, new_dw, matl);

		/*
		// PVoidratio
		ParticleVariable<double>  pVoidRatio;

		new_dw->allocateAndPut(pVoidRatio, lb->pVoidRatioLabel, pset);

		ParticleSubset::iterator iter = pset->begin();
		for (; iter != pset->end(); iter++) {	
			particleIndex idx = *iter;

			pVoidRatio[idx] = 0.0;
		}
		*/

		// Carry forward the data local to this constitutive model
		std::vector<constParticleVariable<double> > ISVs(d_NINSV + 1);
		std::vector<ParticleVariable<double> > ISVs_new(d_NINSV + 1);

		for (int i = 0; i < d_NINSV; i++) {
			old_dw->get(ISVs[i], ISVLabels[i], pset);
			new_dw->allocateAndPut(ISVs_new[i], ISVLabels_preReloc[i], pset);
			ISVs_new[i].copyData(ISVs[i]);
		}

		// Don't affect the strain energy or timestep size
		new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());

		if (flag->d_reductionVars->accStrainEnergy ||
			flag->d_reductionVars->strainEnergy) {
			new_dw->put(sum_vartype(0.), lb->StrainEnergyLabel);
		}
	}

}

void HypoplasticB::addInitialComputesAndRequires(Task* task,
	const MPMMaterial* matl,
	const PatchSet*) const
{
	// Add the computes and requires that are common to all explicit
	// constitutive models.  The method is defined in the ConstitutiveModel
	// base class.
	const MaterialSubset* matlset = matl->thisMaterial();

	cout << "In add InitialComputesAnd" << endl;

	// Other constitutive model and input dependent computes and requires
	for (int i = 0; i < d_NINSV; i++) {
		task->computes(ISVLabels[i], matlset);
	}
}

void HypoplasticB::addComputesAndRequires(Task* task,
	const MPMMaterial* matl,
	const PatchSet* patches) const
{
	// Add the computes and requires that are common to all explicit
	// constitutive models.  The method is defined in the ConstitutiveModel
	// base class.
	const MaterialSubset* matlset = matl->thisMaterial();
	addSharedCRForHypoExplicit(task, matlset, patches);

	//task->computes(lb->pVoidRatioLabel, matlset);

	// Computes and requires for internal state data
	for (int i = 0; i < d_NINSV; i++) {
		task->requires(Task::OldDW, ISVLabels[i], matlset, Ghost::None);
		task->computes(ISVLabels_preReloc[i], matlset);
	}
}

void HypoplasticB::addComputesAndRequires(Task*,
	const MPMMaterial*,
	const PatchSet*,
	const bool) const
{
}

double HypoplasticB::computeRhoMicroCM(double pressure,
	const double p_ref,
	const MPMMaterial* matl,
	double temperature,
	double rho_guess)
{
	double rho_orig = matl->getInitialDensity();
	double p_gauge = pressure - p_ref;
	double rho_cur;
	double bulk = (UI[8]) / (3 * (1 - 2 * UI[9]));



	rho_cur = rho_orig / (1 - p_gauge / bulk);

	return rho_cur;

#if 1
	cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR HypoplasticB" << endl;
#endif
}

void HypoplasticB::computePressEOSCM(double rho_cur, double& pressure,
	double p_ref,
	double& dp_drho, double& tmp,
	const MPMMaterial* matl,
	double temperature)
{

	double bulk = (UI[8]) / (3 * (1 - 2 * UI[9]));
	double rho_orig = matl->getInitialDensity();

	double p_g = bulk * (1.0 - rho_orig / rho_cur);
	pressure = p_ref + p_g;
	dp_drho = bulk * rho_orig / (rho_cur*rho_cur);
	tmp = bulk / rho_cur;  // speed of sound squared

#if 1
	cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR HypoplasticB" << endl;
#endif
}

double HypoplasticB::getCompressibility()
{
	//TU MOZNA ZMIENIAC:
	return 1.0;
	//  return 1.0/((UI[8])/(3*(1-2*UI[9])));
}

void
HypoplasticB::getInputParameters(ProblemSpecP& ps)
{
	ps->getWithDefault("ei0_B", UI[0], 0.0);
	ps->getWithDefault("ed0_B", UI[1], 0.0);
	ps->getWithDefault("ec0_B", UI[2], 0.0);
	ps->getWithDefault("phic_B", UI[3], 0.0);
	ps->getWithDefault("hs_B", UI[4], 0.0);
	ps->getWithDefault("beta_B", UI[5], 0.0);
	ps->getWithDefault("n_B", UI[6], 0.0);
	ps->getWithDefault("alpha_B", UI[7], 0.0);
	ps->getWithDefault("E_B", UI[8], 0.0);
	ps->getWithDefault("v_B", UI[9], 0.0);
	ps->getWithDefault("epocz_B", UI[10], 0.0);
	ps->getWithDefault("nl_B", UI[11], 0.0);
	ps->getWithDefault("lchar_B", UI[12], 0.0);
	ps->getWithDefault("mean_pressure", UI[13], 0.0);
	ps->getWithDefault("phase", UI[14], 0.0);
	ps->getWithDefault("phase_change", UI[15], 0.0);
	ps->getWithDefault("Volumerate", UI[16], 0.0);
}

void
HypoplasticB::initializeLocalMPMLabels()
{
	vector<string> ISVNames;

	ISVNames.push_back("ei0_B");
	ISVNames.push_back("ed0_B");
	ISVNames.push_back("ec0_B");
	ISVNames.push_back("phic_B");
	ISVNames.push_back("hs_B");
	ISVNames.push_back("beta_B");
	ISVNames.push_back("n_B");
	ISVNames.push_back("alpha_B");
	ISVNames.push_back("E_B");
	ISVNames.push_back("v_B");
	ISVNames.push_back("epocz_B");
	ISVNames.push_back("nl_B");
	ISVNames.push_back("lchar_B");
	ISVNames.push_back("mean_pressure");
	ISVNames.push_back("phase");
	ISVNames.push_back("phase_change");
	ISVNames.push_back("Volumerate");

	for (int i = 0; i < d_NINSV; i++) {
		ISVLabels.push_back(VarLabel::create(ISVNames[i],
			ParticleVariable<double>::getTypeDescription()));
		ISVLabels_preReloc.push_back(VarLabel::create(ISVNames[i] + "+",
			ParticleVariable<double>::getTypeDescription()));
	}
}

void HypoplasticB::CalculateStress(int &nblk, int &ninsv, double &dt, double UI[], double stress[], double D[], double svarg[], double &USM)
//C**********************************************************************
//C     input arguments
//C     ===============
//C      NBLK       int                   Number of blocks to be processed
//C      NINSV      int                   Number of internal state vars
//C      DTARG      dp                    Current time increment
//C      UI       dp,ar(nprop)            User inputs
//C      D          dp,ar(6)              Strain increment
//C
//C     input output arguments
//C     ======================
//C      STRESS   dp,ar(6)                stress
//C      SVARG    dp,ar(ninsv)            state variables/statenew
//C
//C     output arguments
//C     ================
//C      USM      dp                      uniaxial strain modulus
//C
//C***********************************************************************
//C
//C      stresss and strains, plastic strain tensors
//C          11, 22, 33, 12, 23, 13
//C
//C***********************************************************************
{

	/*
ei0 - maximum void ratio
ed0 - void ratio at maximum densification
ec0 - critical void ratio
phic - critical angle of internal friction
hs - granular hardness
beta - stiffness coefficient
n - compression coefficient
alpha - pycontrophy coefficient
E - young modulus
v - poisson ratio
eini - initial void ratio
nl - non-local (0 - OFF; 1 - ON)
lchar - characterstic length
   */

	double eini = UI[10];
	double ei0 = UI[0];
	double ed0 = UI[1];
	double ec0 = UI[2];
	double phic = UI[3];
	double hs = UI[4];
	double beta = UI[5];
	double n = UI[6];
	double alpha = UI[7];
	double E = UI[8];
	double v = UI[9];

	double PI = 2 * asin(1.0);
	double R = PI / 180;
	double phi = phic * R;

	// Enforce stress
	if (stress[0] >= 0.0)
	{
		stress[0] = -1.0;
	}
	if (stress[1] >= 0.0)
	{
		stress[1] = -1.0;
	}
	if (stress[2] >= 0.0)
	{
		stress[2] = -1.0;
	}
	   	 
	double esd = svarg[10];
	double volumeRate = svarg[16];
	double trT = stress[0] + stress[1] + stress[2];
	double ps = -trT / 3.0;

	double a = 0.375;
	double b = 3.0;
	double c = 0.333333333333333;
	double d = 6.0;
	double c1 = (sqrt(a))*(3.0 - sin(phi)) / (sin(phi));
	double c2 = a * (3.0 + sin(phi)) / (sin(phi));
	double ei = ei0 * exp(-pow((3.0*ps / hs), n));
	double ed = ed0 * exp(-pow((3.0*ps / hs), n));
	double ec = ec0 * exp(-pow((3.0*ps / hs), n));
	double hi = (1.0 / pow(c1, 2.0)) + c - pow(((ei0 - ed0) / (ec0 - ed0)), alpha)*(1.0 / (c1*sqrt(b)));
	double fd = pow(((esd - ed) / (ec - ed)), alpha);
	double fs = (hs / (n*hi))*pow((ei / esd), beta)*((1.0 + ei) / ei)*pow((3.0*ps / hs), (1.0 - n));

			// Start calculate stress
			double T11 = stress[0] / trT;
			double T22 = stress[1] / trT;
			double T33 = stress[2] / trT;
			double T1212 = stress[3] / trT;
			double T2323 = stress[4] / trT;
			double T1313 = stress[5] / trT;

			double TSS1 = (stress[0] - trT / 3.0) / trT;
			double TSS2 = (stress[1] - trT / 3.0) / trT;
			double TSS3 = (stress[2] - trT / 3.0) / trT;
			double TSS12 = stress[3] / trT;
			double TSS23 = stress[4] / trT;
			double TSS13 = stress[5] / trT;

			double tr1 = pow(TSS1, 2.0) + pow(TSS12, 2.0) + pow(TSS13, 2.0);
			double tr2 = pow(TSS12, 2.0) + pow(TSS2, 2.0) + pow(TSS23, 2.0);
			double tr3 = pow(TSS13, 2.0) + pow(TSS23, 2.0) + pow(TSS3, 2.0);
			double tr4 = TSS1 * TSS12 + TSS2 * TSS12 + TSS13 * TSS23;
			double tr5 = TSS1 * TSS13 + TSS12 * TSS23 + TSS13 * TSS3;
			double tr6 = TSS12 * TSS13 + TSS2 * TSS23 + TSS23 * TSS3;
			double tr7 = TSS1 * TSS13 + TSS12 * TSS23 + TSS13 * TSS3;
			double tr8 = TSS12 * TSS13 + TSS2 * TSS23 + TSS23 * TSS3;
			double tr9 = pow(TSS13, 2.0) + pow(TSS23, 2.0) + pow(TSS3, 2.0);

			double trTs2 = tr1 + tr2 + tr3;
			double trTs3 = (TSS1*tr1 + TSS12 * tr4 + TSS13 * tr5) + (TSS12*tr4 + TSS2 * tr2 + TSS23 * tr6) + (TSS13*tr7 + TSS23 * tr8 + TSS3 * tr9);
			double PtsI = sqrt(pow(TSS1, 2.0) + pow(TSS2, 2.0) + pow(TSS3, 2.0) + 2.0*pow(TSS12, 2.0) + 2.0*pow(TSS23, 2.0) + 2.0*pow(TSS13, 2.0));

			double a1;
			double cos3o;
			if (trTs2 == 0.0)
			{
				a1 = 1 / c1;
			}
			else
			{
				cos3o = -sqrt(d)*trTs3 / pow((trTs2), 1.5);
				if (cos3o >= 1.0)
				{
					cos3o = 1.0;
				}
				if (cos3o <= -1.0)
				{
					cos3o = -1.0;
				}
				a1 = 1 / (c1 + (c2*(1.0 + cos3o)*PtsI));
			}

			double trTTD = T11 * D[0] + T22 * D[1] + T33 * D[2] + 2.0*T1212*D[3] + 2.0*T2323*D[4] + 2.0*T1313*D[5];

			double LTD1 = pow(a1, 2.0)*D[0] + T11 * trTTD;
			double LTD2 = pow(a1, 2.0)*D[1] + T22 * trTTD;
			double LTD3 = pow(a1, 2.0)*D[2] + T33 * trTTD;
			double LTD12 = pow(a1, 2.0)*D[3] + T1212 * trTTD;
			double LTD23 = pow(a1, 2.0)*D[4] + T2323 * trTTD;
			double LTD13 = pow(a1, 2.0)*D[5] + T1313 * trTTD;

			double NT1 = a1 * ((2.0*stress[0] - trT / 3.0) / (trT));
			double NT2 = a1 * ((2.0*stress[1] - trT / 3.0) / (trT));
			double NT3 = a1 * ((2.0*stress[2] - trT / 3.0) / (trT));
			double NT12 = a1 * (2.0*stress[3] / (trT));
			double NT23 = a1 * (2.0*stress[4] / (trT));
			double NT13 = a1 * (2.0*stress[5] / (trT));

			double dlooocB = pow(D[0], 2) + pow(D[1], 2) + pow(D[2], 2) + 2.0*pow(D[3], 2) + 2.0*pow(D[4], 2) + 2.0*pow(D[5], 2);
			double dloc = sqrt(dlooocB);

			double dT1 = fs * (LTD1 + fd * NT1*dloc);
			double dT2 = fs * (LTD2 + fd * NT2*dloc);
			double dT3 = fs * (LTD3 + fd * NT3*dloc);
			double dT12 = fs * (LTD12 + fd * NT12*dloc);
			double dT23 = fs * (LTD23 + fd * NT23*dloc);
			double dT13 = fs * (LTD13 + fd * NT13*dloc);

			stress[0] += dT1 * dt;
			stress[1] += dT2 * dt;
			stress[2] += dT3 * dt;
			stress[3] += dT12 * dt;
			stress[4] += dT23 * dt;
			stress[5] += dT13 * dt;

			//iNLB = iNLB + 1;

			double de = (1.0 + esd)*(D[0] + D[1] + D[2]);

			//Void ratio
			svarg[10] += de * dt;

			// mean pressure
			svarg[13] = stress[0] + stress[1] + stress[2];

			// phase solid
			svarg[14] = 0;
		

			// Void ratio cut off
			double phase_change = UI[15];
			if (volumeRate >= (1 + ei) / (1 + eini)) {

				if (phase_change) {
					stress[0] = 0;
					stress[1] = 0;
					stress[2] = 0;
					stress[3] = 0;
					stress[4] = 0;
					stress[5] = 0;

					// Cut off void ratio
					svarg[10] = ei;

					// mean pressure
					svarg[13] = stress[0] + stress[1] + stress[2];


					// Detect phase change
					svarg[14] = 1;
				}
			}

			// positive mean stress corresponds to extension
			// Tension cut off
			else if (trT >= 0) {
				if (phase_change) {
					stress[0] = 0;
					stress[1] = 0;
					stress[2] = 0;
					stress[3] = 0;
					stress[4] = 0;
					stress[5] = 0;

					svarg[10] = esd;

					// mean pressure
					svarg[13] = stress[0] + stress[1] + stress[2];

					// phase solid
					svarg[14] = 1;
				}
			}
			   		 	  
			if (svarg[10] <= ed) {
				// Cut off void ratio
				svarg[10] = ed;
			}

	
	/////////////////////////////////////
	double Kmod = (E) / (3 * (1 - 2 * v));
	double Gmod = (E) / (2 * (1 + v));
	double Factor = 1.0;
	//factor is  a quick fix, as otherwise the USM is too low and predicted stable time step is way too high
	USM = Factor * (Gmod + 0.3*Kmod) / 3.0;

}

void HypoplasticB::CheckModel(double UI[])
{
	if (UI[0] < 0.0) cerr << "Maximum void ratio in the HypoplasticB Model equal to: " << UI[0] << " This will cause the code to malfuncion. Any results obtained are invalid." << endl;
	if (UI[1] < 0.0) cerr << "Void ratio at maximum densification in the HypoplasticB Model equal to: " << UI[1] << " This will cause the code to malfuncion. Any results obtained are invalid." << endl;
	if (UI[2] < 0.0) cerr << "Critical void ratio in the HypoplasticB Model equal to: " << UI[2] << " This will cause the code to malfuncion. Any results obtained are invalid." << endl;
	if (UI[3] < 0.0) cerr << "Critical angle of internal friction in the HypoplasticB Model equal to: " << UI[3] << " This will cause the code to malfuncion. Any results obtained are invalid." << endl;
	if (UI[4] < 0.0) cerr << "Granular hardness in the HypoplasticB Model equal to: " << UI[4] << " This will cause the code to malfuncion. Any results obtained are invalid." << endl;
	if (UI[5] < 0.0) cerr << "Stiffness coefficient in the HypoplasticB Model equal to: " << UI[5] << " This will cause the code to malfuncion. Any results obtained are invalid." << endl;
	if (UI[6] < 0.0) cerr << "Compression coefficient in the HypoplasticB Model equal to: " << UI[6] << " This will cause the code to malfuncion. Any results obtained are invalid." << endl;
	if (UI[7] < 0.0) cerr << "Pycontrophy coefficient in the HypoplasticB Model equal to: " << UI[7] << " This will cause the code to malfuncion. Any results obtained are invalid." << endl;
	if (UI[8] <= 0.0) cerr << "Young modulus in the HypoplasticB Model equal to: " << UI[8] << " This will cause the code to malfuncion. Any results obtained are invalid." << endl;
	if (UI[9] <= 0.0) cerr << "Poisson ratio in the HypoplasticB Model equal to: " << UI[9] << " This will cause the code to malfuncion. Any results obtained are invalid." << endl;
}

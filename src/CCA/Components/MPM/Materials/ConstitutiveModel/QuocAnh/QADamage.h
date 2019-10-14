/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

 //  QADamage.h
 //  class ConstitutiveModel ConstitutiveModel data type -- 3D -
 //  holds ConstitutiveModel
 //  information for the FLIP technique:
 //    This is for Compressible NeoHookean materials
 //    Features:
 //      Usage:


#ifndef __QADamage_CONSTITUTIVE_MODEL_H__
#define __QADamage_CONSTITUTIVE_MODEL_H__

namespace Uintah {
	// Structures for Plasticitity

	struct QADamageStateData {
		double Alpha;
	};
	class TypeDescription;
	const TypeDescription* fun_getTypeDescription(QADamageStateData*);
}

#include <Core/Util/Endian.h>

namespace Uintah {
	inline void swapbytes(Uintah::QADamageStateData& d)
	{
		swapbytes(d.Alpha);
	}
} // namespace Uintah

#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ImplicitCM.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/MPMEquationOfState.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Math/Matrix3.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <cmath>
#include <vector>

namespace Uintah {
	// Classes needed by QADamage
	class TypeDescription;

	class QADamage : public ConstitutiveModel, public ImplicitCM {

		///////////////
		// Variables //
		///////////////
	public:

		// Add State variables
		int d_INPUT;
		double UI[100];
		double rinit[100];

		std::vector<const VarLabel*> ISVLabels;
		std::vector<const VarLabel*> ISVLabels_preReloc;

		// Basic Requirements //
		////////////////////////
		// Create datatype for storing model parameters
		struct CMData {
			double Bulk;
			double tauDev;
			// For Plasticity
			double FlowStress;
			double K;
			double Alpha;

			// For Damage
			double length;
			double mesh;
			double kRatio;
			double tensile;
			double Gf;
			double nNonlocal;
			double a;
			double b;
			double beta;
			double ref_eqstrain;
		};

		struct YieldDistribution {
			std::string dist;
			double range;
			int seed;
		};

		const VarLabel* bElBarLabel;
		const VarLabel* bElBarLabel_preReloc;

		const VarLabel* pDeformRateLabel;
		const VarLabel* pDeformRateLabel_preReloc;


		// Plasticity Requirements //
		/////////////////////////////
		const VarLabel* pPlasticStrainLabel;
		const VarLabel* pPlasticStrainLabel_preReloc;
		const VarLabel* pYieldStressLabel;
		const VarLabel* pYieldStressLabel_preReloc;

	protected:

		// Basic Requirements //
		////////////////////////
		CMData d_initialData;
		bool d_useModifiedEOS;
		int d_8or27;

		// MohrColoumb options
		double d_friction_angle;  // Assumed to come in degrees
		double d_tensile_cutoff;  // Fraction of the cohesion at which
								  // tensile failure occurs

		//__________________________________
		//  Plasticity
		bool d_usePlasticity;
		YieldDistribution d_yield;

		// Initial stress state
		bool d_useInitialStress;
		double d_init_pressure;  // Initial pressure

		// Model factories
		//bool d_useEOSFactory;
		MPMEquationOfState* d_eos;

		///////////////
		// Functions //
		///////////////
	private:
		// Prevent copying of this class
		// copy constructor
		QADamage& operator=(const QADamage &cm);

		// Plasticity requirements
		//friend const TypeDescription* fun_getTypeDescriptiont(StateData*);

	public:
		// constructor
		QADamage(ProblemSpecP& ps, MPMFlags* flag, bool plas, bool dam);

		// specifcy what to output from the constitutive model to an .xml file
		virtual void outputProblemSpec(ProblemSpecP& ps, bool output_cm_tag = true);

		// clone
		QADamage* clone();

		// destructor
		virtual ~QADamage();

		// carry forward CM data for RigidMPM
		virtual void carryForward(const PatchSubset* patches,
			const MPMMaterial* matl,
			DataWarehouse* old_dw,
			DataWarehouse* new_dw);

		virtual void initializeCMData(const Patch* patch,
			const MPMMaterial* matl,
			DataWarehouse* new_dw);


		// Scheduling Functions //
		//////////////////////////
		virtual void addComputesAndRequires(Task* task,
			const MPMMaterial* matl,
			const PatchSet* patches) const;

		virtual void addComputesAndRequires(Task* task,
			const MPMMaterial* matl,
			const PatchSet* patches,
			const bool recursion,
			const bool schedPar = true) const;

		virtual void addInitialComputesAndRequires(Task* task,
			const MPMMaterial* matl,
			const PatchSet* patches) const;


		// Compute Functions //
		///////////////////////
		// main computation of pressure from constitutive model's equation of state
		virtual void computePressEOSCM(double rho_m, double& press_eos,
			double p_ref,
			double& dp_drho, double& ss_new,
			const MPMMaterial* matl,
			double temperature);

		// main computation of density from constitutive model's equation of state
		virtual double computeRhoMicroCM(double pressure,
			const double p_ref,
			const MPMMaterial* matl,
			double temperature,
			double rho_guess);

		// compute stable timestep for this patch
		virtual void computeStableTimeStep(const Patch* patch,
			const MPMMaterial* matl,
			DataWarehouse* new_dw);

		// compute stress at each particle in the patch
		virtual void computeStressTensor(const PatchSubset* patches,
			const MPMMaterial* matl,
			DataWarehouse* old_dw,
			DataWarehouse* new_dw);

		// Damage specific CST for solver
		virtual void computeStressTensorImplicit(const PatchSubset* patches,
			const MPMMaterial* matl,
			DataWarehouse* old_dw,
			DataWarehouse* new_dw,
			Solver* solver,
			const bool);


		// Helper Functions //
		//////////////////////
		virtual void addParticleState(std::vector<const VarLabel*>& from,
			std::vector<const VarLabel*>& to);

		// Returns the compressibility of the material
		virtual double getCompressibility();


		virtual void addSplitParticlesComputesAndRequires(Task* task,
			const MPMMaterial* matl,
			const PatchSet* patches);

		virtual void splitCMSpecificParticleData(const Patch* patch,
			const int dwi,
			const int fourOrEight,
			ParticleVariable<int> &prefOld,
			ParticleVariable<int> &prefNew,
			const unsigned int oldNumPar,
			const unsigned int numNewPartNeeded,
			DataWarehouse* old_dw,
			DataWarehouse* new_dw);

	private:

		void getYieldStressDistribution(ProblemSpecP& ps);

		void setYieldStressDistribution(const QADamage* cm);

		void createPlasticityLabels();

	protected:
		// compute stress at each particle in the patch
		void computeStressTensorImplicit(const PatchSubset* patches,
			const MPMMaterial* matl,
			DataWarehouse* old_dw,
			DataWarehouse* new_dw);

		/*! Compute tangent stiffness matrix */
		void computeTangentStiffnessMatrix(const Matrix3& sigDev,
			const double&  mubar,
			const double&  J,
			const double&  bulk,
			double D[6][6]);
		/*! Compute BT*Sig*B (KGeo) */
		void BnlTSigBnl(const Matrix3& sig, const double Bnl[3][24],
			double BnTsigBn[24][24]) const;

		/*! Compute K matrix */
		void computeStiffnessMatrix(const double B[6][24],
			const double Bnl[3][24],
			const double D[6][6],
			const Matrix3& sig,
			const double& vol_old,
			const double& vol_new,
			double Kmatrix[24][24]);

		double computeDensity(const double& rho_orig,
			const double& pressure);

		void computePressure(const double& rho_orig,
			const double& rho_cur,
			double& pressure,
			double& dp_drho,
			double& csquared);
	};
} // End namespace Uintah

#endif  // __QADamage_CONSTITUTIVE_MODEL_H__

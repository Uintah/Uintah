#ifndef __VISCOSCRAM_FOR_BINDER_H__
#define __VISCOSCRAM_FOR_BINDER_H__

#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <math.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

#ifndef M_PI
# define M_PI           3.14159265358979323846  /* pi */
#endif

namespace Uintah {

/**************************************

CLASS
   ViscoScramForBinderForBinder
   
   Implementation of ViscoScramForBinder that is appropriate for the binder
   in PBX materials.

GENERAL INFORMATION

   ViscoScramForBinderForBinder.h

   Biswajit Banerjee, Scott Bardenhagen, Jim Guilkey
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2003 University of Utah

KEYWORDS

   Viscoelasticity, Statistical Crack Mechanics

DESCRIPTION
   
   The basic model for the binder is based on the paper by
   
   Mas, Clements, Blumenthal, Cady, Gray and Liu, 2001,
   "A Viscoelastic Model For PBX Binders"
   in Shock Compression of Condensed Matter - 2001, pp. 661-664.

   The shear modulus is given by a series of generalized Maxwell
   elements, the relaxation times are based on a time-temperature
   superposition principle of the Willimas-Landel-Ferry type.

   The bulk modulus is an input parameter that is assumed to be
   independent of strain rate.

   The ViscoScram model is based on the paper by
  
   Bennett, Haberman, Johnson, Asay and Henson, 1998,
   "A Constitutive Model for the Shock Ignition and Mechanical
    Response of High Explosives"
   J. Mech. Phys. Solids, v. 46, n. 12, pp. 2303-2322.

   Integration of stress is done using a fourth order Runge-Kutta
   approximation.

   Crack evolution is also determined using a Runge-Kutta scheme.

   Options are available to turn cracks on or off.

WARNING
  
   Only isotropic materials, linear viscoelasticity, small strains.
   No plasticity.

****************************************/
  class ViscoScramForBinder : public ConstitutiveModel {

  private:

    bool d_useModifiedEOS;
    bool d_doCrack;    

  public:

    // Material constants
    struct CMData {
      double  bulkModulus;
      int     numMaxwellElements;
      double* shearModulus; 

      double  reducedTemperature_WLF;
      double  constantA1_WLF;
      double  constantA2_WLF;
      double  constantB1_RelaxTime;
      int     constantB2_RelaxTime;

      double  initialSize_Crack;
      double  powerValue_Crack;
      double  initialRadius_Crack;
      double  maxGrowthRate_Crack;
      double  stressIntensityF_Crack;
      double  frictionCoeff_Crack;
    };

    struct Statedata {

      Matrix3 sigDev[22];

      //double numElements;
      //Matrix3* sigDev;

      //Statedata() 
      //{
      //  numElements = 0;
      //}

      //Statedata(const Statedata& st)
      //{
      //  numElements = st.numElements;
      //  int nn = (int) numElements;
      //  sigDev = new Matrix3[nn];
      //  for (int ii = 0; ii < nn; ++ii) {
      //    sigDev[ii] = st.sigDev[ii];
      //  }
      //}
      
      //~Statedata()
      //{
      //  delete sigDev;
      //}
    };

  private:

    friend const Uintah::TypeDescription* fun_getTypeDescription(Statedata*);

    CMData d_initialData;

    // Prevent copying of this class
    ViscoScramForBinder(const ViscoScramForBinder &cm);
    ViscoScramForBinder& operator=(const ViscoScramForBinder &cm);

  public:

    // constructors
    ViscoScramForBinder(ProblemSpecP& ps, MPMLabel* lb, int n8or27);
       
    // destructor
    virtual ~ViscoScramForBinder();

    // compute stable timestep for this patch
    virtual void computeStableTimestep(const Patch* patch,
				       const MPMMaterial* matl,
				       DataWarehouse* new_dw);

    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
				     const MPMMaterial* matl,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw);

    virtual void computeStressTensor(const PatchSubset* patches,
				     const MPMMaterial* matl,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw,
				     Solver* solver,
				     const bool recursion);
	 
    // initialize  each particle's constitutive model data
    virtual void initializeCMData(const Patch* patch,
				  const MPMMaterial* matl,
				  DataWarehouse* new_dw);

    virtual void allocateCMData(DataWarehouse* new_dw,
				ParticleSubset* subset,
				map<const VarLabel*, ParticleVariableBase*>* newState);


    virtual void addInitialComputesAndRequires(Task* task,
					       const MPMMaterial* matl,
					       const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patches,
					const bool recursion) const;

    virtual double computeRhoMicroCM(double pressure,
				     const double p_ref,
				     const MPMMaterial* matl);

    virtual void computePressEOSCM(double rho_m, double& press_eos,
				   double p_ref,
				   double& dp_drho, double& ss_new,
				   const MPMMaterial* matl);

    virtual double getCompressibility();

    // class function to read correct number of parameters
    // from the input file
    static void readParameters(ProblemSpecP ps, double *p_array);

    // class function to write correct number of parameters
    // from the input file, and create a new object
    static ConstitutiveModel* readParametersAndCreate(ProblemSpecP ps);

    // member function to read correct number of parameters
    // from the input file, and any other particle information
    // need to restart the model for this particle
    // and create a new object
    static ConstitutiveModel* 
      readRestartParametersAndCreate(ProblemSpecP ps);

    virtual void addParticleState(std::vector<const VarLabel*>& from,
				  std::vector<const VarLabel*>& to);
    // class function to create a new object from parameters
    static ConstitutiveModel* create(double *p_array);

    const VarLabel* pStatedataLabel;
    const VarLabel* pStatedataLabel_preReloc;

  private:

    // Runge-Kutta for crack radius
    double doRungeKuttaForCrack(double (ViscoScramForBinder::*fptr)(double, 
                                                                    double, 
                                                                    double),
				double  y, 
                                double  h,
				double  K0, 
                                double  sigma,
				double* kk) ;
    // Crack growth equations
    double crackGrowthEqn1(double c, double K0, double sigma) ;
    double crackGrowthEqn2(double c, double K0, double sigma) ;

    // Runge-Kutta for deviatoric stress
    void doRungeKuttaForStress(void (ViscoScramForBinder::*fptr)(Matrix3*, 
                                                                 double, 
                                                                 double*, 
                                                                 double*, 
				                                 Matrix3&, 
                                                                 double, 
                                                                 Matrix3*), 
			       Matrix3* y_n, 
                               double   h, 
			       double*  rkc, 
                               double   c,
			       double*  G_n, 
                               double*  Tau_n, 
			       Matrix3& eDot, 
                               double   cDot,
			       Matrix3* y_rk);

    // Deviatoric stress equations
    void stressEqnWithCrack(Matrix3* S_n, 
                            double   c,
			    double*  G_n, 
                            double*  Tau_n,
			    Matrix3& eDot, 
                            double   cDot, 
                            Matrix3* k_n);

    void stressEqnWithoutCrack(Matrix3* S_n, 
                               double   c,
		               double*  G_n, 
                               double*  Tau_n,
		               Matrix3& eDot, 
                               double   cDot, 
                               Matrix3* k_n);

    // Solve the stress equation using a fourth-order Runge-Kutta scheme
    void doRungeKuttaForStressAlt(void (ViscoScramForBinder::*fptr)
				       (int, Matrix3&, double, double, 
				        Matrix3&, Matrix3&, double,
				        double, Matrix3&, double, 
				        Matrix3&), 
				  Matrix3* y_n,
				  double h, 
				  double* rkc, 
				  double c,
				  double* G_n, 
				  double* RTau_n, 
				  Matrix3& DPrime,
				  double cDot,
				  Matrix3* y_rk);
      void stressEqnWithCrack(int index,
			      Matrix3& S_n,
			      double c,
			      double G,
			      Matrix3& sumS_nOverTau_n,
			      Matrix3& S,
			      double G_n,
			      double RTau_n,
			      Matrix3& DPrime,
			      double cDot,
			      Matrix3& k_n);
      void stressEqnWithoutCrack(int index,
				 Matrix3& S_n,
				 double ,
				 double ,
				 Matrix3& ,
				 Matrix3& ,
				 double G_n,
				 double RTau_n,
				 Matrix3& DPrime,
				 double ,
				 Matrix3& k_n);
  };

} // End namespace Uintah
      
namespace SCIRun {
  void swapbytes( Uintah::ViscoScramForBinder::Statedata& d);
} // namespace SCIRun

#endif  // __VISCOSCRAM_FOR_BINDER_H__ 


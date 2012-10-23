/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef __VISCOSCRAM_FOR_BINDER_H__
#define __VISCOSCRAM_FOR_BINDER_H__

#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <cmath>
#include <vector>

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
    //ViscoScramForBinder(const ViscoScramForBinder &cm);
    ViscoScramForBinder& operator=(const ViscoScramForBinder &cm);

  public:

    // constructors
    ViscoScramForBinder(ProblemSpecP& ps, MPMLabel* lb, int n8or27);
    ViscoScramForBinder(const ViscoScramForBinder* cm);
       
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

    // initialize  each particle's constitutive model data
    virtual void initializeCMData(const Patch* patch,
                                  const MPMMaterial* matl,
                                  DataWarehouse* new_dw);


    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb) const;

    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* subset,
                                   map<const VarLabel*, ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw);


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


    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);

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


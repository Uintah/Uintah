/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef __VISCOSCRAM_FULL_CONSTITUTIVE_MODEL_H__
#define __VISCOSCRAM_FULL_CONSTITUTIVE_MODEL_H__

#include "ViscoScram.h"
#include <Core/Math/Matrix3.h>
#include <Core/Math/FastMatrix.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <cmath>
#include <vector>

namespace Uintah {

  class MPMLabel;
  class MPMFlags;

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class ViscoSCRAMHotSpot
    \brief Extended version of ViscoSCRAM including hotspot calculation
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2000 University of Utah

    Refernces:\n

    Bennett, J.G. et al., 1998, "A constitutive model for the non-shock 
    ignition and mechanical response of high explosives," J. Mech. Phys. Solids
    v. 46, n. 12, pp. 2303-2322.\n

    Hackett, R.M. and Bennett, J.G., 2000, "An implicit finite element 
    material model for energetic composite materials," Int. J. Numer. Meth.
    Engng., v. 49, pp. 1191-1209.\n

    Dienes, J.K. and Kershner, J.D., 2001, "Crack dynamics and explosive 
    burn via generalized coordinates," J. Compter-Aided Materials Design,
    v. 7, pp. 217-237.
  */
  /////////////////////////////////////////////////////////////////////////////

  class ViscoSCRAMHotSpot : public ViscoScram {

  public:

    class FVector {

    public:

      int nn;          // number of maxwell elements
      double a;        // can be either c or cdot
      Matrix3 b_n[5];  // can be either s_n or sdot_n
      
      FVector()
        {
          Matrix3 zero(0.0);
          nn = 5;
          a = 0.0;
          for (int ii = 0; ii < nn; ++ii) b_n[ii] = zero;
        }

      FVector(double aa, Matrix3* bb_n)
        {
          nn = 5;
          a = aa;
          for (int ii = 0; ii < nn; ++ii) b_n[ii] = bb_n[ii];
        }

      ~FVector()
        {
        }

      FVector
        operator+(const FVector& fv) const
        {
          FVector fv_new;
          if (fv.nn == nn) {
            fv_new.a = a + fv.a;
            for (int ii = 0; ii < nn; ++ii) fv_new.b_n[ii] = b_n[ii]+fv.b_n[ii];
          } else {
            fv_new.a = a; 
            for (int ii = 0; ii < nn; ++ii) fv_new.b_n[ii] = b_n[ii];
          }
          return fv_new;
        }

      FVector&
        operator+=(const FVector& fv) 
        {
          if (fv.nn == nn) {
            a += fv.a;
            for (int ii = 0; ii < nn; ++ii) b_n[ii] += fv.b_n[ii];
          }
          return *this;
        }

      FVector
        operator*(double val) const
        {
          FVector fv_new;
          fv_new.a = a*val;
          for (int ii = 0; ii < nn; ++ii) fv_new.b_n[ii] = b_n[ii]*val;
          return fv_new;
        }

      FVector&
        operator*=(double val) 
        {
          a *= val;
          for (int ii = 0; ii < nn; ++ii) b_n[ii] *= val;
          return *this;
        }
    };
 
    struct MatConstData {
      double Chi;   // fraction of work converted into heat
      double delH;
      double Z;
      double EoverR;
      double mu_d;  // dynamic coefficient of friction
      double vfHE;  // volume fraction of HE
    };

    const VarLabel* pHotSpotT1Label;
    const VarLabel* pHotSpotT2Label;
    const VarLabel* pHotSpotPhi1Label;
    const VarLabel* pHotSpotPhi2Label;
    const VarLabel* pChemHeatRateLabel;

    const VarLabel* pHotSpotT1Label_preReloc;
    const VarLabel* pHotSpotT2Label_preReloc;
    const VarLabel* pHotSpotPhi1Label_preReloc;
    const VarLabel* pHotSpotPhi2Label_preReloc;
    const VarLabel* pChemHeatRateLabel_preReloc;

  private:

    MatConstData d_matConst;

    /*! Prevent assignment of this class */
    ViscoSCRAMHotSpot& operator=(const ViscoSCRAMHotSpot &cm);

  public:
    // constructors
    ViscoSCRAMHotSpot(ProblemSpecP& ps, MPMFlags* flag);
    ViscoSCRAMHotSpot(const ViscoSCRAMHotSpot* cm);
       
    // destructor
    virtual ~ViscoSCRAMHotSpot();

    ViscoSCRAMHotSpot* clone();

    /*! Computes and requires for initialization of history variables */
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;

    /*! initialize  each particle's constitutive model data */
    virtual void initializeCMData(const Patch* patch,
                                  const MPMMaterial* matl,
                                  DataWarehouse* new_dw);

    /*! Set up data required by and computed in computeStressTensor */
    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;

    /*! Set up data required by and computed in computeStressTensor 
        for implicit methods */
    virtual void addComputesAndRequires(Task* ,
                                        const MPMMaterial* ,
                                        const PatchSet* ,
                                        const bool ) const
    {
    }

    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);


    /*! carry forward CM data (computed in computeStressTensor) for RigidMPM */
    virtual void carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

    /*! Set up data required in the particle conversion process */
    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb) const;

    /*! Copy data from the delset to the addset in the particle 
      conversion process */
    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* subset,
      map<const VarLabel*, ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw);

    /*! Add the particle data that have to be saved at the end of each 
      timestep */
    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);

  protected:

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Compute K_I */
    //
    ///////////////////////////////////////////////////////////////////////////
    double computeK_I(double c, double sigEff);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Compute K_{0mu} */
    //
    ///////////////////////////////////////////////////////////////////////////
    double computeK_0mu(double c, double sig_m);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Compute K^' */
    //
    ///////////////////////////////////////////////////////////////////////////
    double computeK_prime(double c, double sig_m);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Compute K_1 */
    //
    ///////////////////////////////////////////////////////////////////////////
    double computeK_1(double c, double sig_m);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Compute cdot */
    //
    ///////////////////////////////////////////////////////////////////////////
    double computeCdot(const Matrix3& s, double sig_m, double c, double vres);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Compute sdot_n */
    //
    ///////////////////////////////////////////////////////////////////////////
    Matrix3 computeSdot_mw(const Matrix3& edot, const Matrix3& s, Matrix3* s_n, 
                           double* G_n, double c, double cdot,
                           int mwelem, int numMaxwellElem);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Evaluate the quantities cdot and sdot_n  */
    //
    ///////////////////////////////////////////////////////////////////////////
    FVector evaluateRateEquations(const FVector& Y, const Matrix3& edot,
                                  double sig_m, double* G_n, double vres);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Solve an ordinary differential equation of the form
      dy/dt = f(y,t)
      using a fourth-order Runge-Kutta method
      between t=T and t=T+delT (h = delT) */
    //
    ///////////////////////////////////////////////////////////////////////////
    FVector integrateRateEquations(const FVector& Y_old, const Matrix3& edot, 
                                   double sig_m, double* G_n, double vres, 
                                   double delT, double& cdot);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Compute viscous work rate */
    //
    ///////////////////////////////////////////////////////////////////////////
    double computeViscousWorkRate(int numElem, Matrix3* s_n_new, double* G_n);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Compute cracking damage work rate */
    //
    ///////////////////////////////////////////////////////////////////////////
    double computeCrackingWorkRate(int numElem, double c_new, Matrix3* s_n_new,
                                   const Matrix3& edot, const Matrix3& s_new, 
                                   double* G_n, double cdot);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Compute bulk chemicak heating rate */
    //
    ///////////////////////////////////////////////////////////////////////////
    double computeChemicalHeatRate(double rho, double T_old);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Compute the conduction K matrix for the hotspot model */
    //
    ///////////////////////////////////////////////////////////////////////////
    void computeHotSpotKmatrix(double y1, double y2, double k,
                               FastMatrix& KK);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Compute the heat capacity matrix C for the hotspot model */
    //
    ///////////////////////////////////////////////////////////////////////////
    void computeHotSpotCmatrix(double y1, double y2, double rho, 
                               double Cv, FastMatrix& CC);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Compute the chemical heat rate matrix Qdot for the hotspot model */
    //
    ///////////////////////////////////////////////////////////////////////////
    void computeHotSpotQdotmatrix(double y1, double y2, double rho, double mu_d,
                                  double sig_m, double edotmax, 
                                  double T1, double T2, double* Qdot);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Compute the rate of temperature change at the hot spot */
    //
    ///////////////////////////////////////////////////////////////////////////
    void evaluateTdot(double* T, FastMatrix& k, FastMatrix& C, 
                      double y1, double y2, double rho, double mu_d,
                      double sig_m, double edotmax, double* Tdot);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Compute the increment of temperature at the hot spot using a 
      Fourth-order Runge-Kutta scheme */
    //
    ///////////////////////////////////////////////////////////////////////////
    void updateHotSpotTemperature(double* T, double y1, double y2, 
                                  double kappa, double rho, 
                                  double Cv, double mu_d,
                                  double sig_m, double edotmax, double delT);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Evaluate the hot spot model */
    //
    ///////////////////////////////////////////////////////////////////////////
    void evaluateHotSpotModel(double sig_m, const Matrix3& sig, 
                              const Matrix3& edot, double* T,
                              double kappa, double rho, double Cv, double delT);

    //////////////////////////////////////////////////////////////////////////
    //
    /*! Express a stress tensor in terms of a new set of bases (with the vector 
      e1Prime being the new direction of e1 ) */
    // 
    //////////////////////////////////////////////////////////////////////////
    Matrix3 stressInRotatedBasis(const Matrix3& sig, Vector& e1Prime);
  };

} // End namespace Uintah

#endif  // __VISCOSCRAM_FULL_CONSTITUTIVE_MODEL_H__ 


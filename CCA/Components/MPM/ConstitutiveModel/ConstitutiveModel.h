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


#ifndef __CONSTITUTIVE_MODEL_H__
#define __CONSTITUTIVE_MODEL_H__

#include <Core/Grid/Variables/ComputeSet.h>
#include <vector>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/Array3.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/LinearInterpolator.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Math/FastMatrix.h>
#include <CCA/Components/MPM/MPMFlags.h>


namespace Uintah {

  class Task;
  class Patch;
  class VarLabel;
  class MPMLabel;
  class MPMFlags;
  class MPMMaterial;
  class DataWarehouse;
  class ParticleSubset;
  class ParticleVariableBase;

  //////////////////////////////////////////////////////////////////////////
  /*!
    \class ConstitutiveModel
   
    \brief Base class for contitutive models.

    \author Steven G. Parker \n
    Department of Computer Science \n
    University of Utah \n
    Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n
    Copyright (C) 2000 SCI Group \n

    Long description...
  */
  //////////////////////////////////////////////////////////////////////////

  class ConstitutiveModel {
  public:
         
    ConstitutiveModel(MPMFlags* MFlag);
    ConstitutiveModel(const ConstitutiveModel* cm);
    virtual ~ConstitutiveModel();

    virtual void outputProblemSpec(ProblemSpecP& ps,
                                   bool output_cm_tag = true) = 0;

         
    // Basic constitutive model calculations
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);
                                     
    virtual void computeStressTensorImplicit(const PatchSubset* patches,
                                             const MPMMaterial* matl,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw);

    ///////////////////////////////////////////////////////////////////////
    /*! Initial computes and requires for the constitutive model */
    ///////////////////////////////////////////////////////////////////////
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;

    ///////////////////////////////////////////////////////////////////////
    /*! Initialize the variables used in the CM */
    ///////////////////////////////////////////////////////////////////////
    virtual void initializeCMData(const Patch* patch,
                                  const MPMMaterial* matl,
                                  DataWarehouse* new_dw) = 0;

    ///////////////////////////////////////////////////////////////////////
    /*! Set up the computes and requires for the task that computes the
        stress tensor and associated kinematic and thermal quantities */
    ///////////////////////////////////////////////////////////////////////
    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        const bool recursion,
                                        const bool SchedParent) const;

    virtual void scheduleCheckNeedAddMPMMaterial(Task* task,
                                                 const MPMMaterial* matl,
                                                 const PatchSet* patches) const;

    // Determine if addition of an acceptor material is needed
    virtual void checkNeedAddMPMMaterial(const PatchSubset* patches,
                                         const MPMMaterial* matl,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw);

    /////////////////////////////////////////////////////////////////
    /*! Add particle conversion related requires to the task graph */
    /////////////////////////////////////////////////////////////////
    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb) const;

    /////////////////////////////////////////////////////////////////
    /*! Copy the data from the particle to be deleted to the particle
        to be added */
    /////////////////////////////////////////////////////////////////
    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* addset,
                          map<const VarLabel*, ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw) = 0;

    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to) = 0;


    ////////////////////////////////////////////////////////////////////////
    /*! Carry forward CM variables for RigidMPM */
    ////////////////////////////////////////////////////////////////////////
    virtual void carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

    virtual double computeRhoMicroCM(double pressure,
                                     const double p_ref,
                                     const MPMMaterial* matl,
                                     double temperature,
                                     double rho_guess) = 0;

    virtual void computePressEOSCM(double rho_m, double& press_eos,
                                   double p_ref,
                                   double& dp_drho, double& ss_new,
                                   const MPMMaterial* matl, 
                                   double temperature) = 0;

    virtual double getCompressibility() = 0;

    virtual Vector getInitialFiberDir();

    double computeRhoMicro(double press,double gamma,
                           double cv, double Temp, double rho_guess);
         
    void computePressEOS(double rhoM, double gamma,
                         double cv, double Temp,
                         double& press, double& dp_drho,
                         double& dp_de);

    //////////
    // Convert J-integral into stress intensity for hypoelastic materials 
    // (for FRACTURE)
    virtual void ConvertJToK(const MPMMaterial* matl,const string& stressState,
                    const Vector& J,const double& C,const Vector& V,Vector& SIF);

    //////////                       
    // Detect if crack propagates and the direction (for FRACTURE)
    virtual short CrackPropagates(const double& Vc,const double& KI,
                                  const double& KII,double& theta);


    virtual void addRequiresDamageParameter(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patches) const;

    virtual void getDamageParameter(const Patch* patch, 
                                    ParticleVariable<int>& damage, int dwi,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw);

    inline void setWorld(const ProcessorGroup* myworld)
      {
        d_world = myworld;
      }

    inline void setSharedState(SimulationState* sharedState)
      {
        d_sharedState = sharedState;
      }

    // Make a clone of the constitutive model

    virtual ConstitutiveModel* clone() = 0;

    void computeDeformationGradientFromDisplacement(
                                           constNCVariable<Vector> gDisp,
                                           ParticleSubset* pset,
                                           constParticleVariable<Point> px,
                                           constParticleVariable<Vector> psize,
                                           ParticleVariable<Matrix3> &Fnew,
                                           constParticleVariable<Matrix3> &Fold,
                                           Vector dx,
                                           ParticleInterpolator* interp);

    void computeDeformationGradientFromVelocity(
                                           constNCVariable<Vector> gVel,
                                           ParticleSubset* pset,
                                           constParticleVariable<Point> px,
                                           constParticleVariable<Vector> psize,
                                           constParticleVariable<Matrix3> Fold,
                                           ParticleVariable<Matrix3> &Fnew,
                                           Vector dx,
                                           ParticleInterpolator* interp,
                                           const double& delT);

    void computeDeformationGradientFromTotalDisplacement(
                                           constNCVariable<Vector> gDisp,
                                           ParticleSubset* pset,
                                           constParticleVariable<Point> px,
                                           ParticleVariable<Matrix3> &Fnew,
                                           constParticleVariable<Matrix3> &Fold,
                                           Vector dx,
                                           constParticleVariable<Vector> psize,
                                           ParticleInterpolator* interp);
                                                                                
    void computeDeformationGradientFromIncrementalDisplacement(
                                           constNCVariable<Vector> IncDisp,
                                           ParticleSubset* pset,
                                           constParticleVariable<Point> px,
                                           constParticleVariable<Matrix3> Fold,
                                           ParticleVariable<Matrix3> &Fnew,
                                           Vector dx,
                                           constParticleVariable<Vector> psize,
                                           ParticleInterpolator* interp);
  protected:

    inline void computeVelocityGradient(Matrix3& velGrad,
                                        vector<IntVector>& ni,
                                        vector<Vector>& d_S,
                                        const double* oodx, 
                                        constNCVariable<Vector>& gVelocity)
      {
          for(int k = 0; k < flag->d_8or27; k++) {
            const Vector& gvel = gVelocity[ni[k]];
            for (int j = 0; j<3; j++){
              double d_SXoodx = d_S[k][j]*oodx[j];
              for (int i = 0; i<3; i++) {
                velGrad(i,j) += gvel[i] * d_SXoodx;
                //std::cerr << "Grid vel = " << gvel << " dS = " << d_S[k][j] 
                //          << " oodx = " << oodx[j] << endl;
                //std::cerr << " VelGrad(" << i << "," << j << ") = " << velGrad(i,j) << endl;
              }
            }
          }
      };


     inline void computeAxiSymVelocityGradient(Matrix3& velGrad,
                                             vector<IntVector>& ni,
                                             vector<Vector>& d_S,
                                             vector<double>& S,
                                             const double* oodx,
                                             constNCVariable<Vector>& gVelocity,
                                             const Point& px)
     {
        // x -> r, y -> z, z -> theta
        for(int k = 0; k < flag->d_8or27; k++) {
          Vector gvel = gVelocity[ni[k]];
          for (int j = 0; j<2; j++){
            for (int i = 0; i<2; i++) {
              velGrad(i,j)+=gvel[i] * d_S[k][j] * oodx[j];
            }
          }
          velGrad(2,2) += gvel.x()*S[k]/px.x();
        }
     };


    inline void computeVelocityGradient(Matrix3& velGrad,
                                        vector<IntVector>& ni,
                                        vector<Vector>& d_S,
                                        const double* oodx, 
                                        const short pgFld[],
                                        constNCVariable<Vector>& gVelocity,
                                        constNCVariable<Vector>& GVelocity)
      {
        Vector gvel(0.,0.,0);
        for(int k = 0; k < flag->d_8or27; k++) {
          if(pgFld[k]==1)  gvel = gVelocity[ni[k]];
          if(pgFld[k]==2)  gvel = GVelocity[ni[k]];
          for (int j = 0; j<3; j++){
            double d_SXoodx = d_S[k][j]*oodx[j];
            for (int i = 0; i<3; i++) {
              velGrad(i,j) += gvel[i] * d_SXoodx;
            }
          }
        }
      };
    
    /*! Calculate gradient of a vector field for 8 noded interpolation */
    inline void computeGrad(Matrix3& grad,
                            vector<IntVector>& ni,
                            vector<Vector>& d_S,
                            const double* oodx, 
                            constNCVariable<Vector>& gVec)
      {
        // Compute gradient matrix
        grad.set(0.0);
        for(int k = 0; k < flag->d_8or27; k++) {
          const Vector& vec = gVec[ni[k]];
          for (int j = 0; j<3; j++){
            double fac = d_S[k][j]*oodx[j];
            for (int i = 0; i<3; i++) {
              grad(i,j) += vec[i]*fac;
            }
          }
        }
      }

    /*! Calculate gradient of vector field for 8 noded interpolation, B matrix
        for Kmat and B matrix for Kgeo */
    inline void computeGradAndBmats(Matrix3& grad,
                                    vector<IntVector>& ni,
                                    vector<Vector>& d_S,
                                    const double* oodx, 
                                    constNCVariable<Vector>& gVec,
                                    const Array3<int>& l2g,
                                    double B[6][24],
                                    double Bnl[3][24],
                                    int* dof)
    {
    /*!
      \brief Calculate the artificial bulk viscosity (q)

      \f[
      q = \rho (A_1 | c D_{kk} dx | + A_2 D_{kk}^2 dx^2) 
      ~~\text{if}~~ D_{kk} < 0
      \f]
      \f[
      q = 0 ~~\text{if}~~ D_{kk} >= 0
      \f]

      where \f$ \rho \f$ = current density \n
      \f$ dx \f$ = characteristic length = (dx+dy+dz)/3 \n
      \f$ A_1 \f$ = Coeff1 (default = 0.2) \n
      \f$ A_2 \f$ = Coeff2 (default = 2.0) \n
      \f$ c \f$ = Local bulk sound speed = \f$ \sqrt{K/\rho} \f$ \n
      \f$ D_{kk} \f$ = Trace of rate of deformation tensor \n
    */

      int l2g_node_num = -1;
      
      computeGrad(grad,ni,d_S,oodx,gVec);
      
      for (int k = 0; k < 8; k++) {
        B[0][3*k] = d_S[k][0]*oodx[0];
        B[3][3*k] = d_S[k][1]*oodx[1];
        B[5][3*k] = d_S[k][2]*oodx[2];
        B[1][3*k] = 0.;
        B[2][3*k] = 0.;
        B[4][3*k] = 0.;
        
        B[1][3*k+1] = d_S[k][1]*oodx[1];
        B[3][3*k+1] = d_S[k][0]*oodx[0];
        B[4][3*k+1] = d_S[k][2]*oodx[2];
        B[0][3*k+1] = 0.;
        B[2][3*k+1] = 0.;
        B[5][3*k+1] = 0.;
        
        B[2][3*k+2] = d_S[k][2]*oodx[2];
        B[4][3*k+2] = d_S[k][1]*oodx[1];
        B[5][3*k+2] = d_S[k][0]*oodx[0];
        B[0][3*k+2] = 0.;
        B[1][3*k+2] = 0.;
        B[3][3*k+2] = 0.;
        
        Bnl[0][3*k] = d_S[k][0]*oodx[0];
        Bnl[1][3*k] = 0.;
        Bnl[2][3*k] = 0.;
        Bnl[0][3*k+1] = 0.;
        Bnl[1][3*k+1] = d_S[k][1]*oodx[1];
        Bnl[2][3*k+1] = 0.;
        Bnl[0][3*k+2] = 0.;
        Bnl[1][3*k+2] = 0.;
        Bnl[2][3*k+2] = d_S[k][2]*oodx[2];
        
        // Need to loop over the neighboring patches l2g to get the right
        // dof number.
        l2g_node_num = l2g[ni[k]];
        dof[3*k]  =l2g_node_num;
        dof[3*k+1]=l2g_node_num+1;
        dof[3*k+2]=l2g_node_num+2;
      }
    }

    double artificialBulkViscosity(double Dkk, double c, double rho,
                                   double dx) const;

  protected:

    ///////////////////////////////////////////////////////////////////////
    /*! Initialize the common quantities that all the explicit constituive
     *  models compute : called by initializeCMData */
    ///////////////////////////////////////////////////////////////////////
    void initSharedDataForExplicit(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouse* new_dw);


    /////////////////////////////////////////////////////////////////
    /*! Computes and Requires common to all hypo-elastic constitutive models
     *  that do explicit time stepping : called by addComputesAndRequires */
    /////////////////////////////////////////////////////////////////
    void addSharedCRForHypoExplicit(Task* task,
                                    const MaterialSubset* matlset,
                                    const PatchSet* patches) const;

    /////////////////////////////////////////////////////////////////
    /*! Computes and Requires common to all constitutive models that
     *  do explicit time stepping : called by addComputesAndRequires */
    /////////////////////////////////////////////////////////////////
    void addSharedCRForExplicit(Task* task,
                                const MaterialSubset* matlset,
                                const PatchSet* patches) const;

    /////////////////////////////////////////////////////////////////
    /*! Particle conversion related requires common to all constitutive 
        models that do explicit time stepping : called by 
        allocateCMDataAddRequires */
    /////////////////////////////////////////////////////////////////
    void addSharedRForConvertExplicit(Task* task,
                                      const MaterialSubset* matlset,
                                      const PatchSet* ) const;

    /////////////////////////////////////////////////////////////////
    /*! Copy the data common to all constitutive models from the 
        particle to be deleted to the particle to be added. 
        Called by allocateCMDataAdd */
    /////////////////////////////////////////////////////////////////
    void copyDelToAddSetForConvertExplicit(DataWarehouse* new_dw,
                                           ParticleSubset* delset,
                                           ParticleSubset* addset,
           map<const VarLabel*, ParticleVariableBase*>* newState);

    /////////////////////////////////////////////////////////////////
    /*! Carry forward the data common to all constitutive models 
        when using RigidMPM.
        Called by carryForward */
    /////////////////////////////////////////////////////////////////
    void carryForwardSharedData(ParticleSubset* pset,
                                DataWarehouse*  old_dw,
                                DataWarehouse*  new_dw,
                                const MPMMaterial* matl);


    MPMLabel* lb;
    MPMFlags* flag;
    int NGP;
    int NGN;
    const ProcessorGroup* d_world;

    // don't store SimulationStateP or it will add a reference 
    // that will never be removed
    SimulationState* d_sharedState;
  };
} // End namespace Uintah
      


#endif  // __CONSTITUTIVE_MODEL_H__


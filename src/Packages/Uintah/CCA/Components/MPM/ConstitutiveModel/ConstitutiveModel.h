#ifndef __CONSTITUTIVE_MODEL_H__
#define __CONSTITUTIVE_MODEL_H__

#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Core/Containers/StaticArray.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Packages/Uintah/CCA/Components/MPM/PetscSolver.h>
#include <Packages/Uintah/CCA/Components/MPM/SimpleSolver.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>

#define MAX_BASIS 27

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
         
    ConstitutiveModel();
    virtual ~ConstitutiveModel();
         
    // Basic constitutive model calculations
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw,
#ifdef HAVE_PETSC
                                     MPMPetscSolver* solver,
#else
                                     SimpleSolver* solver,
#endif
                                     const bool recursion);
         
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
                                        const bool recursion) const;

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
                                     const MPMMaterial* matl) = 0;

    virtual void computePressEOSCM(double rho_m, double& press_eos,
                                   double p_ref,
                                   double& dp_drho, double& ss_new,
                                   const MPMMaterial* matl) = 0;

    virtual double getCompressibility() = 0;

    virtual Vector getInitialFiberDir();

    double computeRhoMicro(double press,double gamma,
                           double cv, double Temp);
         
    void computePressEOS(double rhoM, double gamma,
                         double cv, double Temp,
                         double& press, double& dp_drho,
                         double& dp_de);

    //////////
    // Convert J-integral into stress intensity for hypoelastic materials 
    // (for FRACTURE)
    virtual void ConvertJToK(const MPMMaterial* matl,const Vector& J,
                             const double& C,const Vector& V,Vector& SIF);

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

  protected:

    // Compute velocity gradient (default case)
    Matrix3 computeVelGrad(const Patch* patch,
                           const double* oodx, 
                           const Point& px, 
                           const Vector& psize, 
                           constNCVariable<Vector>& gVelocity,
                           double erosion);

    // Compute velocity gradient for fracture
    Matrix3 computeVelGrad(const Patch* patch,
                           const double* oodx,
                           const Point& px,
                           const Vector& psize,
                           const short pgFld[], 
                           constNCVariable<Vector>& gVelocity,
                           constNCVariable<Vector>& GVelocity);

    // Calculate velocity gradient for 27 noded interpolation
    Matrix3 computeVelocityGradient(const Patch* patch,
                                    const double* oodx, 
                                    const Point& px, 
                                    const Vector& psize, 
                                    constNCVariable<Vector>& gVelocity);

    // Calculate velocity gradient for 27 noded interpolation (for Erosion)
    Matrix3 computeVelocityGradient(const Patch* patch,
                                    const double* oodx, 
                                    const Point& px, 
                                    const Vector& psize, 
                                    constNCVariable<Vector>& gVelocity,
                                    double erosion);

    // Calculate velocity gradient for 27 noded interpolation (for FRACTURE)
    Matrix3 computeVelocityGradient(const Patch* patch,
                                    const double* oodx,
                                    const Point& px,
                                    const Vector& psize,
                                    const short pgFld[], 
                                    constNCVariable<Vector>& gVelocity,
                                    constNCVariable<Vector>& GVelocity);
 
    // Calculate velocity gradient for 8 noded interpolation (for Erosion)
    Matrix3 computeVelocityGradient(const Patch* patch,
                                    const double* oodx, 
                                    const Point& px, 
                                    constNCVariable<Vector>& gVelocity,
                                    double erosion);

    // Calculate velocity gradient for 8 noded interpolation
    Matrix3 computeVelocityGradient(const Patch* patch,
                                    const double* oodx, 
                                    const Point& px, 
                                    constNCVariable<Vector>& gVelocity);

    // Calculate velocity gradient for 8 noded interpolation (for FRACTURE) 
    Matrix3 computeVelocityGradient(const Patch* patch,
                                    const double* oodx,
                                    const Point& px,
                                    const short pgFld[],
                                    constNCVariable<Vector>& gVelocity,
                                    constNCVariable<Vector>& GVelocity);

    /*! Calculate gradient of a vector field for 8 noded interpolation */
    void computeGrad(const Patch* patch,
                     const double* oodx, 
                     const Point& px, 
                     constNCVariable<Vector>& gVec,
                     Matrix3& grad);

    /*! Calculate gradient of vector field for 8 noded interpolation, B matrix
        for Kmat and B matrix for Kgeo */
    void computeGradAndBmats(const Patch* patch,
                             const double* oodx, 
                             const Point& px, 
                             constNCVariable<Vector>& gVec,
                             const Array3<int>& l2g,
                             Matrix3& grad,
                             double B[6][24],
                             double Bnl[3][24],
                             int* dof);

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
    double artificialBulkViscosity(double Dkk, double c, double rho,
                                   double dx) const;

    void BtDB(const double B[6][24], const double D[6][6], 
              double Km[24][24]) const;
    void BnltDBnl(double Bnl[3][24], double sig[3][3], double Kg[24][24]) const;

  protected:

    ///////////////////////////////////////////////////////////////////////
    /*! Initialize the common quantities that all the explicit constituive
     *  models compute : called by initializeCMData */
    ///////////////////////////////////////////////////////////////////////
    void initSharedDataForExplicit(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouse* new_dw);

    ///////////////////////////////////////////////////////////////////////
    /*! Initialize the common quantities that all the implicit constituive
     *  models compute : called by initializeCMData */
    ///////////////////////////////////////////////////////////////////////
    void initSharedDataForImplicit(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouse* new_dw);

    /////////////////////////////////////////////////////////////////
    /*! Computes and Requires common to all constitutive models that
     *  do explicit time stepping : called by addComputesAndRequires */
    /////////////////////////////////////////////////////////////////
    void addSharedCRForExplicit(Task* task,
                                const MaterialSubset* matlset,
                                const PatchSet* patches) const;

    /////////////////////////////////////////////////////////////////
    /*! Computes and Requires common to all constitutive models that
     *  do implicit time stepping : called by addComputesAndRequires */
    /////////////////////////////////////////////////////////////////
    void addSharedCRForImplicit(Task* task,
                                const MaterialSubset* matlset,
                                const PatchSet* patches) const;

    /////////////////////////////////////////////////////////////////
    /*! Computes and Requires common to all constitutive models that
     *  do implicit time stepping : called by addComputesAndRequires */
    /////////////////////////////////////////////////////////////////
    void addSharedCRForImplicit(Task* task,
                                const MaterialSubset* matlset,
                                const PatchSet* patches,
                                const bool recurse) const;

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
    /*! Set up the particle state common to all constitutive models 
        Called by addParticleState */
    /////////////////////////////////////////////////////////////////
    void addSharedParticleState(std::vector<const VarLabel*>& from,
                                std::vector<const VarLabel*>& to);

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
  };
} // End namespace Uintah
      


#endif  // __CONSTITUTIVE_MODEL_H__


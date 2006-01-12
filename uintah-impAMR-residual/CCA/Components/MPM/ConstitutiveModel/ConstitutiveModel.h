#ifndef __CONSTITUTIVE_MODEL_H__
#define __CONSTITUTIVE_MODEL_H__

#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h>
#include <Core/Containers/StaticArray.h>
#include <Packages/Uintah/Core/Grid/Variables/Array3.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>


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
    ConstitutiveModel(MPMLabel* Mlb, MPMFlags* MFlag);
    virtual ~ConstitutiveModel();
         
    // Basic constitutive model calculations
    virtual void computeStressTensor(const PatchSubset* patches,
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

    // Make a clone of the constitutive model

    virtual ConstitutiveModel* clone() = 0;

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
	      }
	    }
	  }
      };


    inline void computeVelocityGradient(Matrix3& velGrad,
					vector<IntVector>& ni,
					vector<Vector>& d_S,
					const double* oodx, 
					constNCVariable<Vector>& gVelocity,
					double erosion)
      {
	  for(int k = 0; k < flag->d_8or27; k++) {
	    const Vector& gvel = gVelocity[ni[k]];
	    d_S[k] *= erosion;
	    for (int j = 0; j<3; j++){
	      double d_SXoodx = d_S[k][j]*oodx[j];
	      for (int i = 0; i<3; i++) {
		velGrad(i,j) += gvel[i] * d_SXoodx;
	      }
	    }
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
	Vector gvel;
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
	for(int k = 0; k < 8; k++) {
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

    void BtDB(const double B[6][24], const double D[6][6], 
              double Km[24][24]) const;

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
  };
} // End namespace Uintah
      


#endif  // __CONSTITUTIVE_MODEL_H__


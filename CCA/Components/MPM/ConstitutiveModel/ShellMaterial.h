#ifndef __SHELL_CONSTITUTIVE_MODEL_H__
#define __SHELL_CONSTITUTIVE_MODEL_H__

#include <math.h>
#include "ConstitutiveModel.h"
#include "PlasticityModel.h"
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

////////////////////////////////////////////////////////////////////////////
/*! 
   \class ShellMaterial
   \brief Material model for shells (stresses normal to the shell are zero).
   \author Biswajit Banerjee \n
   C-SAFE and Department of Mechanical Engineering \n
   University of Utah \n
   Copyright (C) 2003 University of Utah \n
   \warning  Only isotropic hypoelastic shells implemented.
*/
////////////////////////////////////////////////////////////////////////////

  class ShellMaterial : public ConstitutiveModel {

  public:

    /*! Datatype for storing model parameters */
    struct CMData {
      double Bulk;
      double Shear;
    };

    // Local variables
    const VarLabel* pNormalRotRateLabel; 
    const VarLabel* pRotationLabel;
    const VarLabel* pDefGradTopLabel;
    const VarLabel* pDefGradCenLabel;
    const VarLabel* pDefGradBotLabel;
    const VarLabel* pStressTopLabel;
    const VarLabel* pStressCenLabel;
    const VarLabel* pStressBotLabel;

    const VarLabel* pNormalRotRateLabel_preReloc; 
    const VarLabel* pDefGradTopLabel_preReloc;
    const VarLabel* pDefGradCenLabel_preReloc;
    const VarLabel* pDefGradBotLabel_preReloc;
    const VarLabel* pStressTopLabel_preReloc;
    const VarLabel* pStressCenLabel_preReloc;
    const VarLabel* pStressBotLabel_preReloc;

    const VarLabel* pAverageMomentLabel;
    const VarLabel* pNormalDotAvStressLabel;
    const VarLabel* pRotMassLabel;
    const VarLabel* pNormalRotAccLabel;

  private:
    CMData d_initialData;

    // Prevent copying of this class
    // copy constructor
    ShellMaterial(const ShellMaterial &cm);
    ShellMaterial& operator=(const ShellMaterial &cm);

  public:
    // constructors
    ShellMaterial(ProblemSpecP& ps,  MPMLabel* lb, int n8or27);
       
    // destructor
    virtual ~ShellMaterial();

    virtual void addInitialComputesAndRequires(Task* task,
					       const MPMMaterial* matl,
					       const PatchSet* patches) const;

    // initialize  each particle's constitutive model data
    virtual void initializeCMData(const Patch* patch,
				  const MPMMaterial* matl,
				  DataWarehouse* new_dw);

    virtual void addParticleState(std::vector<const VarLabel*>& from,
				  std::vector<const VarLabel*>& to);
	 
    // compute stable timestep for this patch
    virtual void computeStableTimestep(const Patch* patch,
				       const MPMMaterial* matl,
				       DataWarehouse* new_dw);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Add computes and requires for interpolation of particle rotation to 
        grid */
    //
    void addComputesRequiresParticleRotToGrid(Task* task,
					      const MPMMaterial* matl,
					      const PatchSet* patches);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Actually interpolate normal rotation from particles to the grid */
    //
    void interpolateParticleRotToGrid(const PatchSubset* patches,
				      const MPMMaterial* matl,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw);

    virtual void addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patches) const;

    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
				     const MPMMaterial* matl,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Add computes and requires computation of rotational internal moment */
    //
    void addComputesRequiresRotInternalMoment(Task* task,
					      const MPMMaterial* matl,
					      const PatchSet* patches);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Actually compute rotational Internal moment */
    //
    void computeRotInternalMoment(const PatchSubset* patches,
				  const MPMMaterial* matl,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Add computes and requires computation of rotational acceleration */
    //
    void addComputesRequiresRotAcceleration(Task* task,
					    const MPMMaterial* matl,
					    const PatchSet* patches);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Actually compute rotational acceleration */
    //
    void computeRotAcceleration(const PatchSubset* patches,
				const MPMMaterial* matl,
				DataWarehouse* old_dw,
				DataWarehouse* new_dw);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Add computes and requires update of rotation rate */
    //
    void addComputesRequiresRotRateUpdate(Task* task,
					  const MPMMaterial* matl,
					  const PatchSet* patches); 

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Actually update rotation rate */
    //
    void particleNormalRotRateUpdate(const PatchSubset* patches,
				     const MPMMaterial* matl,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw);

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
    static ConstitutiveModel* readRestartParametersAndCreate(ProblemSpecP ps);

    // class function to create a new object from parameters
    static ConstitutiveModel* create(double *p_array);

    virtual void addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patches,
					const bool recursion) const
    {
    }

    virtual void computeStressTensor(const PatchSubset* patches,
				     const MPMMaterial* matl,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw,
				     Solver* solver,
				     const bool recursion)
    {
    }
	 
  protected:

    // Calculate the incremental rotation matrix for a shell particle
    Matrix3 calcIncrementalRotation(const Vector& r, const Vector& n,
				    double delT);

    // Calculate the total rotation matrix for a shell particle
    void calcTotalRotation(const Vector& n0, const Vector& n, Matrix3& R);

    // Calculate the rotation matrix given and angle and the axis
    // of rotation
    Matrix3 calcRotationMatrix(double angle, const Vector& axis);

    // Calculate the in-plane velocity and rotation gradient 
    void calcInPlaneGradient(const Vector& n, Matrix3& velGrad,
                             Matrix3& rotGrad);

    // Calculate the shell elastic stress
    void computeShellElasticStress(Matrix3& F, Matrix3& sig);

    // Calculate the plane stress deformation gradient corresponding
    // to sig33 = 0 and the Cauchy stress
    bool computePlaneStressAndDefGrad(Matrix3& F, Matrix3& sig);

  };
} // End namespace Uintah
      


#endif  // __SHELL_CONSTITUTIVE_MODEL_H__ 


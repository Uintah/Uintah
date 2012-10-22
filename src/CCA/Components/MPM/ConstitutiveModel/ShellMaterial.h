/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef __SHELL_CONSTITUTIVE_MODEL_H__
#define __SHELL_CONSTITUTIVE_MODEL_H__

#include <cmath>
#include "ConstitutiveModel.h"
#include <Core/Math/Matrix3.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <vector>

namespace Uintah {

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class ShellMaterial
    \brief Material model for shells (stresses normal to the shell are zero).
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
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

  protected:
    CMData d_initialData;
    bool d_includeFlowWork;

  private:
    // Prevent copying of this class
    // copy constructor
    //ShellMaterial(const ShellMaterial &cm);
    ShellMaterial& operator=(const ShellMaterial &cm);

  public:
    // constructors
    ShellMaterial(ProblemSpecP& ps, MPMFlags* flag);
    ShellMaterial(const ShellMaterial* cm);
       
    // destructor
    virtual ~ShellMaterial();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone
    ShellMaterial* clone();

    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;

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
    virtual void addComputesRequiresParticleRotToGrid(Task* task,
                                                      const MPMMaterial* matl,
                                                      const PatchSet* patches);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Actually interpolate normal rotation from particles to the grid */
    //
    virtual void interpolateParticleRotToGrid(const PatchSubset* patches,
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
    virtual void addComputesRequiresRotInternalMoment(Task* task,
                                                      const MPMMaterial* matl,
                                                      const PatchSet* patches);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Actually compute rotational Internal moment */
    //
    virtual void computeRotInternalMoment(const PatchSubset* patches,
                                          const MPMMaterial* matl,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Add computes and requires computation of rotational acceleration */
    //
    virtual void addComputesRequiresRotAcceleration(Task* task,
                                                    const MPMMaterial* matl,
                                                    const PatchSet* patches);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Actually compute rotational acceleration */
    //
    virtual void computeRotAcceleration(const PatchSubset* patches,
                                        const MPMMaterial* matl,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Add computes and requires update of rotation rate */
    //
    virtual void addComputesRequiresRotRateUpdate(Task* task,
                                                  const MPMMaterial* matl,
                                                  const PatchSet* patches); 

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Actually update rotation rate */
    //
    virtual void particleNormalRotRateUpdate(const PatchSubset* patches,
                                             const MPMMaterial* matl,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw);

    virtual double computeRhoMicroCM(double pressure,
                                     const double p_ref,
                                     const MPMMaterial* matl,
                                     double temperature,
                                     double rho_guess);

    virtual void computePressEOSCM(double rho_m, double& press_eos,
                                   double p_ref,
                                   double& dp_drho, double& ss_new,
                                   const MPMMaterial* matl,
                                   double temperature);

    virtual double getCompressibility();

    virtual void addComputesAndRequires(Task* ,
                                        const MPMMaterial* ,
                                        const PatchSet* ,
                                        const bool ) const
      {
      }
         
  protected:

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Calculate the incremental rotation matrix for a shell particle */
    //
    Matrix3 calcIncrementalRotation(const Vector& r, const Vector& n,
                                    double delT);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Calculate the total rotation matrix for a shell particle */
    //
    void calcTotalRotation(const Vector& n0, const Vector& n, Matrix3& R);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Calculate the rotation matrix given and angle and the axis */
    // of rotation
    //
    Matrix3 calcRotationMatrix(double angle, const Vector& axis);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Calculate the in-plane velocity and rotation gradient  */
    //
    void calcInPlaneGradient(const Vector& n, Matrix3& velGrad,
                             Matrix3& rotGrad);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Calculate the shell elastic stress */
    //
    void computeShellElasticStress(Matrix3& F, Matrix3& sig,
                                   double bulk, double shear);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Calculate the plane stress deformation gradient corresponding
    // to sig33 = 0 and the Cauchy stress */
    //
    virtual bool computePlaneStressAndDefGrad(Matrix3& F, Matrix3& sig,
                                      double bulk, double shear);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Calculate the derivative of the Kirchhoff stress component tau_33
    //  with respect to the deformation gradient components */
    //
    void dtau_33_dF(const Matrix3& F, double J, Matrix3& dTdF);

  };
} // End namespace Uintah
      


#endif  // __SHELL_CONSTITUTIVE_MODEL_H__ 


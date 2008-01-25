#ifndef __IMPLICIT_CM_H__
#define __IMPLICIT_CM_H__

#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h>
#include <Core/Containers/StaticArray.h>
#include <Packages/Uintah/Core/Grid/Variables/Array3.h>
#include <Packages/Uintah/CCA/Components/MPM/Solver.h>
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
    \class ImplicitCM
   
    \brief Base class for contitutive models.

    \author Steven G. Parker \n
    Department of Computer Science \n
    University of Utah \n
    Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n
    Copyright (C) 2000 SCI Group \n

    Long description...
  */
  //////////////////////////////////////////////////////////////////////////

  class ImplicitCM {
  public:
         
    ImplicitCM();
    ImplicitCM(const ImplicitCM* cm);
    virtual ~ImplicitCM();
         
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw,
                                     Solver* solver,
                                     const bool recursion);
         
    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        const bool recursion) const;

    void carryForwardSharedDataImplicit(ParticleSubset* pset,
                                        DataWarehouse*  old_dw,
                                        DataWarehouse*  new_dw,
                                        const MPMMaterial* matl);

  protected:


    void BnltDBnl(double Bnl[3][24], double sig[3][3], double Kg[24][24]) const;

    void BnltDBnlGIMP(double Bnl[3][81], double sig[3][3],
                      double Kg[81][81]) const;

    void BtDB(const double B[6][24], const double D[6][6], 
              double Km[24][24]) const;

    void BtDBGIMP(const double B[6][81], const double D[6][6], 
                  double Km[81][81]) const;

    void loadBMats(Array3<int> l2g, int dof[24], double B[6][24], 
                   double Bnl[3][24], vector<Vector> d_S, 
                   vector<IntVector> ni, double oodx[3]) const;

    void loadBMatsGIMP(Array3<int> l2g, int dof[81], double B[6][81], 
                       double Bnl[3][81], vector<Vector> d_S, 
                       vector<IntVector> ni, double oodx[3]) const;

    ///////////////////////////////////////////////////////////////////////
    /*! Initialize the common quantities that all the implicit constituive
     *  models compute : called by initializeCMData */
    ///////////////////////////////////////////////////////////////////////
    void initSharedDataForImplicit(const Patch* patch,
                                   const MPMMaterial* matl,
                                   DataWarehouse* new_dw);


    /////////////////////////////////////////////////////////////////
    /*! Computes and Requires common to all constitutive models that
     *  do implicit time stepping : called by addComputesAndRequires */
    /////////////////////////////////////////////////////////////////
    void addSharedCRForImplicit(Task* task,
                                const MaterialSubset* matlset,
                                const bool reset) const;

    void addSharedCRForImplicitHypo(Task* task,
                                    const MaterialSubset* matlset,
                                    const bool reset) const;

    /////////////////////////////////////////////////////////////////
    /*! Computes and Requires common to all constitutive models that
     *  do implicit time stepping : called by addComputesAndRequires */
    /////////////////////////////////////////////////////////////////
    void addSharedCRForImplicit(Task* task,
                                const MaterialSubset* matlset,
                                const bool reset,
                                const bool recurse) const;

    void addSharedCRForImplicitHypo(Task* task,
                                    const MaterialSubset* matlset,
                                    const bool reset,
                                    const bool recurse) const;

    MPMLabel* d_lb;

  };
} // End namespace Uintah
      


#endif  


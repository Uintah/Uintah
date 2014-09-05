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
#include <Packages/Uintah/CCA/Components/MPM/PetscSolver.h>
#include <Packages/Uintah/CCA/Components/MPM/SimpleSolver.h>
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
    ImplicitCM(MPMLabel* Mlb);
    virtual ~ImplicitCM();
         
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
         


    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        const bool recursion) const;

  protected:


    void BnltDBnl(double Bnl[3][24], double sig[3][3], double Kg[24][24]) const;

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
                                const PatchSet* patches) const;

    /////////////////////////////////////////////////////////////////
    /*! Computes and Requires common to all constitutive models that
     *  do implicit time stepping : called by addComputesAndRequires */
    /////////////////////////////////////////////////////////////////
    void addSharedCRForImplicit(Task* task,
                                const MaterialSubset* matlset,
                                const PatchSet* patches,
                                const bool recurse) const;

    MPMLabel* d_lb;

  };
} // End namespace Uintah
      


#endif  


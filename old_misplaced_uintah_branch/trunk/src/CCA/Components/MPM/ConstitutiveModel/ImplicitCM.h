#ifndef __IMPLICIT_CM_H__
#define __IMPLICIT_CM_H__

#include <Core/Grid/Variables/ComputeSet.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h>
#include <SCIRun/Core/Containers/StaticArray.h>
#include <Core/Grid/Variables/Array3.h>
#include <CCA/Components/MPM/Solver.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
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

  protected:


    void BnltDBnl(double Bnl[3][24], double sig[3][3], double Kg[24][24]) const;

    void loadBMats(Array3<int> l2g, int dof[24], double B[6][24], 
                   double Bnl[3][24], vector<Vector> d_S, 
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


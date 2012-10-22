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

#ifndef __IMPLICIT_CM_H__
#define __IMPLICIT_CM_H__

#include <Core/Grid/Variables/ComputeSet.h>
#include <vector>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h>
#include <Core/Containers/StaticArray.h>
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

    Long description...
  */
  //////////////////////////////////////////////////////////////////////////

  class ImplicitCM {
  public:
         
    ImplicitCM();
    ImplicitCM(const ImplicitCM* cm);
    virtual ~ImplicitCM();
         
    virtual void computeStressTensorImplicit(const PatchSubset* patches,
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
                                const bool recurse,
                                const bool SchedParent) const;

    void addSharedCRForImplicitHypo(Task* task,
                                    const MaterialSubset* matlset,
                                    const bool reset,
                                    const bool recurse,
                                    const bool SchedParent) const;

    MPMLabel* d_lb;

  };
} // End namespace Uintah
      


#endif  


/*
 * The MIT License
 *
 * Copyright (c) 1997-2013 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#ifndef UINTAH_MD_LABEL_H
#define UINTAH_MD_LABEL_H

#include <Core/Containers/LinearArray3.h>
#include <Core/Grid/Variables/Reductions.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Disclosure/TypeUtils.h>

#include <sci_defs/fftw_defs.h>

#include <complex>

namespace Uintah {

  using namespace SCIRun;

  class VarLabel;

  typedef std::complex<double> dblcomplex;
  typedef ReductionVariable<Matrix3, Reductions::Sum<Matrix3> > matrix_sum;

  /**
   *  @class MDLabel
   *  @ingroup MD
   *  @author Alan Humphrey and Justin Hooper
   *  @date   December, 2012
   *
   *  @brief
   *
   *  @param
   */
  class MDLabel {

    public:

      MDLabel();

      ~MDLabel();

      ///////////////////////////////////////////////////////////////////////////
      // Particle Variables
      const VarLabel* pXLabel;
      const VarLabel* pXLabel_preReloc;
      const VarLabel* pNonbondedForceLabel;
      const VarLabel* pNonbondedForceLabel_preReloc;
      const VarLabel* pElectrostaticsForceLabel;
      const VarLabel* pElectrostaticsForceLabel_preReloc;
      const VarLabel* pAccelLabel;
      const VarLabel* pAccelLabel_preReloc;
      const VarLabel* pVelocityLabel;
      const VarLabel* pVelocityLabel_preReloc;
      const VarLabel* pEnergyLabel;
      const VarLabel* pEnergyLabel_preReloc;
      const VarLabel* pMassLabel;
      const VarLabel* pMassLabel_preReloc;
      const VarLabel* pChargeLabel;
      const VarLabel* pChargeLabel_preReloc;
      const VarLabel* pParticleIDLabel;
      const VarLabel* pParticleIDLabel_preReloc;

      ///////////////////////////////////////////////////////////////////////////
      // Reduction Variables - General
      const VarLabel* vdwEnergyLabel;

      ///////////////////////////////////////////////////////////////////////////
      // Reduction Variables - Electrostatics
      const VarLabel* spmeFourierEnergyLabel;
      const VarLabel* spmeFourierStressLabel;

#ifdef HAVE_FFTW
      ///////////////////////////////////////////////////////////////////////////
      // Reduction Variables - specific to SPME
      const VarLabel* globalQLabel;
      const VarLabel* forwardTransformPlanLabel;
      const VarLabel* backwardTransformPlanLabel;
#endif

  };

}  // End namespace Uintah

#endif

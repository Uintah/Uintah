/*
 *
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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
 *
 * ----------------------------------------------------------
 * MDSubcomponentInterface.h
 *
 *  Created on: Apr 30, 2014
 *      Author: jbhooper
 */

#ifndef MDSUBCOMPONENT_H_
#define MDSUBCOMPONENT_H_

#include <Core/Grid/Task.h>

#include <CCA/Components/MD/MDUtil.h>
#include <CCA/Components/MD/MDLabel.h>

namespace Uintah {

  // Define the common subcomponent interface that Uintah::MD will use to establish variable
  //   dependencies with the core MD driver.

  class MDSubcomponent {

    public:

      MDSubcomponent () { }

      virtual ~MDSubcomponent() { }

      virtual void registerRequiredParticleStates( varLabelArray&,
                                                   varLabelArray&,
                                                   MDLabel* ) const = 0;

      virtual void addInitializeRequirements( Task*, MDLabel*, const PatchSet*, const MaterialSet*, const Level* ) const = 0;
      virtual void addInitializeComputes(     Task*, MDLabel*, const PatchSet*, const MaterialSet*, const Level* ) const = 0;

      virtual void addSetupRequirements( Task*, MDLabel*) const = 0;
      virtual void addSetupComputes(     Task*, MDLabel*) const = 0;

      virtual void addCalculateRequirements( Task*, MDLabel*, const PatchSet*, const MaterialSet*, const Level* ) const = 0;
      virtual void addCalculateComputes(     Task*, MDLabel*, const PatchSet*, const MaterialSet*, const Level* ) const = 0;

      virtual void addFinalizeRequirements( Task*, MDLabel* ) const = 0;
      virtual void addFinalizeComputes(     Task*, MDLabel* ) const = 0;

    private:
  };
}

#endif /* MDSUBCOMPONENT_H_ */

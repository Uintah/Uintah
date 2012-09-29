/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_MPM_COMMON_H
#define UINTAH_HOMEBREW_MPM_COMMON_H

#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Util/DebugStream.h>

namespace Uintah {

  class ProcessorGroup;

  using namespace SCIRun;
  
  class MPMCommon {

  public:

    MPMCommon(const ProcessorGroup* myworld);
    virtual ~MPMCommon();

    virtual void materialProblemSetup(const ProblemSpecP& prob_spec,
                                      SimulationStateP& sharedState,
                                      MPMFlags* flags);

    virtual void cohesiveZoneProblemSetup(const ProblemSpecP& prob_spec,
                                          SimulationStateP& sharedState,
                                          MPMFlags* flags);
   protected:
    const ProcessorGroup* d_myworld;
  };
}

#endif

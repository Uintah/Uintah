/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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


#ifndef UINTAH_SPICE_CORE_SPICELABEL_H
#define UINTAH_SPICE_CORE_SPICELABEL_H

#include <vector>

namespace Uintah {

  class VarLabel;

  //______________________________________________________________________
  //  Labels owned by SPICE.  SPICE uses ICE's labels (ICELabel) for every
  //  variable the two components share; only the variables that ICE never
  //  computes live here.  The string names match the ones used when this
  //  code lived in ICE, so existing udas and post-processing scripts work.
  class SPICELabel {
  public:
    SPICELabel();
    ~SPICELabel();

    //__________________________________
    //  CentralUpwind (Kurganov-Tadmor) face variables
    const VarLabel* ktAlphaOwn_FCXLabel;    // KT own-side blend weight alpha_own
    const VarLabel* ktAlphaOwn_FCYLabel;
    const VarLabel* ktAlphaOwn_FCZLabel;
    const VarLabel* ktaSf_FCXLabel;    // KT wave-speed term a*Sf
    const VarLabel* ktaSf_FCYLabel;
    const VarLabel* ktaSf_FCZLabel;
    const VarLabel* ktKappa_FCXLabel;    // KT/upwind blending function kappa
    const VarLabel* ktKappa_FCYLabel;
    const VarLabel* ktKappa_FCZLabel;
    const VarLabel* ktPhiOwn_FCXLabel;    // KT own-side face mass flux
    const VarLabel* ktPhiOwn_FCYLabel;
    const VarLabel* ktPhiOwn_FCZLabel;
    const VarLabel* ktPhiNei_FCXLabel;    // KT nei-side face mass flux
    const VarLabel* ktPhiNei_FCYLabel;
    const VarLabel* ktPhiNei_FCZLabel;
    const VarLabel* ktUvol_FCXLabel;    // KT volumetric face velocity
    const VarLabel* ktUvol_FCYLabel;
    const VarLabel* ktUvol_FCZLabel;

    //__________________________________
    //  PITCH outer loop
    //  *_adv0: the k=0 (main graph) advected conserved state, captured so
    //          a later outer iteration can read it from ParentNewDW without
    //          also modifying the same label in the sub-graph's own NewDW.
    //  pitchResidual: max(|drho|/rho, |dT|/T) over interior cells, used
    //          for the optional <outerTolerance> early exit.
    const VarLabel* mass_adv0Label;
    const VarLabel* mom_adv0Label;
    const VarLabel* eng_adv0Label;
    const VarLabel* sp_vol_adv0Label;
    const VarLabel* pitchResidualLabel;
  };

} // end namespace Uintah

#endif

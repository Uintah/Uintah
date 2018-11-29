/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef CCA_COMPONENTS_ELECTROCHEM_ECLABEL_H
#define CCA_COMPONENTS_ELECTROCHEM_ECLABEL_H

#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

namespace Uintah {
  struct ECLabel {
    ECLabel() {
      cc_matid         = VarLabel::create("cc.MatId",
                                           CCVariable<int>::getTypeDescription());
      cc_concentration = VarLabel::create("cc.Concentration",
                                           CCVariable<double>::getTypeDescription());
      fcx_flux         = VarLabel::create("fcx.Flux",
                                           SFCXVariable<double>::getTypeDescription());
      fcy_flux         = VarLabel::create("fcy.Flux",
                                           SFCYVariable<double>::getTypeDescription());
      fcz_flux         = VarLabel::create("fcz.Flux",
                                           SFCZVariable<double>::getTypeDescription());
    }
    ~ECLabel() {
      VarLabel::destroy(cc_matid);
      VarLabel::destroy(cc_concentration);
      VarLabel::destroy(fcx_flux);
      VarLabel::destroy(fcy_flux);
      VarLabel::destroy(fcz_flux);
    }

    const VarLabel* cc_matid;
    const VarLabel* cc_concentration;
    const VarLabel* fcx_flux;
    const VarLabel* fcy_flux;
    const VarLabel* fcz_flux;
  };
} // End Uintah namespace

#endif //CCA_COMPONENTS_ELECTROCHEM_ECLABEL_H

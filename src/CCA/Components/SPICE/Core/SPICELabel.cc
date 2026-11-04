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


#include <CCA/Components/SPICE/Core/SPICELabel.h>

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>

using namespace Uintah;

SPICELabel::SPICELabel()
{
  const TypeDescription* SFCX_double = SFCXVariable<double>::getTypeDescription();
  const TypeDescription* SFCY_double = SFCYVariable<double>::getTypeDescription();
  const TypeDescription* SFCZ_double = SFCZVariable<double>::getTypeDescription();
  const TypeDescription* CC_double   = CCVariable<double>::getTypeDescription();
  const TypeDescription* CC_Vector   = CCVariable<Vector>::getTypeDescription();
  const TypeDescription* max_variable= max_vartype::getTypeDescription();

  //__________________________________
  //  CentralUpwind (Kurganov-Tadmor) face variables
  ktAlphaOwn_FCXLabel = VarLabel::create( "ktAlphaOwn_FCX", SFCX_double );
  ktAlphaOwn_FCYLabel = VarLabel::create( "ktAlphaOwn_FCY", SFCY_double );
  ktAlphaOwn_FCZLabel = VarLabel::create( "ktAlphaOwn_FCZ", SFCZ_double );
  ktaSf_FCXLabel = VarLabel::create( "ktaSf_FCX", SFCX_double );
  ktaSf_FCYLabel = VarLabel::create( "ktaSf_FCY", SFCY_double );
  ktaSf_FCZLabel = VarLabel::create( "ktaSf_FCZ", SFCZ_double );
  ktKappa_FCXLabel = VarLabel::create( "ktKappa_FCX", SFCX_double );
  ktKappa_FCYLabel = VarLabel::create( "ktKappa_FCY", SFCY_double );
  ktKappa_FCZLabel = VarLabel::create( "ktKappa_FCZ", SFCZ_double );
  ktPhiOwn_FCXLabel = VarLabel::create( "ktPhiOwn_FCX", SFCX_double );
  ktPhiOwn_FCYLabel = VarLabel::create( "ktPhiOwn_FCY", SFCY_double );
  ktPhiOwn_FCZLabel = VarLabel::create( "ktPhiOwn_FCZ", SFCZ_double );
  ktPhiNei_FCXLabel = VarLabel::create( "ktPhiNei_FCX", SFCX_double );
  ktPhiNei_FCYLabel = VarLabel::create( "ktPhiNei_FCY", SFCY_double );
  ktPhiNei_FCZLabel = VarLabel::create( "ktPhiNei_FCZ", SFCZ_double );
  ktUvol_FCXLabel = VarLabel::create( "ktUvol_FCX", SFCX_double );
  ktUvol_FCYLabel = VarLabel::create( "ktUvol_FCY", SFCY_double );
  ktUvol_FCZLabel = VarLabel::create( "ktUvol_FCZ", SFCZ_double );

  //__________________________________
  //  PIMPLE outer loop
  mass_adv0Label      = VarLabel::create( "mass_adv0",      CC_double );
  mom_adv0Label       = VarLabel::create( "mom_adv0",       CC_Vector );
  eng_adv0Label       = VarLabel::create( "eng_adv0",       CC_double );
  sp_vol_adv0Label    = VarLabel::create( "sp_vol_adv0",    CC_double );
  pimpleResidualLabel = VarLabel::create( "pimpleResidual", max_variable );
}

SPICELabel::~SPICELabel()
{
  VarLabel::destroy( ktAlphaOwn_FCXLabel );
  VarLabel::destroy( ktAlphaOwn_FCYLabel );
  VarLabel::destroy( ktAlphaOwn_FCZLabel );
  VarLabel::destroy( ktaSf_FCXLabel );
  VarLabel::destroy( ktaSf_FCYLabel );
  VarLabel::destroy( ktaSf_FCZLabel );
  VarLabel::destroy( ktKappa_FCXLabel );
  VarLabel::destroy( ktKappa_FCYLabel );
  VarLabel::destroy( ktKappa_FCZLabel );
  VarLabel::destroy( ktPhiOwn_FCXLabel );
  VarLabel::destroy( ktPhiOwn_FCYLabel );
  VarLabel::destroy( ktPhiOwn_FCZLabel );
  VarLabel::destroy( ktPhiNei_FCXLabel );
  VarLabel::destroy( ktPhiNei_FCYLabel );
  VarLabel::destroy( ktPhiNei_FCZLabel );
  VarLabel::destroy( ktUvol_FCXLabel );
  VarLabel::destroy( ktUvol_FCYLabel );
  VarLabel::destroy( ktUvol_FCZLabel );

  VarLabel::destroy( mass_adv0Label );
  VarLabel::destroy( mom_adv0Label );
  VarLabel::destroy( eng_adv0Label );
  VarLabel::destroy( sp_vol_adv0Label );
  VarLabel::destroy( pimpleResidualLabel );
}

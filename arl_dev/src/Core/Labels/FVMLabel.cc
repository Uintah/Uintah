/*
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
 */
#include <Core/Labels/FVMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using namespace Uintah;
using namespace std;


FVMLabel::FVMLabel()
{
  concentration_CCLabel = VarLabel::create("cc.concentration",
			CCVariable<double>::getTypeDescription());
  
  flux_SFCXLabel = VarLabel::create( "fcx.flux",
			SFCXVariable<Vector>::getTypeDescription() );
			
  flux_SFCYLabel = VarLabel::create( "fcy.flux",
			SFCYVariable<Vector>::getTypeDescription() );

  flux_SFCZLabel = VarLabel::create( "fcz.flux",
			SFCZVariable<Vector>::getTypeDescription() );

} 

FVMLabel::~FVMLabel()
{
  VarLabel::destroy(concentration_CCLabel);
  VarLabel::destroy(flux_SFCXLabel);
  VarLabel::destroy(flux_SFCYLabel);
  VarLabel::destroy(flux_SFCZLabel);
}

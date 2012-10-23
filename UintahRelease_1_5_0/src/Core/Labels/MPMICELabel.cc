/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


#include <Core/Labels/MPMICELabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;

MPMICELabel::MPMICELabel()
{
  // Cell centered variables
  cMassLabel         = VarLabel::create( "c.mass",
                     CCVariable<double>::getTypeDescription() );
  vel_CCLabel        = VarLabel::create( "vel_CC",
                     CCVariable<Vector>::getTypeDescription() );
  temp_CCLabel       = VarLabel::create("temp_CC",
                     CCVariable<double>::getTypeDescription() );
  press_NCLabel      = VarLabel::create("pressureNC",
                     NCVariable<double>::getTypeDescription());
  burnedMassCCLabel   = VarLabel::create("burnedMass",
                     CCVariable<double>::getTypeDescription());
  onSurfaceLabel      = VarLabel::create("onSurface",
                     CCVariable<double>::getTypeDescription());
  surfaceTempLabel    = VarLabel::create("surfaceTemp",
                     CCVariable<double>::getTypeDescription());   
  scratchLabel        = VarLabel::create("scratch",
                     CCVariable<double>::getTypeDescription());
  scratch1Label        = VarLabel::create("scratch1",
                     CCVariable<double>::getTypeDescription());
  scratch2Label        = VarLabel::create("scratch2",
                     CCVariable<double>::getTypeDescription());
  scratch3Label        = VarLabel::create("scratch3",
                     CCVariable<double>::getTypeDescription()); 
  scratchVecLabel      = VarLabel::create("scratchVec",
                     CCVariable<Vector>::getTypeDescription());
  NC_CCweightLabel     = VarLabel::create("NC_CCweight",
                     NCVariable<double>::getTypeDescription());
  gMassLabel           = VarLabel::create( "g.mass",
                     NCVariable<double>::getTypeDescription() );
  gVelocityLabel       = VarLabel::create( "g.velocity",
	             NCVariable<Vector>::getTypeDescription() );

  //______ D U C T   T A P E__________
  //  WSB1 burn model
  TempGradLabel        = VarLabel::create("TempGrad",
                     CCVariable<double>::getTypeDescription());
  aveSurfTempLabel     = VarLabel::create("aveSurfTemp",
                     CCVariable<double>::getTypeDescription());
} 

MPMICELabel::~MPMICELabel()
{
  
  VarLabel::destroy(cMassLabel);
  VarLabel::destroy(vel_CCLabel);
  VarLabel::destroy(temp_CCLabel);
  VarLabel::destroy(press_NCLabel);
  VarLabel::destroy(burnedMassCCLabel);
  VarLabel::destroy(surfaceTempLabel);
  VarLabel::destroy(onSurfaceLabel);
  VarLabel::destroy(scratchLabel);
  VarLabel::destroy(scratch1Label);
  VarLabel::destroy(scratch2Label);
  VarLabel::destroy(scratch3Label);
  VarLabel::destroy(scratchVecLabel); 
  VarLabel::destroy(NC_CCweightLabel);
  VarLabel::destroy(gMassLabel);
  VarLabel::destroy(gVelocityLabel);
  //______ D U C T   T A P E__________
  //  WSB1 burn model
  VarLabel::destroy(TempGradLabel);
  VarLabel::destroy(aveSurfTempLabel);
}

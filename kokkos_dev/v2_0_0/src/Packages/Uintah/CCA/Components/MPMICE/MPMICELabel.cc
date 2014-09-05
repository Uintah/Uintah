#include <Packages/Uintah/CCA/Components/MPMICE/MPMICELabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Uintah;

MPMICELabel::MPMICELabel()
{
  // Cell centered variables
  cMassLabel    = VarLabel::create( "c.mass",
                     CCVariable<double>::getTypeDescription() );
  cVolumeLabel  = VarLabel::create( "c.volume",
                     CCVariable<double>::getTypeDescription() );
  vel_CCLabel   = VarLabel::create( "vel_CC",
                     CCVariable<Vector>::getTypeDescription() );
  temp_CCLabel     = VarLabel::create("temp_CC",
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
  VarLabel::destroy(cVolumeLabel);
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
  //______ D U C T   T A P E__________
  //  WSB1 burn model
  VarLabel::destroy(TempGradLabel);
  VarLabel::destroy(aveSurfTempLabel);
}

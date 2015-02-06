/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <Core/Labels/AngioLabel.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using namespace Uintah;
using std::cerr;
using std::endl;


AngioLabel::AngioLabel()
{
  pGrowthLabel = VarLabel::create( "p.growth",
                        ParticleVariable<Vector>::getTypeDescription());
  
  pGrowthLabel_preReloc = VarLabel::create( "p.growth+",
                        ParticleVariable<Vector>::getTypeDescription());
  
  pLengthLabel = VarLabel::create( "p.length",
                        ParticleVariable<double>::getTypeDescription());
  
  pLengthLabel_preReloc = VarLabel::create( "p.length+",
                        ParticleVariable<double>::getTypeDescription());
  
  pPhiLabel = VarLabel::create( "p.phi",
                        ParticleVariable<double>::getTypeDescription());
  
  pPhiLabel_preReloc = VarLabel::create( "p.phi+",
                        ParticleVariable<double>::getTypeDescription());
  
  pRadiusLabel = VarLabel::create( "p.radius",
                        ParticleVariable<double>::getTypeDescription());
  
  pRadiusLabel_preReloc = VarLabel::create( "p.radius+",
                        ParticleVariable<double>::getTypeDescription());
  
  pTimeOfBirthLabel = VarLabel::create( "p.tofb",
                        ParticleVariable<double>::getTypeDescription());
  
  pTimeOfBirthLabel_preReloc = VarLabel::create( "p.tofb+",
                        ParticleVariable<double>::getTypeDescription());
  
  pRecentBranchLabel = VarLabel::create( "p.recentbranch",
                        ParticleVariable<double>::getTypeDescription());
  
  pRecentBranchLabel_preReloc = VarLabel::create( "p.recentbranch+",
                        ParticleVariable<int>::getTypeDescription());
  
  pTip0Label = VarLabel::create( "p.tip0",
                        ParticleVariable<int>::getTypeDescription());
  
  pTip0Label_preReloc = VarLabel::create( "p.tip0+",
                        ParticleVariable<int>::getTypeDescription());
  
  pTip1Label = VarLabel::create( "p.tip1",
                        ParticleVariable<int>::getTypeDescription());
  
  pTip1Label_preReloc = VarLabel::create( "p.tip1+",
                        ParticleVariable<int>::getTypeDescription());
  
  pVolumeLabel = VarLabel::create( "p.volume",
                        ParticleVariable<double>::getTypeDescription());
  
  pVolumeLabel_preReloc = VarLabel::create( "p.volume+",
                        ParticleVariable<double>::getTypeDescription());
                                                                                
  pMassLabel = VarLabel::create( "p.mass",
                        ParticleVariable<double>::getTypeDescription() );

  pMassLabel_preReloc = VarLabel::create( "p.mass+",
                        ParticleVariable<double>::getTypeDescription() );

  pXLabel = VarLabel::create("p.x",
                             ParticleVariable<Point>::getTypeDescription(),
                             IntVector(0,0,0), VarLabel::PositionVariable);
  
  pXLabel_preReloc = VarLabel::create( "p.x+",
                        ParticleVariable<Point>::getTypeDescription(),
                        IntVector(0,0,0),
                        VarLabel::PositionVariable);
  
  pParentLabel = VarLabel::create( "p.parent",
                        ParticleVariable<int>::getTypeDescription() );

  pParentLabel_preReloc = VarLabel::create( "p.parent+",
                        ParticleVariable<int>::getTypeDescription() );

  pParticleIDLabel = VarLabel::create("p.particleID",
                        ParticleVariable<long64>::getTypeDescription() );

  pParticleIDLabel_preReloc = VarLabel::create("p.particleID+",
                        ParticleVariable<long64>::getTypeDescription() );


  VesselDensityLabel = VarLabel::create("g.vessel_density",
                        NCVariable<double>::getTypeDescription() );
  
  SmoothedVesselDensityLabel = VarLabel::create("g.smooth_vessel_density",
                        NCVariable<double>::getTypeDescription() );
  
  VesselDensityGradientLabel = VarLabel::create("g.vessel_density_gradient",
                        NCVariable<Vector>::getTypeDescription() );

  CollagenThetaLabel = VarLabel::create("g.collagen_theta",
                        NCVariable<double>::getTypeDescription() );
  
  CollagenDevLabel = VarLabel::create("g.collagen_dev",
                        NCVariable<double>::getTypeDescription() );
  

  delTLabel = VarLabel::create( "delT", delt_vartype::getTypeDescription() );

  pCellNAPIDLabel =
    VarLabel::create("cellNAPID", CCVariable<short int>::getTypeDescription());

  partCountLabel = VarLabel::create("particleCount",
                                   sumlong_vartype::getTypeDescription());
} 

AngioLabel::~AngioLabel()
{
  VarLabel::destroy(pVolumeLabel);
  VarLabel::destroy(pVolumeLabel_preReloc);
  VarLabel::destroy(pMassLabel);
  VarLabel::destroy(pMassLabel_preReloc);
  VarLabel::destroy(pLengthLabel);
  VarLabel::destroy(pLengthLabel_preReloc);
  VarLabel::destroy(pPhiLabel);
  VarLabel::destroy(pPhiLabel_preReloc);
  VarLabel::destroy(pRadiusLabel);
  VarLabel::destroy(pRadiusLabel_preReloc);
  VarLabel::destroy(pTimeOfBirthLabel);
  VarLabel::destroy(pTimeOfBirthLabel_preReloc);
  VarLabel::destroy(pTip0Label);
  VarLabel::destroy(pTip0Label_preReloc);
  VarLabel::destroy(pTip1Label);
  VarLabel::destroy(pTip1Label_preReloc);
  VarLabel::destroy(pRecentBranchLabel);
  VarLabel::destroy(pRecentBranchLabel_preReloc);
  VarLabel::destroy(pXLabel);
  VarLabel::destroy(pXLabel_preReloc);
  VarLabel::destroy(pGrowthLabel);
  VarLabel::destroy(pGrowthLabel_preReloc);
  VarLabel::destroy(pParentLabel);
  VarLabel::destroy(pParentLabel_preReloc);
  VarLabel::destroy(pParticleIDLabel);
  VarLabel::destroy(pParticleIDLabel_preReloc);


  VarLabel::destroy(VesselDensityLabel);
  VarLabel::destroy(SmoothedVesselDensityLabel);
  VarLabel::destroy(VesselDensityGradientLabel);
  VarLabel::destroy(CollagenThetaLabel);
  VarLabel::destroy(CollagenDevLabel);

  VarLabel::destroy(delTLabel);
  VarLabel::destroy(partCountLabel);
  VarLabel::destroy(pCellNAPIDLabel);
}

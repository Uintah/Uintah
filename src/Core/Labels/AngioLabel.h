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


#ifndef UINTAH_HOMEBREW_ANGIOLABEL_H
#define UINTAH_HOMEBREW_ANGIOLABEL_H

#include <vector>

namespace Uintah {

using std::vector;
  class VarLabel;

    class AngioLabel {
    public:

      AngioLabel();
      ~AngioLabel();

      //PermanentParticleState
      const VarLabel* pVolumeLabel;
      const VarLabel* pVolumeLabel_preReloc;
      const VarLabel* pMassLabel;
      const VarLabel* pMassLabel_preReloc;
      const VarLabel* pXLabel;
      const VarLabel* pXLabel_preReloc;
      const VarLabel* pGrowthLabel;
      const VarLabel* pGrowthLabel_preReloc;
      const VarLabel* pLengthLabel;
      const VarLabel* pLengthLabel_preReloc;
      const VarLabel* pPhiLabel;
      const VarLabel* pPhiLabel_preReloc;
      const VarLabel* pRadiusLabel;
      const VarLabel* pRadiusLabel_preReloc;
      const VarLabel* pTip0Label;
      const VarLabel* pTip0Label_preReloc;
      const VarLabel* pTip1Label;
      const VarLabel* pTip1Label_preReloc;
      const VarLabel* pRecentBranchLabel;
      const VarLabel* pRecentBranchLabel_preReloc;
      const VarLabel* pTimeOfBirthLabel;
      const VarLabel* pTimeOfBirthLabel_preReloc;
      const VarLabel* pParentLabel;
      const VarLabel* pParentLabel_preReloc;
      const VarLabel* pParticleIDLabel;
      const VarLabel* pParticleIDLabel_preReloc;

      // Grid Variables
      const VarLabel* VesselDensityLabel;
      const VarLabel* SmoothedVesselDensityLabel;
      const VarLabel* VesselDensityGradientLabel;
      const VarLabel* CollagenThetaLabel;
      const VarLabel* CollagenDevLabel;

      //Miscellaneous Variables
      const VarLabel* delTLabel;
      const VarLabel* partCountLabel;
      const VarLabel* pCellNAPIDLabel;
    };
} // End namespace Uintah

#endif

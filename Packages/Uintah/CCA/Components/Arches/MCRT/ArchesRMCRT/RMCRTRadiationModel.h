/// RMCRTRadiationModel.h-------------------------------------------------------
/// Reverse Monte Carlo Ray Tracing Radiation Model interface
/// 
/// 
/// 
/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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
#ifndef Uintah_Component_Arches_RMCRTRadiationModel_h
#define Uintah_Component_Arches_RMCRTRadiationModel_h

#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/CCA/Ports/SimulationInterface.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>

namespace Uintah {

class RMCRTRadiationModel {

public:
  // constructor
  RMCRTRadiationModel( const ArchesLabel* label );

  // destructor
  ~RMCRTRadiationModel();
  
  //methods: 
  /** @brief Set any parameters from the input file, initialize constants, etc... */
  void problemSetup( const ProblemSpecP& params );
  
  /** @brief Schedule the solution of the radiative transport equation using RMCRT */
  void sched_solve( const LevelP& level, SchedulerP& sched );

  /** @brief Actually solve the radiative transport equation using RMCRT */
  void solve( const ProcessorGroup* pc,  
              const PatchSubset* patches, 
              const MaterialSubset*, 
              DataWarehouse* old_dw, 
              DataWarehouse* new_dw );
 

private:


  // variables:
  int d_constNumRays;
  double d_refTemp, d_uScale, d_refComp;
  bool d_radcal, d_wsgg, d_planckmean, d_patchmean, d_fssk, d_fsck;
  const ArchesLabel* d_lab; 

}; // end class RMCRTRadiationModel
} // end uintah namespace

#endif

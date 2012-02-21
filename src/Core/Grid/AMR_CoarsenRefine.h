#ifndef Uintah_AMR_CoarsenRefine_h
#define Uintah_AMR_CoarsenRefine_h
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

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <CCA/Ports/DataWarehouseP.h>

namespace Uintah {

  class IntVector;
  class Level;


  template<typename T>                                 
    void coarsenDriver_std(const IntVector& cl, 
                           const IntVector& ch,
                           const IntVector& fl,
                           const IntVector& fh,
                           const IntVector& refinementRatio,
                           const double ratio,
                           const Level* coarseLevel,
                           constCCVariable<T>& fine_q_CC,
                           CCVariable<T>& coarse_q_CC );

  template<typename T>
    void coarsenDriver_massWeighted( const IntVector & cl, 
                                     const IntVector & ch,
                                     const IntVector & fl,
                                     const IntVector & fh,
                                     const IntVector & refinementRatio,
                                     const Level* coarseLevel,
                                     constCCVariable<double>& cMass,
                                     constCCVariable<T>& fine_q_CC,
                                     CCVariable<T>& coarse_q_CC );
  template<class T>
  void fineToCoarseOperator(CCVariable<T>& q_CC,
                            const bool,
                            const VarLabel* varLabel,
                            const int indx,
                            DataWarehouse* new_dw,
                            const Patch* coarsePatch,
                            const Level* coarseLevel,
                            const Level* fineLevel);
                                     

}  // end namespace Uintah

#endif // Uintah_AMR_CoarsenRefine_h

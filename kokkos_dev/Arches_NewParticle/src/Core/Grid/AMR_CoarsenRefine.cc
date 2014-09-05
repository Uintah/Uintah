/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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


#include <Core/Grid/AMR_CoarsenRefine.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/AMR.h>
#include <Core/Grid/Level.h>

namespace Uintah {


//______________________________________________________________________
//
template<typename T>
void coarsenDriver_std(const IntVector& cl, 
                       const IntVector& ch,                     
                       const IntVector& fl,                     
                       const IntVector& fh,                     
                       const IntVector& refinementRatio,        
                       const double ratio,                            
                       const Level* coarseLevel,                
                       constCCVariable<T>& fine_q_CC,           
                       CCVariable<T>& coarse_q_CC )
{
  T zero(0.0);
  // iterate over coarse level cells
  for(CellIterator iter(cl, ch); !iter.done(); iter++){
    IntVector c = *iter;
    T q_CC_tmp(zero);
    IntVector fineStart = coarseLevel->mapCellToFiner(c);

    // for each coarse level cell iterate over the fine level cells   
    for(CellIterator inside(IntVector(0,0,0),refinementRatio );
        !inside.done(); inside++){
      IntVector fc = fineStart + *inside;
      
      if( fc.x() >= fl.x() && fc.y() >= fl.y() && fc.z() >= fl.z() &&
          fc.x() <= fh.x() && fc.y() <= fh.y() && fc.z() <= fh.z() ) {
        q_CC_tmp += fine_q_CC[fc];
      }
    }
    coarse_q_CC[c] =q_CC_tmp*ratio;
  }
}

//______________________________________________________________________
// 
template<typename T>
void coarsenDriver_massWeighted( const IntVector & cl,
                                 const IntVector & ch,
                                 const IntVector & fl,
                                 const IntVector & fh,
                                 const IntVector & refinementRatio,
                                 const Level* coarseLevel,
                                 constCCVariable<double>& cMass,
                                 constCCVariable<T>& fine_q_CC,
                                 CCVariable<T>& coarse_q_CC )
{
  T zero(0.0);
  // iterate over coarse level cells
  for(CellIterator iter(cl, ch); !iter.done(); iter++){
    IntVector c = *iter;
    T q_CC_tmp(zero);
    double mass_CC_tmp=0.;
    IntVector fineStart = coarseLevel->mapCellToFiner(c);

    // for each coarse level cell iterate over the fine level cells   
    for(CellIterator inside(IntVector(0,0,0),refinementRatio );
        !inside.done(); inside++){
      IntVector fc = fineStart + *inside;
      
      if( fc.x() >= fl.x() && fc.y() >= fl.y() && fc.z() >= fl.z() &&
          fc.x() <= fh.x() && fc.y() <= fh.y() && fc.z() <= fh.z() ) {
        q_CC_tmp += fine_q_CC[fc]*cMass[fc];
        mass_CC_tmp += cMass[fc];
      }
      
    }
    coarse_q_CC[c] =q_CC_tmp/mass_CC_tmp;
  }
}




// Explicit template instantiations:
template void coarsenDriver_std<double>( const IntVector& cl, const IntVector& ch, const IntVector& fl, const IntVector& fh,
                                         const IntVector& refinementRatio, const double ratio,
                                         const Level* coarseLevel, constCCVariable<double >& fine_q_CC, CCVariable<double>& coarse_q_CC );
template void coarsenDriver_std<Vector>( const IntVector& cl, const IntVector& ch, const IntVector& fl, const IntVector& fh,
                                         const IntVector& refinementRatio, const double ratio,
                                         const Level* coarseLevel, constCCVariable<Vector >& fine_q_CC, CCVariable<Vector>& coarse_q_CC );
                                            

template void coarsenDriver_massWeighted<double>( const IntVector & cl, const IntVector & ch, const IntVector & fl, const IntVector & fh, const IntVector & refinementRatio,
                                                  const Level* coarseLevel, constCCVariable<double>& cMass, constCCVariable<double>& fine_q_CC, CCVariable<double>& coarse_q_CC );
template void coarsenDriver_massWeighted<Vector>( const IntVector & cl, const IntVector & ch, const IntVector & fl, const IntVector & fh, const IntVector & refinementRatio,
                                                  const Level* coarseLevel, constCCVariable<double>& cMass, constCCVariable<Vector>& fine_q_CC, CCVariable<Vector>& coarse_q_CC );

}  // end namespace Uintah

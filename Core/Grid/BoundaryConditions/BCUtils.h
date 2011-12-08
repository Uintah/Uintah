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
#ifndef Packages_Uintah_Core_Grid_BC_BCUtils_h
#define Packages_Uintah_Core_Grid_BC_BCUtils_h
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Patch.h>

namespace Uintah {

void is_BC_specified(const ProblemSpecP& prob_spec, string variable, const MaterialSubset* matls);
  
//______________________________________________________________________
//  Neumann BC:  CCVariable
 template<class T>
 int setNeumannBC_CC( const Patch* patch,
                      const Patch::FaceType face,
                      CCVariable<T>& var,               
                      Iterator& bound_ptr,                 
                      T& value,                         
                      const Vector& cell_dx)                  
{
 IntVector oneCell = patch->faceDirection(face);
 IntVector dir= patch->getFaceAxes(face);
 double dx = cell_dx[dir[0]];

 int nCells = 0;

 if (value == T(0)) {   //    Z E R O  N E U M A N N
   for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
     IntVector adjCell = *bound_ptr - oneCell;
     var[*bound_ptr] = var[adjCell];
   }
   nCells += bound_ptr.size();;
 }else{                //    N E U M A N N  First Order differencing
   for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
     IntVector adjCell = *bound_ptr - oneCell;
     var[*bound_ptr] = var[adjCell] - value * dx;
   }
   nCells += bound_ptr.size();;
 }
 return nCells;
}

//______________________________________________________________________
//  Dirichlet BC:    CCVariable
 template<class T>
 int setDirichletBC_CC( CCVariable<T>& var,     
                        Iterator& bound_ptr,    
                        T& value) 
{
 for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
   var[*bound_ptr] = value;
 }
 int nCells = bound_ptr.size();
 return nCells;
}

} // End namespace Uintah
#endif

#ifndef Packages_Uintah_Core_Grid_BC_BCUtils_h
#define Packages_Uintah_Core_Grid_BC_BCUtils_h

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

#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>

namespace Uintah {

void is_BC_specified( const ProblemSpecP & prob_spec, string variable, const MaterialSubset* matls );
  
/* --------------------------------------------------------------------- 
 For a given domain face, material, variable this returns 
 the following:
    - boundary condition iterator
    - any value associated with that BC
    - BC_kind ( Dirichlet, symmetry, Neumann.....)
 ---------------------------------------------------------------------  */
template <class T>
bool
getIteratorBCValueBCKind( const Patch           * patch, 
                          const Patch::FaceType   face,
                          const int               child,
                          const string          & desc,
                          const int               mat_id,
                                T               & bc_value,
                                Iterator        & bound_ptr,
                                string          & bc_kind );
  
  
template <class T>
bool getIteratorBCValue( const Patch           * patch,
                         const Patch::FaceType   face,
                         const int               child,
                         const string          & desc,
                         const int               mat_id,
                         T                     & bc_value,
                         Iterator              & bound_ptr )
{ 
  bool foundBC = false;

  const BoundCondBase* bc;
  const BoundCond<T>* new_bcs;
  const BCDataArray* bcda = patch->getBCDataArray(face);
  //__________________________________
  //  non-symmetric BCs
  // find the bc_value and kind
  if( !foundBC ){
    bc = bcda->getBoundCondData(mat_id,desc,child);
    new_bcs = dynamic_cast<const BoundCond<T> *>(bc);

    if (new_bcs != 0) {
      bc_value = new_bcs->getValue();
      foundBC = true;
    }
    //    delete bc;  FIXME
  }
  
  //__________________________________
  //  Now deteriming the iterator
  if(foundBC){
    // For this face find the iterator
    bound_ptr = bcda->getCellFaceIterator( mat_id, child, patch );
    
    // bulletproofing
    if (bound_ptr.done()){  // size of the iterator is 0
      return false;
    }
    return true;
  }
  return false;
}  

void
getBCKind( const Patch           * patch, 
           const Patch::FaceType   face,
           const int               child,
           const string          & desc,
           const int               mat_id,
           std::string           & bc_kind,
           std::string           & face_label );
  

//______________________________________________________________________
//  Neumann BC:  CCVariable
template<class T>
int
setNeumannBC_CC( const Patch           * patch,
                 const Patch::FaceType   face,
                       CCVariable<T>   & var,
                       Iterator        & bound_ptr,
                       T               & value,
                 const Vector          & cell_dx )
{
 SCIRun::IntVector oneCell = patch->faceDirection(face);
 SCIRun::IntVector dir= patch->getFaceAxes(face);
 double dx = cell_dx[dir[0]];

 int nCells = 0;

 if (value == T(0)) {   //    Z E R O  N E U M A N N
   for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
     SCIRun::IntVector adjCell = *bound_ptr - oneCell;
     var[*bound_ptr] = var[adjCell];
   }
   nCells += bound_ptr.size();;
 }else{                //    N E U M A N N  First Order differencing
   for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
     SCIRun::IntVector adjCell = *bound_ptr - oneCell;
     var[*bound_ptr] = var[adjCell] - value * dx;
   }
   nCells += bound_ptr.size();;
 }
 return nCells;
}

//______________________________________________________________________
//  Dirichlet BC:    CCVariable
template<class T>
int
setDirichletBC_CC( CCVariable<T> & var,
                   Iterator      & bound_ptr,
                   T             & value )
{
 for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
   var[*bound_ptr] = value;
 }
 int nCells = bound_ptr.size();
 return nCells;
}

} // End namespace Uintah

#endif

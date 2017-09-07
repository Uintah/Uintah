/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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
#ifndef Packages_Uintah_Core_Grid_BC_BCUtils_h
#define Packages_Uintah_Core_Grid_BC_BCUtils_h
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Patch.h>

namespace Uintah {
  
  void is_BC_specified(const ProblemSpecP& prob_spec, std::string variable, const MaterialSubset* matls);
  
  
  /* ---------------------------------------------------------------------
   For a given domain face, material, variable this returns
   the following:
   - boundary condition iterator
   - any value associated with that BC
   - BC_kind ( Dirichlet, symmetry, Neumann.....)
   ---------------------------------------------------------------------  */
  template <class T>
  bool getIteratorBCValueBCKind( const Patch* patch,
                                 const Patch::FaceType face,
                                 const int child,
                                 const std::string& desc,
                                 const int mat_id,
                                 T& bc_value,
                                 Iterator& bound_ptr,
                                 std::string& bc_kind)
  {
    bc_value=T(-9);
    bc_kind="NotSet";
    bool foundBC = false;
    
    //__________________________________
    //  Any variable with zero Neumann BC
    if (desc == "zeroNeumann" ){
      bc_kind = "zeroNeumann";
      bc_value = T(0.0);
      foundBC = true;
    }
    
    const BoundCondBase* bc;
    const BoundCond<T>* new_bcs;
    const BCDataArray* bcd = patch->getBCDataArray(face);
    //__________________________________
    //  non-symmetric BCs
    // find the bc_value and kind
    if( !foundBC ){
      bc = bcd->getBoundCondData(mat_id,desc,child);
      new_bcs = dynamic_cast<const BoundCond<T> *>(bc);
      
      if (new_bcs != 0) {
        bc_value = new_bcs->getValue();
        bc_kind  = new_bcs->getBCType();
        foundBC = true;
      }
      delete bc;
    }
    
    //__________________________________
    // Symmetry
    if( !foundBC ){
      bc = bcd->getBoundCondData(mat_id,"Symmetric",child);
      std::string test =(bc) ? bc->getBCType() : "NotSet";
      
      if (test == "symmetry") {
        bc_kind  = "symmetry";
        bc_value = T(0.0);
        foundBC = true;
      }
      delete bc;
    }
    
    //__________________________________
    //  Now deteriming the iterator
    if(foundBC){
      // For this face find the iterator
      bcd->getCellFaceIterator(mat_id,bound_ptr,child);
      
      // bulletproofing
      if (bound_ptr.done()){  // size of the iterator is 0
        return false;
      }
      return true;
    }
    return false;
  }
  
  template <class T>
  bool getIteratorBCValue( const Patch* patch,
                          const Patch::FaceType face,
                          const int child,
                          const std::string& desc,
                          const int mat_id,
                          T& bc_value,
                          Iterator& bound_ptr )
  {
    bool foundBC = false;
    
    const BoundCondBase* bc;
    const BoundCond<T>* new_bcs;
    const BCDataArray* bcd = patch->getBCDataArray(face);
    //__________________________________
    //  non-symmetric BCs
    // find the bc_value and kind
    if( !foundBC ){
      bc = bcd->getBoundCondData(mat_id,desc,child);
      new_bcs = dynamic_cast<const BoundCond<T> *>(bc);
      
      if (new_bcs != 0) {
        bc_value = new_bcs->getValue();
        foundBC = true;
      }
      delete bc;
    }
    
    //__________________________________
    //  Now deteriming the iterator
    if(foundBC){
      // For this face find the iterator
      bcd->getCellFaceIterator(mat_id,bound_ptr,child);
      
      // bulletproofing
      if (bound_ptr.done()){  // size of the iterator is 0
        return false;
      }
      return true;
    }
    return false;
  }
  
  /**
   *  \author Tony Saad
   *  \date   August 29, 2013
   *  \brief  Allows one to get the boundary value for a given variable (desc) on a given
   boundary face (face, child). This function is templated on the expected datatype. It is
   recommended that you first use getBCKind, among other utilities, to anticipate the datatype
   expected for this variable.
   */
  template <class T>
  bool getBCValue( const Patch* patch,
                  const Patch::FaceType face,
                  const int child,
                  const std::string& desc,
                  const int mat_id,
                  T& bc_value )
  {
    bool foundBC = false;
    
    const BoundCondBase* bc;
    const BoundCond<T>* new_bcs;
    const BCDataArray* bcd = patch->getBCDataArray(face);
    //__________________________________
    //  non-symmetric BCs
    // find the bc_value and kind
    if( !foundBC ){
      bc = bcd->getBoundCondData(mat_id,desc,child);
      new_bcs = dynamic_cast<const BoundCond<T> *>(bc);
      
      if (new_bcs != 0) {
        bc_value = new_bcs->getValue();
        foundBC = true;
      }
      delete bc;
    }
    return foundBC;
  }
  
  
  void
  getBCKind( const Patch* patch,
             const Patch::FaceType face,
             const int child,
             const std::string& desc,
             const int mat_id,
             std::string& bc_kind,
             std::string& face_label );
  
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
    Uintah::IntVector oneCell = patch->faceDirection(face);
    Uintah::IntVector dir= patch->getFaceAxes(face);
    double dx = cell_dx[dir[0]];
    
    int nCells = 0;
    
    if (value == T(0)) {   //    Z E R O  N E U M A N N
      for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
        Uintah::IntVector adjCell = *bound_ptr - oneCell;
        var[*bound_ptr] = var[adjCell];
      }
      nCells += bound_ptr.size();;
    }else{                //    N E U M A N N  First Order differencing
      for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
        Uintah::IntVector adjCell = *bound_ptr - oneCell;
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

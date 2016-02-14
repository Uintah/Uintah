/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#include <CCA/Components/MiniAero/BoundaryCond.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InternalError.h>

 // setenv SCI_DEBUG "BC_dbg:+,BC_CC:+,BC_FC:+"
 // Note:  BC_dbg doesn't work if the iterator bound is
 //        not defined

using namespace std;

namespace Uintah {
namespace MiniAeroNS {

/* --------------------------------------------------------------------- 
 Purpose~   Takes care any CCvariable< >
 ---------------------------------------------------------------------  */
template<class T> 
void setBC(CCVariable<T>& var_CC,
           const string& desc,
           const Patch* patch,
           const int mat_id )
{

  if(patch->hasBoundaryFaces() == false){
    return;
  }

  BC_CC << "-------- setBC (double) \t"<< desc << " mat_id = " 
             << mat_id <<  ", Patch: "<< patch->getID() << endl;
  Vector cell_dx = patch->dCell();

  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  
  // Iterate over the faces encompassing the domain
  for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
    Patch::FaceType face = *iter;
    string bc_kind = "NotSet";      
    int nCells = 0;

    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);

    for (int child = 0;  child < numChildren; child++) {
      T bc_value(-9);
      Iterator bound_ptr;

      bool foundIterator = getIteratorBCValueBCKind<T>( patch, face, child, desc, mat_id,
                                                        bc_value, bound_ptr,bc_kind); 
                                                
      if (foundIterator ) {
        //__________________________________
        // Dirichlet
        if(bc_kind == "Dirichlet"){
           nCells += setDirichletBC_CC< T >( var_CC, bound_ptr, bc_value);
        }
        //__________________________________
        // Neumann
        else if(bc_kind == "Neumann"){
           nCells += setNeumannBC_CC< T >( patch, face, var_CC, bound_ptr, bc_value, cell_dx);
        }                                   
        //__________________________________
        // zeroNeumann 
        else if ( bc_kind == "zeroNeumann" ) {
          bc_value = 0.0;
          nCells += setNeumannBC_CC< T >( patch, face, var_CC, bound_ptr, bc_value, cell_dx);
        }
        //__________________________________
        //  Symmetry
        else if ( bc_kind == "symmetry") {
          nCells += setSymmetryBC_CC( patch, face, var_CC, bound_ptr);
        }
        
        //__________________________________
        //  debugging
        if( BC_dbg.active() ) {
          bound_ptr.reset();
          cout  <<"Face: "<< patch->getFaceName(face) <<"\t numCellsTouched " << nCells
                <<"\t child " << child  <<" NumChildren "<<numChildren 
                <<"\t BC kind "<< bc_kind <<" \tBC value "<< bc_value
                <<"\t bound_itr "<< bound_ptr << endl;
        }
      }  // found iterator
    }  // child loop
    
    BC_CC << "      "<< patch->getFaceName(face) << " \t " << bc_kind << " numChildren: " << numChildren 
	  << " nCellsTouched: " << nCells << endl;

    //__________________________________
    //  bulletproofing
    Patch::FaceIteratorType type = Patch::ExtraPlusEdgeCells;
    int nFaceCells = numFaceCells(patch,  type, face);
    
    bool throwEx = false;
    if(nCells != nFaceCells){
      if( desc == "set_if_sym_BC" && bc_kind == "NotSet"){
        throwEx = false;
      }else{
        throwEx = true;
      }
    }
   
    if(throwEx){
      ostringstream warn;
      warn << "ERROR: MiniAero: SetBC(CCVariable) Boundary conditions were not set correctly ("<< desc<< ", " 
           << patch->getFaceName(face) << ", " << bc_kind  << " numChildren: " << numChildren 
           << " nCells Touched: " << nCells << " nCells on boundary: "<< nFaceCells << ") " << endl;
      throw InternalError(warn.str(), __FILE__, __LINE__);
    }
  }  // faces loop
}


 int setSymmetryBC_CC( const Patch* patch,
                       const Patch::FaceType face,
                       CCVariable<double>& var_CC,               
                       Iterator& bound_ptr)                  
{
   IntVector oneCell = patch->faceDirection(face);
   for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
     IntVector adjCell = *bound_ptr - oneCell;
     var_CC[*bound_ptr] = var_CC[adjCell];
   }
   int nCells = bound_ptr.size();
   return nCells;
}

/* --------------------------------------------------------------------- 
 Function~  setSymmetryBC_CC--
 Tangent components Neumann = 0
 Normal components = -variable[Interior]
 ---------------------------------------------------------------------  */
 int setSymmetryBC_CC( const Patch* patch,
                       const Patch::FaceType face,
                       CCVariable<Vector>& var_CC,               
                       Iterator& bound_ptr)                  
{
   IntVector oneCell = patch->faceDirection(face);
   for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
     IntVector adjCell = *bound_ptr - oneCell;
     var_CC[*bound_ptr] =  -var_CC[adjCell];
   }
   int nCells = bound_ptr.size();
   return nCells;
}

/* --------------------------------------------------------------------- 
 Function~  setGradientBC--
 Calculate the gradients in the extra cells.  Only need the normal
component for now.
 ---------------------------------------------------------------------  */

void setGradientBC(
           CCVariable<Vector> & gradient_var_CC,
           constCCVariable<double> & var_CC,
           const string& desc,
           const Patch* patch,
           const int mat_id )
{
  if(patch->hasBoundaryFaces() == false){
    return;
  }

  BC_CC << "-------- setBC (double) \t"<< desc << " mat_id = " 
             << mat_id <<  ", Patch: "<< patch->getID() << endl;
  Vector cell_dx = patch->dCell();

  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  
  // Iterate over the faces encompassing the domain
  for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
    Patch::FaceType face = *iter;
    string bc_kind = "NotSet";      
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);

    for (int child = 0;  child < numChildren; child++) {
      Iterator bound_ptr;
      double bc_value(-9); 
      bool foundIterator = getIteratorBCValueBCKind<double>(patch, face, child, desc, mat_id,
                                                        bc_value, bound_ptr,bc_kind);

      if(foundIterator) {
        IntVector oneCell = patch->faceDirection(face);
        int P_dir = patch->getFaceAxes(face)[0];  // principal direction
 
        //Update the normal gradient 
        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
          IntVector adjCell = *bound_ptr - oneCell;
          gradient_var_CC[*bound_ptr][P_dir] =  (var_CC[adjCell]-var_CC[*bound_ptr])/cell_dx[P_dir];
        }
      }
    }
  }
}

/* --------------------------------------------------------------------- 
 Function~  setGradientBC--
 Calculate the gradients in the extra cells for vector quantities(velocity).
 ---------------------------------------------------------------------  */

void setGradientBC(
           CCVariable<Matrix3> & gradient_var_CC,
           constCCVariable<Vector> & var_CC,
           const string& desc,
           const Patch* patch,
           const int mat_id )
{
  if(patch->hasBoundaryFaces() == false){
    return;
  }

  BC_CC << "-------- setBC (double) \t"<< desc << " mat_id = " 
             << mat_id <<  ", Patch: "<< patch->getID() << endl;
  Vector cell_dx = patch->dCell();

  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  
  // Iterate over the faces encompassing the domain
  for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
    Patch::FaceType face = *iter;
    string bc_kind = "NotSet";      
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);

    for (int child = 0;  child < numChildren; child++) {
      Iterator bound_ptr;
      double bc_value(-9); 
      bool foundIterator = getIteratorBCValueBCKind<double>(patch, face, child, desc, mat_id,
                                                        bc_value, bound_ptr,bc_kind);

      if(foundIterator) {
        IntVector oneCell = patch->faceDirection(face);
        int P_dir = patch->getFaceAxes(face)[0];  // principal direction
 
        //Update the normal gradient 
        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
          IntVector adjCell = *bound_ptr - oneCell;
          for(int icomp = 0; icomp < 3; icomp++)
            gradient_var_CC[*bound_ptr](P_dir, icomp) =  (var_CC[adjCell][icomp]-var_CC[*bound_ptr][icomp])/cell_dx[P_dir];
        }
      }
    }
  }
}

//______________________________________________________________________
//
int numFaceCells(const Patch* patch, 
                 const Patch::FaceIteratorType type,
                 const Patch::FaceType face)
{
  IntVector lo = patch->getFaceIterator(face,type).begin();
  IntVector hi = patch->getFaceIterator(face,type).end();
  int numFaceCells = (hi.x()-lo.x())  *  (hi.y()-lo.y())  *  (hi.z()-lo.z());
  return numFaceCells;
}


// Explicit template instantiations:
template void setBC(CCVariable<double>& variable, const std::string& desc,const Patch* patch, const int mat_id);
template void setBC(CCVariable<Vector>& variable, const std::string& desc,const Patch* patch, const int mat_id);



}  // using namespace MiniAeroNS
}  // using namespace Uintah

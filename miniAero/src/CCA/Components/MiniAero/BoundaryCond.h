/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#ifndef __MiniAero_SM_BOUNDARYCOND_H__
#define __MiniAero_SM_BOUNDARYCOND_H__
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Util/DebugStream.h>



namespace Uintah {
namespace MiniAeroNS {

  using namespace SCIRun;
  
  static DebugStream BC_dbg("BC_dbg", false);
  static DebugStream BC_CC("BC_CC", false);
  static DebugStream BC_FC("BC_FC", false);
  
  class DataWarehouse;
 
  void BC_bulletproofing(const ProblemSpecP& prob_spec,
                         SimulationStateP& sharedState );
  
  //__________________________________
  // Main driver method for CCVariables
  template<class T> 
  void setBC(CCVariable<T>& variable, 
             const std::string& desc,
             const Patch* patch,    
             const int mat_id);

  int setSymmetryBC_CC( const Patch* patch,
                        const Patch::FaceType face,
                        CCVariable<Vector>& var_CC,               
                        Iterator& bound_ptr);

  template<class T>
  int setDirichletBC_FC( const Patch* patch,
                         const Patch::FaceType face,       
                         T& vel_FC,                        
                         Iterator& bound_ptr,                  
                         double& value,          
                         const std::string& whichVel);
  
  
  int numFaceCells(const Patch* patch, 
                   const Patch::FaceIteratorType type,
                   const Patch::FaceType face);

/* --------------------------------------------------------------------- 
 Purpose~   does the actual work of setting the BC for face-centered 
            velocities
 ---------------------------------------------------------------------  */
 template<class T>
 int setDirichletBC_FC( const Patch* patch,
                        const Patch::FaceType face,       
                        T& vel_FC,                        
                        Iterator& bound_ptr,                 
                        double& value)           
{
  int nCells = 0;
  IntVector oneCell(0,0,0);

  if ((face == Patch::xminus) ||     
      (face == Patch::yminus) ||     
      (face == Patch::zminus)){      
    oneCell = patch->faceDirection(face);                                        
  }                                                            

  // on (x,y,z)minus faces move inward one cell
  // on (x,y,z)plus faces oneCell == 0                                                             
  for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {  
    IntVector c = *bound_ptr - oneCell;                      
    vel_FC[c] = value;                                       
  }                                
  nCells +=bound_ptr.size();                                              
  return nCells; 
}


/* --------------------------------------------------------------------- 
 Purpose~   Takes capre of face centered velocities
            The normal components are computed in  computeVel_FC
 ---------------------------------------------------------------------  */
 template<class T> 
void setBC_FC(T& vel_FC, 
              const std::string& desc,
              const Patch* patch,    
              const int mat_id)      
{
  BC_FC << "--------setBCFC (SFCVariable) "<< desc<< " mat_id = " << mat_id <<std::endl;
  std::string whichVel = "unknown";
  
  //__________________________________
  // Iterate over the faces encompassing the domain
  std::vector<Patch::FaceType>::const_iterator iter;
  std::vector<Patch::FaceType> bf;

  patch->getBoundaryFaces(bf);

  for (iter  = bf.begin(); iter != bf.end(); ++iter){
    Patch::FaceType face = *iter;
    
    IntVector faceDir = Abs(patch->faceDirection(face));
    
    // SFC(X,Y,Z) Vars can only be set on (x,y,z)+ & (x,y,z)- faces
    if(  (faceDir.x() == 1 &&  (typeid(T) == typeid(SFCXVariable<double>)) ) ||
         (faceDir.y() == 1 &&  (typeid(T) == typeid(SFCYVariable<double>)) ) ||
         (faceDir.z() == 1 &&  (typeid(T) == typeid(SFCZVariable<double>)) ) ){
    
      int nCells = 0;
      std::string bc_kind = "NotSet";
      
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);
      for (int child = 0;  child < numChildren; child++) {

        Vector bc_value(-9,-9,-9);
        Iterator bound_ptr;
        bool foundIterator = 
          getIteratorBCValueBCKind<Vector>( patch, face, child, desc, mat_id,
					         bc_value, bound_ptr,bc_kind); 

        if (foundIterator && bc_kind != "Neumann" ) {
          //__________________________________
          // Extract which SFC variable you're
          //  working on, the value 
          double value=-9;
          if (typeid(T) == typeid(SFCXVariable<double>)) {
            value = bc_value.x();
            whichVel = "X_vel_FC";
          }
          else if (typeid(T) == typeid(SFCYVariable<double>)) {
            value = bc_value.y();
            whichVel = "Y_vel_FC";
          }
          else if (typeid(T) == typeid(SFCZVariable<double>)) {
            value = bc_value.z();
            whichVel = "Z_vel_FC";
          }

          //__________________________________
          //  Neumann BCs
          //The normal components are computed in AddExchangeContributionToFCVel.
          //Do Not Set BC here
          
          //__________________________________
          //  Symmetry boundary conditions
          //  -- faces in the principal dir:     vel[c] = 0
          if (bc_kind == "symmetry") { 
            value = 0.0;                                                                           
            nCells += setDirichletBC_FC<T>( patch, face, vel_FC, bound_ptr, value);    
          }
          //__________________________________
          // Dirichlet
          else if (bc_kind == "Dirichlet") {  
            nCells += setDirichletBC_FC<T>( patch, face, vel_FC, bound_ptr, value);
          }

          //__________________________________
          //  debugging
          if( BC_dbg.active() ) {
            bound_ptr.reset();
            BC_dbg <<whichVel<<" Face: "<< patch->getFaceName(face) <<"\t numCellsTouched " << nCells
                 <<"\t child " << child  <<" NumChildren "<<numChildren 
                 <<"\t BC kind "<< bc_kind <<" \tBC value "<< value
                 <<"\t bound_ptr= " << bound_ptr<< std::endl;
          }              
        }  // Children loop
      }
      BC_FC << "               " << patch->getFaceName(face) << " \t " << whichVel << " \t" << bc_kind << "\t faceDir: " << faceDir << " numChildren: " << numChildren 
                 << " nCells: " << nCells << std::endl;
      //__________________________________
      //  bulletproofing
      Patch::FaceIteratorType type = Patch::ExtraPlusEdgeCells;
      int nFaceCells = numFaceCells(patch,  type, face);
      
      if(nCells != nFaceCells &&  bc_kind != "Neumann"){
        std::ostringstream warn;
        warn << "ERROR MiniAero: Boundary conditions were not set for ("<< whichVel << ", " 
             << patch->getFaceName(face) << ", " << bc_kind  << " numChildren: " << numChildren 
             << " nCells Touched: " << nCells << " nCells on boundary: "<< nFaceCells << ") " << std::endl;
        throw InternalError(warn.str(), __FILE__, __LINE__);
      }
    }  // found iterator
  }  // face loop
}
} // End namespace MiniAeroNS
} // End namespace Uintah
#endif

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

#include <CCA/Components/ICE_sm/BoundaryCond.h>

#include <CCA/Components/ICE_sm/ICEMaterial.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CellIterator.h>

#include <typeinfo>
#include <Core/Exceptions/InternalError.h>

 // setenv SCI_DEBUG "ICE_BC_DBG:+,ICE_BC_DOING:+"
 // Note:  BC_dbg doesn't work if the iterator bound is
 //        not defined

//#define TEST
#undef TEST
using namespace std;
using namespace Uintah;
namespace Uintah {



/* --------------------------------------------------------------------- 
 Function~  setBC-- (pressure)
 ---------------------------------------------------------------------  */
void setBC(CCVariable<double>& press_CC,
           const string& kind, 
           const Patch* patch, 
           const int mat_id,
           DataWarehouse* new_dw)
{
  if(patch->hasBoundaryFaces() == false){
    return;
  }
  
  ice_BC_CC << "-------- setBC (press_CC) \t"<< kind 
            << " mat_id = " << mat_id <<  ", Patch: "<< patch->getID() << endl;


  //__________________________________
  //  N O N  -  L O D I
  //__________________________________
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  
  for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
    Patch::FaceType face = *iter;
    string bc_kind = "NotSet";
    int nCells     = 0;
   
    Vector cell_dx = patch->dCell();
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);

    for (int child = 0;  child < numChildren; child++) {
      double bc_value = -9;
      Iterator bound_ptr;
      
      bool foundIterator = 
        getIteratorBCValueBCKind<double>( patch, face, child, kind, mat_id,
                                               bc_value, bound_ptr,bc_kind); 
      
      if(foundIterator) {                                            
        //__________________________________
        // Dirichlet
        if(bc_kind == "Dirichlet"){
           nCells += setDirichletBC_CC<double>( press_CC, bound_ptr, bc_value);
        }
        //__________________________________
        // Neumann
        else if(bc_kind == "Neumann"){
           nCells += setNeumannBC_CC<double >( patch, face, press_CC, bound_ptr, bc_value, cell_dx);
        } 
        //__________________________________
        //  Symmetry
        else if ( bc_kind == "symmetry" || bc_kind == "zeroNeumann" ) {
          bc_value = 0.0;
          nCells += setNeumannBC_CC<double >( patch, face, press_CC, bound_ptr, bc_value, cell_dx);
        }
        //__________________________________
        //  debugging
        if( BC_dbg.active() ) {
          bound_ptr.reset();
          BC_dbg <<"Face: "<< patch->getFaceName(face) <<"\t numCellsTouched " << nCells
               <<"\t child " << child  <<" NumChildren "<<numChildren 
               <<"\t BC kind "<< bc_kind <<" \tBC value "<< bc_value
               <<"\t bound_ptr = "<< bound_ptr<< endl;
        }
      } 
    }  // child loop
    
    ice_BC_CC << "      "<< patch->getFaceName(face) << " \t " << bc_kind << " numChildren: " << numChildren 
                        << " nCellsTouched: " << nCells << endl;
    //__________________________________
    //  bulletproofing   
    Patch::FaceIteratorType type = Patch::ExtraPlusEdgeCells;
    int nFaceCells = numFaceCells(patch,  type, face);
    
    if(nCells != nFaceCells ){
      ostringstream warn;
      warn << "ERROR: ICE: SetBC(press_CC) Boundary conditions were not set correctly ("
           << patch->getFaceName(face) << ", " << bc_kind  << " numChildren: " << numChildren 
           << " nCells Touched: " << nCells << " nCells on boundary: "<< nFaceCells << ") " << endl;
      throw InternalError(warn.str(), __FILE__, __LINE__);
    }
  }  // faces loop
}
/* --------------------------------------------------------------------- 
 Function~  setBC--
 Purpose~   Takes care any CCvariable<double>, except Pressure
 ---------------------------------------------------------------------  */
void setBC(CCVariable<double>& var_CC,
           const string& desc,
           const CCVariable<double>& gamma,
           const CCVariable<double>& cv,
           const Patch* patch,
           const int mat_id,
           DataWarehouse*)
{

  if(patch->hasBoundaryFaces() == false){
    return;
  }
  ice_BC_CC << "-------- setBC (double) \t"<< desc << " mat_id = " 
             << mat_id <<  ", Patch: "<< patch->getID() << endl;
  Vector cell_dx = patch->dCell();
  
  //__________________________________
  //  N O N  -  L O D I
  //__________________________________
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  
  // Iterate over the faces encompassing the domain
  for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
    Patch::FaceType face = *iter;
    string bc_kind = "NotSet";      
    int nCells = 0;

    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);

    for (int child = 0;  child < numChildren; child++) {
      double bc_value = -9;
      Iterator bound_ptr;

      bool foundIterator = 
        getIteratorBCValueBCKind<double>( patch, face, child, desc, mat_id,
                                               bc_value, bound_ptr,bc_kind); 
                                                
      if (foundIterator ) {
        //__________________________________
        // Dirichlet
        if(bc_kind == "Dirichlet"){
           nCells += setDirichletBC_CC<double>( var_CC, bound_ptr, bc_value);
        }
        //__________________________________
        // Neumann
        else if(bc_kind == "Neumann"){
           nCells += setNeumannBC_CC<double >( patch, face, var_CC, bound_ptr, bc_value, cell_dx);
        }                                   
        //__________________________________
        //  Symmetry
        else if ( bc_kind == "symmetry" || bc_kind == "zeroNeumann" ) {
          bc_value = 0.0;
          nCells += setNeumannBC_CC<double >( patch, face, var_CC, bound_ptr, bc_value, cell_dx);
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
    
    ice_BC_CC << "      "<< patch->getFaceName(face) << " \t " << bc_kind << " numChildren: " << numChildren 
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
      warn << "ERROR: ICE: SetBC(double_CC) Boundary conditions were not set correctly ("<< desc<< ", " 
           << patch->getFaceName(face) << ", " << bc_kind  << " numChildren: " << numChildren 
           << " nCells Touched: " << nCells << " nCells on boundary: "<< nFaceCells << ") " << endl;
      throw InternalError(warn.str(), __FILE__, __LINE__);
    }
  }  // faces loop
}

/* --------------------------------------------------------------------- 
 Function~  setBC--
 Purpose~   Takes care vector boundary condition
 ---------------------------------------------------------------------  */
void setBC(CCVariable<Vector>& var_CC,
           const string& desc,
           const Patch* patch,
           const int mat_id,
           DataWarehouse* )
{
 if(patch->hasBoundaryFaces() == false){
    return;
  }
  ice_BC_CC <<"-------- setBC (Vector_CC) \t"<< desc <<" mat_id = " 
              <<mat_id<<  ", Patch: "<< patch->getID() << endl;
  
  Vector cell_dx = patch->dCell();

  //__________________________________
  //  N O N  -  L O D I
  //__________________________________
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  
  // Iterate over the faces encompassing the domain
  for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
    Patch::FaceType face = *iter;
    int nCells = 0;
    string bc_kind = "NotSet";

    IntVector oneCell = patch->faceDirection(face);
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);
    
    // loop over the geometry objects on a face
    for (int child = 0;  child < numChildren; child++) {
      Vector bc_value = Vector(-9,-9,-9);
      
      Iterator bound_ptr;

      bool foundIterator = 
          getIteratorBCValueBCKind<Vector>(patch, face, child, desc, mat_id,
                                            bc_value, bound_ptr ,bc_kind);
      
      if (foundIterator) {
        
        //__________________________________
        // Dirichlet
        if(bc_kind == "Dirichlet"){
           nCells += setDirichletBC_CC<Vector>( var_CC, bound_ptr, bc_value);
        }
        //__________________________________
        // Neumann
        else if(bc_kind == "Neumann"){
           nCells += setNeumannBC_CC<Vector>( patch, face, var_CC, bound_ptr, bc_value, cell_dx);
        }                                   
        //__________________________________
        //  Symmetry
        else if ( bc_kind == "symmetry" ) {
          nCells += setSymmetryBC_CC( patch, face, var_CC, bound_ptr);
        }
        //__________________________________
        //  debugging
        if( BC_dbg.active() ) {
          BC_dbg <<"Face: "<< patch->getFaceName(face) <<"\t numCellsTouched " << nCells
               <<"\t child " << child  <<" NumChildren "<<numChildren 
               <<"\t BC kind "<< bc_kind <<" \tBC value "<< bc_value
               <<"\t bound_ptr = "<< bound_ptr<< endl;
        }
      }  // found iterator
    }  // child loop
    ice_BC_CC << "      "<< patch->getFaceName(face) << " \t " << bc_kind << " numChildren: " << numChildren 
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
      warn << "ERROR: ICE: SetBC(Vector_CC) Boundary conditions were not set correctly ("<< desc<< ", " 
           << patch->getFaceName(face) << ", " << bc_kind  << " numChildren: " << numChildren 
           << " nCells Touched: " << nCells << " nCells on boundary: "<< nFaceCells << ") " << endl;
      throw InternalError(warn.str(), __FILE__, __LINE__);
    }
  }  // faces loop
}
/* --------------------------------------------------------------------- 
 Function~  setSpecificVolBC-- 
 ---------------------------------------------------------------------  */
void setSpecificVolBC(CCVariable<double>& sp_vol_CC,
                      const string& desc,
                      const bool isMassSp_vol,
                      constCCVariable<double> rho_CC,
                      constCCVariable<double> vol_frac,
                      const Patch* patch,
                      const int mat_id)
{
  if(patch->hasBoundaryFaces() == false){
    return;
  }
  ice_BC_CC << "-------- setSpecificVolBC \t"<< desc <<" "
             << " mat_id = " << mat_id <<  ", Patch: "<< patch->getID() << endl;
                
  Vector dx = patch->dCell();
  double cellVol = dx.x() * dx.y() * dx.z();
                
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  
  for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
    Patch::FaceType face = *iter;
    int nCells = 0;
    string bc_kind = "NotSet";
       
    IntVector dir= patch->getFaceAxes(face);
    Vector cell_dx = patch->dCell();
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);
    
    // iterate over each geometry object along that face
    for (int child = 0;  child < numChildren; child++) {
      double bc_value = -9;
      Iterator bound_ptr;
      
      bool foundIterator = 
        getIteratorBCValueBCKind<double>( patch, face, child, desc, mat_id,
                                          bc_value, bound_ptr,bc_kind); 
                                   
      if(foundIterator) {

        //__________________________________
        // Dirichlet
        if(bc_kind == "Dirichlet"){
           nCells += setDirichletBC_CC<double>( sp_vol_CC, bound_ptr, bc_value);
        }
        //__________________________________
        // Neumann
        else if(bc_kind == "Neumann"){
           nCells += setNeumannBC_CC<double >( patch, face, sp_vol_CC, bound_ptr, bc_value, cell_dx);
        }                                   
        //__________________________________
        //  Symmetry
        else if ( bc_kind == "symmetry" || bc_kind == "zeroNeumann" ) {
          bc_value = 0.0;
          nCells += setNeumannBC_CC<double >( patch, face, sp_vol_CC, bound_ptr, bc_value, cell_dx);
        }
        //__________________________________
        //  Symmetry
        else if(bc_kind == "computeFromDensity"){
        
          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
            IntVector c = *bound_ptr;
            sp_vol_CC[c] = vol_frac[c]/rho_CC[c];
          }
          
          if(isMassSp_vol){  // convert to mass * sp_vol
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector c = *bound_ptr;
              sp_vol_CC[c] = sp_vol_CC[c]*(rho_CC[c]*cellVol);
            }
          }
          nCells += bound_ptr.size();
        }

        //__________________________________
        //  debugging
        if( BC_dbg.active() ) {
          bound_ptr.reset();
          BC_dbg <<"Face: "<< patch->getFaceName(face) <<"\t numCellsTouched " << nCells
               <<"\t child " << child  <<" NumChildren "<<numChildren 
               <<"\t BC kind "<< bc_kind <<" \tBC value "<< bc_value
               <<"\t bound limits = "<< bound_ptr << endl;
        }
      }  // if iterator found
    }  // child loop
    
    ice_BC_CC << "      "<< patch->getFaceName(face) << " \t " << bc_kind << " numChildren: " << numChildren 
               << " nCellsTouched: " << nCells << endl;
    //__________________________________
    //  bulletproofing
#if 0
    Patch::FaceIteratorType type = Patch::ExtraPlusEdgeCells;
    int nFaceCells = numFaceCells(patch,  type, face);
                        
    if(nCells != nFaceCells){
      ostringstream warn;
      warn << "ERROR: ICE: setSpecificVolBC Boundary conditions were not set correctly ("<< desc<< ", " 
           << patch->getFaceName(face) << ", " << bc_kind  << " numChildren: " << numChildren 
           << " nCells Touched: " << nCells << " nCells on boundary: "<< nFaceCells<<") " << endl;
      throw InternalError(warn.str(), __FILE__, __LINE__);
    }
#endif
  }  // faces loop
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
   int P_dir = patch->getFaceAxes(face)[0];  // principal direction
   IntVector sign = IntVector(1,1,1);
   sign[P_dir] = -1;

   for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
     IntVector adjCell = *bound_ptr - oneCell;
     var_CC[*bound_ptr] = sign.asVector() * var_CC[adjCell];
   }
   int nCells = bound_ptr.size();
   return nCells;
}


/* --------------------------------------------------------------------- 
 Function~  BC_bulletproofing--  
 ---------------------------------------------------------------------  */
void BC_bulletproofing(const ProblemSpecP& prob_spec,
                       SimulationStateP& sharedState )
{
  Vector periodic;
  ProblemSpecP grid_ps  = prob_spec->findBlock("Grid");
  ProblemSpecP level_ps = grid_ps->findBlock("Level");
  level_ps->getWithDefault("periodic", periodic, Vector(0,0,0));
  
  Vector tagFace_minus(0,0,0);
  Vector tagFace_plus(0,0,0);
                               
  ProblemSpecP bc_ps  = grid_ps->findBlock("BoundaryConditions");
  int numAllMatls = sharedState->getNumMatls();
  
  // If a face is periodic then is_press_BC_set = true
  map<string,int> is_press_BC_set;
  is_press_BC_set["x-"] = (periodic.x() ==1) ? 1:0;
  is_press_BC_set["x+"] = (periodic.x() ==1) ? 1:0;
  is_press_BC_set["y-"] = (periodic.y() ==1) ? 1:0;
  is_press_BC_set["y+"] = (periodic.y() ==1) ? 1:0;
  is_press_BC_set["z-"] = (periodic.z() ==1) ? 1:0;
  is_press_BC_set["z+"] = (periodic.z() ==1) ? 1:0;
  
  // loop over all boundary conditions for a face
  // This include circles, rectangles, annulus
  for (ProblemSpecP face_ps = bc_ps->findBlock("Face");face_ps != 0; 
                    face_ps=face_ps->findNextBlock("Face")) {
  
    map<string,bool>isBC_set;
    isBC_set["Temperature"] =false;
    isBC_set["Density"]     =false;
    isBC_set["Velocity"]    =false;            
    isBC_set["SpecificVol"] =true;  
    isBC_set["Symmetric"]   =false;    
                      
    map<string,string> face;
    face_ps->getAttributes(face);

    //loop through the attributes and find  (x-,x+,y-,y+... )
    string side = "NULL";
    
    for( map<string,string>::iterator iter = face.begin(); iter !=  face.end(); iter++ ){
      string me = (*iter).second;
      if( me =="x-" || me == "x+" ||
          me =="y-" || me == "y+" ||
          me =="z-" || me == "z+" ){
        side = me;
        continue;
      }
    }

    // tag each face if it's been specified
    if(side == "x-") tagFace_minus.x(1);   
    if(side == "y-") tagFace_minus.y(1);   
    if(side == "z-") tagFace_minus.z(1);   

    if(side == "x+") tagFace_plus.x(1);    
    if(side == "y+") tagFace_plus.y(1);    
    if(side == "z+") tagFace_plus.z(1);

    // loop over all BCTypes for that face 
    for(ProblemSpecP bc_iter = face_ps->findBlock("BCType"); bc_iter != 0;
                     bc_iter = bc_iter->findNextBlock("BCType")){
      map<string,string> bc_type;
      bc_iter->getAttributes(bc_type);
            
      // valid user input      
      if( bc_type["label"] != "Pressure"      && bc_type["label"] != "Temperature" && 
          bc_type["label"] != "SpecificVol"   && bc_type["label"] != "Velocity" &&
          bc_type["label"] != "Density"       && bc_type["label"] != "Symmetric" &&
          bc_type["label"] != "scalar-f"      && bc_type["label"] != "cumulativeEnergyReleased"){
        ostringstream warn;
        warn <<"\n INPUT FILE ERROR:\n The boundary condition label ("<< bc_type["label"] <<") is not valid\n"
             << " Face:  " << face["side"] << " BCType " << bc_type["label"]<< endl;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }  
      
      // specified "all" for a 1 matl problem
      if (bc_type["id"] == "all" && numAllMatls == 1){
        ostringstream warn;
        warn <<"\n__________________________________\n"   
             << "ERROR: This is a single material problem and you've specified 'BCType id = all' \n"
             << "The boundary condition infrastructure treats 'all' and '0' as two separate materials, \n"
             << "setting the boundary conditions twice on each face.  Set BCType id = '0' \n" 
             << " Face:  " << face["side"] << " BCType " << bc_type["label"]<< endl;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      
      // symmetric BCs
      if ( bc_type["label"] == "Symmetric"){
        if (numAllMatls > 1 &&  bc_type["id"] != "all") {
          ostringstream warn;
          warn <<"\n__________________________________\n"   
             << "ERROR: This is a multimaterial problem with a symmetric boundary condition\n"
             << "You must have the id = all instead of id = "<<bc_type["id"]<<"\n"
             << "Face:  " << face["side"] << " BCType " << bc_type["label"]<< endl;
          throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
        }
      }  // symmetric 
      
      // All passed tests on this face set the flags to true
      if(bc_type["label"] == "Pressure" || bc_type["label"] == "Symmetric"){
        is_press_BC_set[side]  +=1;
      }
      isBC_set[bc_type["label"]] = true;
    }  // BCType loop
    
    //__________________________________
    //Now check if all the variables on this face were set
    for( map<string,bool>::iterator iter = isBC_set.begin();iter !=  isBC_set.end(); iter++ ){
      string var = (*iter).first;
      bool isSet = (*iter).second;
      bool isSymmetric = isBC_set["Symmetric"];
      
      if(isSymmetric == false && isSet == false && var != "Symmetric"){
        ostringstream warn;
        warn <<"\n__________________________________\n"   
           << "INPUT FILE ERROR: \n"
           << "The "<<var<<" boundary condition for one of the materials has not been set \n"
           << "Face:  " << face["side"] <<  endl;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
    }
    
    //__________________________________
    //Has the pressure BC been set on this face;
    int isSet  = is_press_BC_set.count( side );

    if(isSet != 1){
      ostringstream warn;
      warn <<"\n__________________________________\n"   
         << "INPUT FILE ERROR: \n"
         << "The pressure boundary condition has not been set OR has been set more than once \n"
         << "Face:  " << side <<  endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
  } //face loop

  //__________________________________
  // Has each non-periodic face has been touched?
  if (periodic.length() == 0){
    if( (tagFace_minus != Vector(1,1,1)) ||
        (tagFace_plus  != Vector(1,1,1)) ){
      ostringstream warn;
      warn <<"\n__________________________________\n "
           << "ERROR: the boundary conditions on one of the faces of the computational domain has not been set \n"<<endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);  
    }
  }
  
  // Periodic BC and missing BC's
  if(periodic.length() != 0){
    for(int dir = 0; dir<3; dir++){
      if( periodic[dir]==0 && ( tagFace_minus[dir] == 0 || tagFace_plus[dir] == 0)){
        ostringstream warn;
        warn <<"\n__________________________________\n "
             << "ERROR: You must specify a boundary condition in direction "<< dir << endl;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);   
      }
    }
  }
  
  // Duplicate periodic BC and normal BCs
  if(periodic.length() != 0){
    for(int dir = 0; dir<3; dir++){
      if( periodic[dir]==1 && ( tagFace_minus[dir] == 1 || tagFace_plus[dir] == 1)){
        ostringstream warn;
        warn <<"\n__________________________________\n "
             << "ERROR: A periodic AND a normal boundary condition have been specifed for \n"
             << " direction: "<< dir << "  You can only have on or the other"<< endl;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);   
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


}  // using namespace Uintah

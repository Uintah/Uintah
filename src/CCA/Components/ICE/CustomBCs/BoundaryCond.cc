/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <CCA/Components/ICE/CustomBCs/BoundaryCond.h>

#include <CCA/Components/ICE/Materials/ICEMaterial.h>
#include <CCA/Components/ICE/EOS/EquationOfState.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/AMR.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CellIterator.h>

#include <typeinfo>
#include <Core/Util/DebugStream.h>
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
 Function~  ImplicitMatrixBC--      
 Purpose~   Along each face of the domain set the stencil weight
 Naming convention
      +x -x +y -y +z -z
       e, w, n, s, t, b 
 
   A.p = beta[c] -
          (A.n + A.s + A.e + A.w + A.t + A.b);
   LHS       
   A.p*delP - (A.e*delP_e + A.w*delP_w + A.n*delP_n + A.s*delP_s 
             + A.t*delP_t + A.b*delP_b )
             
 Suppose the x- face has Press=Neumann BC, then you must add A.w to
 both A.p and set A.w = 0.  If the pressure is Dirichlet BC you leave A.p 
 alone and set A.w = 0;           
       
 ---------------------------------------------------------------------  */
void ImplicitMatrixBC( CCVariable<Stencil7>& A, 
                       const Patch* patch)        
{ 
  cout_BC_CC << "-------- ImplicitMatrixBC Patch: "<< patch->getID()<< endl;
  
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
    Patch::FaceType face = *itr;
    string bc_kind  = "NotSet";
    int nCells = 0;
    
    int mat_id = 0; // hard coded for pressure
    
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);
    for (int child = 0;  child < numChildren; child++) {
      double bc_value = -9;
      
      Iterator bound_ptr;  
      
      bool foundIterator =       
        getIteratorBCValueBCKind<double>( patch, face, child, "Pressure", 
                                         mat_id, bc_value, bound_ptr,bc_kind);
                                    
      // don't set BCs unless we've found the iterator                                   
      if (foundIterator) {
        //__________________________________
        //  Neumann or Dirichlet Press_BC;
        double one_or_zero = -999;
        if(bc_kind == "zeroNeumann" || bc_kind == "Neumann" ||
           bc_kind == "symmetry"    ||bc_kind == "MMS_1"){
          one_or_zero = 1.0;      // subtract from A.p
        }

        if(bc_kind == "Dirichlet" || bc_kind == "LODI" || bc_kind == "Sine" ){
          one_or_zero = 0.0;      // leave A.p Alone
        }                                 
        //__________________________________
        //  Set the BC  

        switch (face) {
        case Patch::xplus:
          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
            IntVector c(*bound_ptr - IntVector(1,0,0));
            A[c].p = A[c].p + one_or_zero * A[c].e;
            A[c].e = 0.0;
          }
          nCells += bound_ptr.size();
          break;
        case Patch::xminus:
          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) { 
            IntVector c(*bound_ptr + IntVector(1,0,0));
            A[c].p = A[c].p + one_or_zero * A[c].w;
            A[c].w = 0.0;
          }
          nCells += bound_ptr.size();
          break;
        case Patch::yplus:
          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) { 
            IntVector c(*bound_ptr - IntVector(0,1,0));
            A[c].p = A[c].p + one_or_zero * A[c].n;
            A[c].n = 0.0;
          }
          nCells += bound_ptr.size();
          break;
        case Patch::yminus:
          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
            IntVector c(*bound_ptr + IntVector(0,1,0)); 
            A[c].p = A[c].p + one_or_zero * A[c].s;
            A[c].s = 0.0;
          }
          nCells += bound_ptr.size();
          break;
        case Patch::zplus:
          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
            IntVector c(*bound_ptr - IntVector(0,0,1));
            A[c].p = A[c].p + one_or_zero * A[c].t;
            A[c].t = 0.0;
          }
          nCells += bound_ptr.size();
          break;
        case Patch::zminus:
          for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
            IntVector c(*bound_ptr + IntVector(0,0,1));
            A[c].p = A[c].p + one_or_zero * A[c].b;
            A[c].b = 0.0;
          }
          nCells += bound_ptr.size();
          break;
        case Patch::numFaces:
          break;
        case Patch::invalidFace:
          break; 
        }
        //__________________________________
        //  debugging
        if( BC_dbg.active() ) {
          bound_ptr.reset();
          BC_dbg <<"Face: "<< patch->getFaceName(face) <<" numCellsTouched " << nCells << "\t one_or_zero " << one_or_zero
               <<"\t child " << child  <<" NumChildren "<<numChildren 
               <<"\t BC kind "<< bc_kind
               <<"\t bound_ptr "<< bound_ptr<< endl;
        }
      } // if iterator found
    } // child loop
    cout_BC_CC << "      "<< patch->getFaceName(face) << " \t " << bc_kind << " numChildren: " << numChildren 
               << " nCellsTouched: " << nCells << endl;
    
    //__________________________________
    //  bulletproofing    
    Patch::FaceIteratorType type = Patch::ExtraPlusEdgeCells;
    int nFaceCells = numFaceCells(patch,  type, face);
    
    if(nCells != nFaceCells){
      ostringstream warn;
      warn << "ERROR: ICE: ImplicitMatrixBC Boundary conditions were not set correctly (" 
           << patch->getFaceName(face) << ", " << bc_kind  << " numChildren: " << numChildren 
           << " nCells Touched: " << nCells << " nCells on boundary: "<< nFaceCells<<") " << endl;
      throw InternalError(warn.str(), __FILE__, __LINE__);
    }
  }  // face loop
  
 /*`==========TESTING==========*/
#ifdef TEST
  //__________________________________
  // On the fine levels at the coarse fine interface 
  // set A_(t,b,e,w,n,s) = c1 * A_(*)_org.
  // We are assuming that the change in pressure impDelP is linearly interpolated
  // between the coarse cell and the fine cell 
  BC_dbg << *patch << " ";
  patch->printPatchBCs(BC_dbg);

  if(patch->hasCoarseFaces() ){  
    cout << " Matrix BC at coarse/Fine interfaces " << endl;
    //__________________________________
    // Iterate over coarsefine interface faces
    vector<Patch::FaceType> cf;
    patch->getCoarseFaces(cf);
    
    for( vector<Patch::FaceType>::const_iterator iter = cf.begin(); iter != cf.end(); ++iter ){
      Patch::FaceType face = *iter;

      const Level* fineLevel = patch->getLevel();
      const Level* coarseLevel = fineLevel->getCoarserLevel().get_rep();
      
      IntVector cl, ch, fl, fh;
      getCoarseFineFaceRange(patch, coarseLevel, face,Patch::ExtraPlusEdgeCells, 1, cl, ch, fl, fh);
 
 
      IntVector refineRatio = fineLevel->getRefinementRatio();
      Vector D = (refineRatio.asVector() + Vector(1))/Vector(2.0);
      
      //Vector C1 = Vector(1.0)/refineRatio.asVector();
      
      Vector C1 = (refineRatio.asVector() - Vector(1))/(refineRatio.asVector() + Vector(1));
      Vector C2 = Vector(1.0) - C1;
      
      int P_dir = patch->getFaceAxes(face)[0];  //principal dir.
      
      IntVector offset = patch->faceDirection(face);
      
      
      for(CellIterator cIter(fl,fh); !cIter.done(); cIter++){
        IntVector f_cell = *cIter;
        f_cell =  f_cell - offset;
        A[f_cell].p += A[f_cell][face];
        
        A[f_cell][face] = C2[P_dir] * A[f_cell][face];
        
        A[f_cell].p -= A[f_cell][face];
      }
    }  // face loop 
  }  // patch has coarse fine interface 
#endif 
/*===========TESTING==========`*/
  
  
}


/* --------------------------------------------------------------------- 
 Function~  set_imp_DelP_BC--      
 Purpose~  set the boundary condition for the change in pressure (imp_del_P_
 computed by the semi-implicit pressure solve.  This routine takes care
 of the BC at the edge of the computational domain and at all coarse/fine
 interfaces.
 ---------------------------------------------------------------------  */
void set_imp_DelP_BC( CCVariable<double>& imp_delP, 
                      const Patch* patch,
                      const VarLabel* label,
                      DataWarehouse* new_dw)        
{ 
  cout_BC_CC << "-------- set_imp_DelP_BC, Patch: "<< patch->getID()<< endl;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
    Patch::FaceType face = *itr;
    
    int mat_id = 0; // hard coded for pressure
    IntVector oneCell = patch->faceDirection(face);
    int nCells = 0;
    string bc_kind  = "NotSet";
    
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);
    for (int child = 0;  child < numChildren; child++) {
      double bc_value = -9;
      Iterator bound_ptr;  
      
      bool foundIterator =       
        getIteratorBCValueBCKind<double>( patch, face, child, "Pressure", 
                                         mat_id, bc_value, bound_ptr,bc_kind);
                                    
      // don't set BCs unless we've found the iterator                                   
      if (foundIterator) {
        //__________________________________
        //  Neumann or Dirichlet Press_BC;
        double one_or_zero = -999;
        if(bc_kind == "zeroNeumann" || bc_kind == "Neumann" ||
           bc_kind == "symmetric"   || bc_kind == "MMS_1"){
          one_or_zero = 1.0;     
        }
        else if(bc_kind == "Dirichlet" || bc_kind == "LODI" || bc_kind == "Sine"){
          one_or_zero = 0.0;
        }                                 
        //__________________________________
        //  Set the BC  
        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
          IntVector c = *bound_ptr;
          IntVector adj = c - oneCell;
          imp_delP[c] = one_or_zero * imp_delP[adj];
        }
        nCells += bound_ptr.size();
        //__________________________________
        //  debugging
        if( BC_dbg.active() ) {
          bound_ptr.reset();
          BC_dbg <<"Face: "<< patch->getFaceName(face)<<" numCellsTouched " << nCells
               <<"\t child " << child  <<" NumChildren "<<numChildren
               <<"\t BC kind "<< bc_kind <<" \tBC value "<< bc_value
               <<"\t bound_ptr = "<< bound_ptr<< endl;
        }
      } // if(foundIterator)
    } // child loop
    
    cout_BC_CC << "      "<< patch->getFaceName(face) << " \t " << bc_kind << " numChildren: " << numChildren 
               << " nCellsTouched: " << nCells << endl;
    //__________________________________
    // bulletproofing      
    Patch::FaceIteratorType type = Patch::ExtraPlusEdgeCells;
    int nFaceCells = numFaceCells(patch,  type, face);
    
              
    if( nCells != nFaceCells){
      ostringstream warn;
      warn << "ERROR: ICE: set_imp_DelP_BC Boundary conditions were not set correctly ("
           << patch->getFaceName(face) << ", " << bc_kind  << " numChildren: " << numChildren 
           << " nCells Touched: " << nCells << " nCells on boundary: "<< nFaceCells<<") " << endl;
      throw InternalError(warn.str(), __FILE__, __LINE__);
    }
  }  // face loop
  
  
  //__________________________________
  // On the fine levels at the coarse fine interface 
  // set imp_delP_fine = to imp_delP_coarse
  BC_dbg << *patch << " ";
  patch->printPatchBCs(BC_dbg);

  if(patch->hasCoarseFaces() ){  
    BC_dbg << " BC at coarse/Fine interfaces " << endl;
    //__________________________________
    // Iterate over coarsefine interface faces
    vector<Patch::FaceType> cf;
    patch->getCoarseFaces(cf);
    for( vector<Patch::FaceType>::const_iterator iter = cf.begin(); iter != cf.end(); ++iter ){
      Patch::FaceType face = *iter;

      const Level* fineLevel = patch->getLevel();
      const Level* coarseLevel = fineLevel->getCoarserLevel().get_rep();
      
      IntVector cl, ch, fl, fh;
      getCoarseFineFaceRange(patch, coarseLevel, face, Patch::ExtraPlusEdgeCells, 1, cl, ch, fl, fh);

      constCCVariable<double> imp_delP_coarse;
      new_dw->getRegion(imp_delP_coarse, label, 0, coarseLevel,cl, ch);
#ifndef TEST
      // piece wise constant 
      for(CellIterator cIter(fl,fh); !cIter.done(); cIter++){
        IntVector f_cell = *cIter;
        IntVector c_cell = fineLevel->mapCellToCoarser(f_cell);
        imp_delP[f_cell] =  imp_delP_coarse[c_cell];
      }
#endif
/*`==========TESTING==========*/
#ifdef TEST
      IntVector refineRatio = fineLevel->getRefinementRatio();
      IntVector offset = patch->faceDirection(face);
      
      Vector C1 = (refineRatio.asVector() - Vector(1))/(refineRatio.asVector() + Vector(1));
      Vector C2 = Vector(1.0) - C1;
      
      int P_dir = patch->getFaceAxes(face)[0];  //principal dir.
      
      cout << " using linear Interpolation for impDelP " << endl;;
     
      for(CellIterator cIter(fl,fh); !cIter.done(); cIter++){
        IntVector f_cell = *cIter;
        IntVector f_adj  = f_cell - offset;
        IntVector c_cell = fineLevel->mapCellToCoarser(f_cell);
        imp_delP[f_cell] =  C2[P_dir] * imp_delP_coarse[c_cell] +
                            C1[P_dir] * imp_delP[f_adj];
      }
      
#endif 
/*===========TESTING==========`*/
    }  // face loop 
  }  // patch has coarse fine interface 
}


/* --------------------------------------------------------------------- 
 Function~  get_rho_micro--
 Purpose~  This handles all the logic of getting rho_micro on the faces
    a) when using lodi bcs get rho_micro for all ice and mpm matls
    b) with gravity != 0 get rho_micro on P_dir faces, for all ICE matls
    c) during initialization only get rho_micro for ice_matls, you can't
       get rho_micro for mpm_matls
 ---------------------------------------------------------------------  */
void get_rho_micro(std::vector<CCVariable<double> >& rho_micro,
                   std::vector<CCVariable<double> >& rho_micro_tmp,
                   std::vector<constCCVariable<double> >& sp_vol_CC,
                   const Patch* patch,
                   const string& which_Var,
                   MaterialManagerP& materialManager,
                   DataWarehouse* new_dw,
                   customBC_globalVars* globalVars,
                   const bool isNotInitialTimeStep)
{
  BC_dbg << "-------- get_rho_micro: (" << which_Var <<")"<< endl;
  
  if( which_Var !="rho_micro" && which_Var !="sp_vol" ){
    throw InternalError("setBC (pressure): Invalid option for which_var", __FILE__, __LINE__);
  }
  
  Vector gravity = globalVars->d_gravity; 
  int numICEMatls  = materialManager->getNumMatls( "ICE" );
    

//  This doesn't work with AMR.  The refine/setBC_fineLevel task only
//  refines ICE matls so we don't have access to sp_vol_mpm.
//
//   bool isNotInitialTimeStep = (materialManager->getCurrentTopLevelTimeStep() > 0)//
//  if (isNotInitialTimeStep) {
//    numMatls += materialManager->getNumMatls( "MPM" );
//  }
      
  //__________________________________
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  
  for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
    Patch::FaceType face = *iter;
    
    if(is_LODI_face(patch, face, materialManager) || gravity.length() > 0) {
      
      //__________________________________
      // Create an iterator that iterates over the face
      // + 2 cells inward (hydrostatic press tweak).  
      // We don't need to hit every  cell on the patch. 
      Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;
      
      CellIterator iter_tmp = patch->getFaceIterator(face, PEC);
      IntVector lo = iter_tmp.begin();
      IntVector hi = iter_tmp.end();
    
      int P_dir = patch->getFaceAxes(face)[0];  //principal dir.
      if(face==Patch::xminus || face==Patch::yminus || face==Patch::zminus){
        hi[P_dir] += 2;
      }
      if(face==Patch::xplus || face==Patch::yplus || face==Patch::zplus){
        lo[P_dir] -= 2;
      }
      CellIterator iterLimits(lo,hi);
      
      for (int m = 0; m < numICEMatls; m++) {
        ICEMaterial* ice_matl = (ICEMaterial*) materialManager->getMaterial( "ICE", m);
        int matl= ice_matl->getDWIndex();
                
        if (which_Var == "rho_micro") { 
          for (CellIterator cIter = iterLimits; !cIter.done(); cIter++) {
            IntVector c = *cIter;
            rho_micro[matl][c] =  rho_micro_tmp[matl][c];
          }
        }
        if (which_Var == "sp_vol") { 
          for (CellIterator cIter = iterLimits; !cIter.done(); cIter++) {
            IntVector c = *cIter;
            rho_micro[matl][c] =  1.0/sp_vol_CC[matl][c];
          }
        }  // sp_vol
      }  // numMatls
    }  // LODI face or gravity != 0
  }  // face iter 
}

 //__________________________________________________________
 // Tack on hydrostatic pressure correction after Dirichlet 
 // or Neumann BC have been applied.  During the intializaton 
 // phase the hydrostatic pressure adjustment is  handled by a
 // separate task, therefore ignore it        
 // 
 // Hydrostatic pressure adjustment (HPA) = 
 //   gravity*<rho_micro>*distance_from_ref_point.
 // R: is BC location, L: is adjacent to BC location
 // 
 // On Dirichlet side walls you still have to add HPA
  
void HydrostaticPressureAdjustment(CCVariable<double>& press_CC,
                                   std::vector<CCVariable<double> >& rho_micro,
                                   const Vector gravity,
                                   const int surroundingMatl_indx,
                                   const string& bc_kind, 
                                   const Patch* patch,
                                   const Patch::FaceType face,
                                   Iterator bound_ptr,
                                   const bool isNotInitialTimeStep)
{  
  if ( isNotInitialTimeStep ){
    const Level* level = patch->getLevel();
    GridP grid = level->getGrid();
    BBox b;
    grid->getSpatialRange(b);
    Vector gridMin = b.min().asVector();
    Vector dx_L0 = grid->getLevel(0)->dCell();
    Vector cell_dx = patch->dCell();

    // Pressure reference point is assumed to be 
    // at the cell-center of cell 0,0,0 
    Vector press_ref_pt = gridMin + 1.5*dx_L0;

    int p_dir = patch->getFaceAxes(face)[0];     // normal  face direction

    //__________________________________
    //Neumann
    bool Neumann_BC = (bc_kind=="Neumann" || bc_kind=="zeroNeumann");
    if ((gravity[p_dir] != 0 && Neumann_BC) ){

      Vector faceDir    = patch->faceDirection(face).asVector();
      double grav       = gravity[p_dir] * (double)faceDir[p_dir]; 
      IntVector oneCell = patch->faceDirection(face);

      for (bound_ptr.reset();!bound_ptr.done(); bound_ptr++) { 
        IntVector L = *bound_ptr - oneCell;
        IntVector R = *bound_ptr;
        double rho_R = rho_micro[surroundingMatl_indx][R];
        double rho_L = rho_micro[surroundingMatl_indx][L];
        double rho_micro_brack = (rho_L + rho_R)/2.0;
        
        press_CC[R] += grav * cell_dx[p_dir] * rho_micro_brack; 
      }     
    }
    //__________________________________
    //  Dirichlet
    if(gravity.length() != 0 && bc_kind =="Dirichlet"){  

      for (bound_ptr.reset();!bound_ptr.done(); bound_ptr++) {
        IntVector R = *bound_ptr;
        Point here_R  = level->getCellPosition(R);
        Vector dist_R = (here_R.asVector() - press_ref_pt);
        double rho_R  = rho_micro[surroundingMatl_indx][R];
        
        // Need the dot product to get the sideWall dirichlet BC's right
        double correction_R = Dot(gravity,dist_R) * rho_R;

        press_CC[R] += correction_R;
      }
    } 
  } //  not initialization step 
}

/* --------------------------------------------------------------------- 
 Function~  setBC-- (pressure)
 ---------------------------------------------------------------------  */
void setBC(CCVariable<double>& press_CC,
           std::vector<CCVariable<double> >& rho_micro_tmp,   //or placeHolder
           std::vector<constCCVariable<double> >& sp_vol_CC,  //or placeHolder
           const int surroundingMatl_indx,
           const string& which_Var,
           const string& kind, 
           const Patch* patch,
           MaterialManagerP& materialManager, 
           const int mat_id,
           DataWarehouse* new_dw,
           customBC_globalVars* globalVars,
           customBC_localVars* localVars,
           const bool isNotInitialTimeStep)
{
  if(patch->hasBoundaryFaces() == false){
    return;
  }
  
  cout_BC_CC << "-------- setBC (press_CC) \t"<< kind <<" " << which_Var
            << " mat_id = " << mat_id <<  ", Patch: "<< patch->getID() << endl;

  int numALLMatls = materialManager->getNumMatls();
  // bool isNotInitialTimeStep = (materialManager->getCurrentTopLevelTimeStep() > 0);  
  Vector gravity = globalVars->d_gravity;
  std::vector<CCVariable<double> > rho_micro(numALLMatls);
  
  for (int m = 0; m < numALLMatls; m++) {
    new_dw->allocateTemporary(rho_micro[m],  patch);
  }
  
  get_rho_micro(rho_micro, rho_micro_tmp, sp_vol_CC, 
                patch, which_Var, materialManager,  new_dw, globalVars,
                isNotInitialTimeStep);
                
  //__________________________________
  //  -Set the LODI BC's first, then let the other BC's wipe out what
  //   was set in the corners and edges. 
  //  -Ignore lodi bcs during intialization phase AND when
  //   lv->setLodiBcs = false              
  //__________________________________
  vector<int> nCells_LODI(Patch::numFaces);
  for (int f = 0; f < Patch::numFaces; f++) {
    nCells_LODI[f] = 0;  // bulletproofing
  }
  
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  // Iterate over the faces encompassing the domain  
  for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ) {
    Patch::FaceType face = *iter;
    
    bool is_lodi_pressBC = patch->haveBC(face,mat_id,"LODI","Pressure");
    
    if(kind == "Pressure"       && is_lodi_pressBC 
       && isNotInitialTimeStep  && localVars->setLodiBcs){
       
       nCells_LODI[face] += 
       FacePress_LODI(patch, press_CC, rho_micro, materialManager,face, localVars->lodi);
    }
  }

  //__________________________________
  //  N O N  -  L O D I
  //__________________________________
  // Iterate over the faces encompassing the domain
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
      
      if(foundIterator && bc_kind != "LODI") {                                            
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
        //  Custom Boundary Conditions
        else if (bc_kind == "MMS_1" && localVars->set_MMS_BCs) {
          nCells += set_MMS_press_BC(patch, face, press_CC, bound_ptr,  bc_kind,
                                      materialManager, 
                                      globalVars->mms,
                                      localVars->mms);
        }                    
        //__________________________________
        //  Sine
        else if (bc_kind == "Sine" && localVars->set_Sine_BCs) {
          nCells += set_Sine_press_BC(patch, face, press_CC, bound_ptr,  bc_kind,
                                       materialManager, 
                                       globalVars->sine,
                                       localVars->sine);
        }
        
        if( globalVars->applyHydrostaticPress ){
          HydrostaticPressureAdjustment(press_CC, rho_micro, gravity, surroundingMatl_indx,
                                        bc_kind, patch, face, bound_ptr, isNotInitialTimeStep);
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
      }  // if bcKind != LODI
    }  // child loop
    
    cout_BC_CC << "      "<< patch->getFaceName(face) << " \t " << bc_kind << " numChildren: " << numChildren 
                        << " nCellsTouched: " << nCells << " nCells_LODI: " << nCells_LODI[face] << endl;
    //__________________________________
    //  bulletproofing   
    Patch::FaceIteratorType type = Patch::ExtraPlusEdgeCells;
    int nFaceCells = numFaceCells(patch,  type, face);
    
    if(nCells != nFaceCells && (nCells_LODI[face] != nFaceCells && isNotInitialTimeStep)){
      ostringstream warn;
      warn << "ERROR: ICE: SetBC(press_CC) Boundary conditions were not set correctly ("
           << patch->getFaceName(face) << ", " << bc_kind  << " numChildren: " << numChildren 
           << " nCells Touched: " << nCells << " nCells on boundary: "<< nFaceCells << " nCells_LODI: " << nCells_LODI[face] <<") " << endl;
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
           MaterialManagerP& materialManager, 
           const int mat_id,
           DataWarehouse*,
           customBC_globalVars* globalVars,
           customBC_localVars* localVars,
           const bool isNotInitialTimeStep)
{

  if(patch->hasBoundaryFaces() == false){
    return;
  }
  cout_BC_CC << "-------- setBC (double) \t"<< desc << " mat_id = " 
             << mat_id <<  ", Patch: "<< patch->getID() << endl;
  Vector cell_dx = patch->dCell();
  // bool isNotInitialTimeStep = (materialManager->getCurrentTopLevelTimeStep() > 0);

  //__________________________________
  //  -Set the LODI BC's first, then let the other BC's wipe out what
  //   was set in the corners and edges. 
  //  -Ignore lodi bcs during intialization phase and when
  //   lv->setLodiBcs = false
  //__________________________________
  vector<int> nCells_LODI(Patch::numFaces);
  for (int f = 0; f < Patch::numFaces; f++) {
    nCells_LODI[f] = 0;  // bulletproofing
  }
  
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
    Patch::FaceType face = *iter;

    bool is_tempBC_lodi=  patch->haveBC(face,mat_id,"LODI","Temperature");  
    bool is_rhoBC_lodi =  patch->haveBC(face,mat_id,"LODI","Density");
    
    if( desc == "Temperature"  && is_tempBC_lodi  && isNotInitialTimeStep && localVars->setLodiBcs ){
      nCells_LODI[face] += FaceTemp_LODI(patch, face, var_CC, localVars->lodi, cell_dx, materialManager);
    }   
    else if (desc == "Density"  && is_rhoBC_lodi  && isNotInitialTimeStep && localVars->setLodiBcs){
      nCells_LODI[face] += FaceDensity_LODI(patch, face, var_CC, localVars->lodi, cell_dx);
    }
  }
  //__________________________________
  //  N O N  -  L O D I
  //__________________________________
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
                                                
      if (foundIterator && bc_kind != "LODI") {
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
        //  Custom Boundary Conditions
        if ( desc == "Temperature" && localVars->setMicroSlipBcs) {
          nCells += set_MicroSlipTemperature_BC( patch,face,var_CC,
                                                 bound_ptr, bc_kind, bc_value,
                                                 localVars->slip );
        }
        else if ( desc == "Temperature" && localVars->set_MMS_BCs) {
          nCells += set_MMS_Temperature_BC(patch, face, var_CC, 
                                             bound_ptr, bc_kind, 
                                             globalVars->mms,
                                             localVars->mms);
        }
        else if ( desc == "Temperature" && localVars->set_Sine_BCs) {
          nCells += set_Sine_Temperature_BC( patch, face, var_CC, 
                                             bound_ptr, bc_kind, 
                                             globalVars->sine,
                                             localVars->sine);
        }
        //__________________________________
        // Temperature and Gravity and ICE Matls
        // -Ignore this during intialization phase,
        //  since we backout the temperature field
        Vector gravity = globalVars->d_gravity;                        
        Material *matl = materialManager->getMaterial(mat_id);
        ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
        int P_dir =  patch->getFaceAxes(face)[0];  // principal direction
        
        if (gravity[P_dir] != 0 && desc == "Temperature" && ice_matl 
             && isNotInitialTimeStep) {
          ice_matl->getEOS()->
              hydrostaticTempAdjustment(face, patch, bound_ptr, gravity,
                                        gamma, cv, cell_dx, var_CC);
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
    
    cout_BC_CC << "      "<< patch->getFaceName(face) << " \t " << bc_kind << " numChildren: " << numChildren 
                        << " nCellsTouched: " << nCells << " nCells_LODI: " << nCells_LODI[face] << endl;
    //__________________________________
    //  bulletproofing
    Patch::FaceIteratorType type = Patch::ExtraPlusEdgeCells;
    int nFaceCells = numFaceCells(patch,  type, face);
    
    bool throwEx = false;
    if(nCells != nFaceCells && (nCells_LODI[face] != nFaceCells && isNotInitialTimeStep)){
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
           << " nCells Touched: " << nCells << " nCells on boundary: "<< nFaceCells << " nCells_LODI: " << nCells_LODI[face] <<") " << endl;
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
           MaterialManagerP& materialManager,
           const int mat_id,
           DataWarehouse* ,
           customBC_globalVars* globalVars,
           customBC_localVars* localVars,
           const bool isNotInitialTimeStep)
{
 if(patch->hasBoundaryFaces() == false){
    return;
  }
  cout_BC_CC <<"-------- setBC (Vector_CC) \t"<< desc <<" mat_id = " 
              <<mat_id<<  ", Patch: "<< patch->getID() << endl;
  
  // bool isNotInitialTimeStep = (materialManager->getCurrentTopLevelTimeStep() > 0);
  Vector cell_dx = patch->dCell();
  //__________________________________
  //  -Set the LODI BC's first, then let the other BC's wipe out what
  //   was set in the corners and edges. 
  //  -Ignore lodi bcs during intialization phase and when
  //   lv->setLodiBcs = false
  //__________________________________
  vector<int> nCells_LODI(Patch::numFaces);
  for (int f = 0; f < Patch::numFaces; f++) {
    nCells_LODI[f] = 0;  // bulletproofing
  }
  
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  for( vector<Patch::FaceType>::const_iterator iter  = bf.begin(); iter != bf.end(); ++iter ){
    Patch::FaceType face = *iter;
    bool is_velBC_lodi   = patch->haveBC(face,mat_id,"LODI","Velocity");
    
    if( desc == "Velocity"      && is_velBC_lodi 
        && isNotInitialTimeStep && localVars->setLodiBcs) {
        
      nCells_LODI[face] += FaceVel_LODI( patch, face, var_CC, localVars->lodi, cell_dx, materialManager);
    }
  }
  //__________________________________
  //  N O N  -  L O D I
  //__________________________________
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
      
      if (foundIterator && bc_kind != "LODI") {
        
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
        //  Custom Boundary Conditions
        else if ( localVars->setMicroSlipBcs ) {
          nCells += set_MicroSlipVelocity_BC( patch,face,var_CC,desc,
                                              bound_ptr, bc_kind, bc_value,
                                              localVars->slip );
        }
        else if ( localVars->set_MMS_BCs ) {
          nCells += set_MMS_Velocity_BC( patch, face, var_CC, desc,
                                         bound_ptr, bc_kind, materialManager,
                                         globalVars->mms,
                                         localVars->mms);
        }
        else if ( localVars->set_Sine_BCs ) {
          nCells += set_Sine_Velocity_BC( patch, face, var_CC, desc,
                                          bound_ptr, bc_kind, materialManager,
                                          globalVars->sine,
                                          localVars->sine );
        }
        else if ( localVars->set_inletVel_BCs ) {
          nCells += set_inletVelocity_BC( patch, face, var_CC, desc,
                                          bound_ptr, bc_kind, bc_value,
                                          globalVars->inletVel,
                                          localVars->inletVel );
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
    cout_BC_CC << "      "<< patch->getFaceName(face) << " \t " << bc_kind << " numChildren: " << numChildren 
                        << " nCellsTouched: " << nCells <<" nCells_LODI: " << nCells_LODI[face] << endl;
    //__________________________________
    //  bulletproofing
    Patch::FaceIteratorType type = Patch::ExtraPlusEdgeCells;
    int nFaceCells = numFaceCells(patch,  type, face);
    
    bool throwEx = false;
    if(nCells != nFaceCells && (nCells_LODI[face] != nFaceCells && isNotInitialTimeStep)){
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
           << " nCells Touched: " << nCells << " nCells on boundary: "<< nFaceCells << " nCells_LODI: " << nCells_LODI[face] <<") " << endl;
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
                      MaterialManagerP& materialManager,
                      const int mat_id)
{
  if(patch->hasBoundaryFaces() == false){
    return;
  }
  cout_BC_CC << "-------- setSpecificVolBC \t"<< desc <<" "
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
    
    cout_BC_CC << "      "<< patch->getFaceName(face) << " \t " << bc_kind << " numChildren: " << numChildren 
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
void BC_bulletproofing(const ProblemSpecP & prob_spec,
                       MaterialManagerP   & materialManager,
                       GridP              & grid )
{
  LevelP L0 = grid->getLevel(0);
  IntVector periodic_L0 = L0->getPeriodicBoundaries();
  
  Vector tagFace_minus(0,0,0);
  Vector tagFace_plus(0,0,0);
  
  ProblemSpecP grid_ps = prob_spec->findBlock("Grid");  
  ProblemSpecP bc_ps   = grid_ps->findBlock("BoundaryConditions");
  int numAllMatls = materialManager->getNumMatls();
  
  // If a face is periodic then is_press_BC_set = true
  map<string,int> is_press_BC_set;
  is_press_BC_set["x-"] = (periodic_L0.x() ==1) ? 1:0;
  is_press_BC_set["x+"] = (periodic_L0.x() ==1) ? 1:0;
  is_press_BC_set["y-"] = (periodic_L0.y() ==1) ? 1:0;
  is_press_BC_set["y+"] = (periodic_L0.y() ==1) ? 1:0;
  is_press_BC_set["z-"] = (periodic_L0.z() ==1) ? 1:0;
  is_press_BC_set["z+"] = (periodic_L0.z() ==1) ? 1:0;
  
  // Loop over all boundary conditions for a face...
  // this includes circles, rectangles, annulus
  for( ProblemSpecP face_ps = bc_ps->findBlock( "Face" ); face_ps != nullptr; face_ps=face_ps->findNextBlock( "Face" ) ) {
  
    map<string,bool>isBC_set;
    isBC_set["Temperature"] =false;
    isBC_set["Density"]     =false;
    isBC_set["Velocity"]    =false;            
    isBC_set["SpecificVol"] =true;  
    isBC_set["Symmetric"]   =false;    
                      
    map<string,string> face;
    face_ps->getAttributes(face);

    //loop through the attributes and find  (x-,x+,y-,y+... )
    string side = "nullptr";
    
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
    for( ProblemSpecP bc_iter = face_ps->findBlock( "BCType" ); bc_iter != nullptr; bc_iter = bc_iter->findNextBlock( "BCType" ) ) {
      map<string,string> bc_type;
      bc_iter->getAttributes(bc_type);
            
      // valid user input      
      if( bc_type["label"] != "Pressure"      && bc_type["label"] != "Temperature" && 
          bc_type["label"] != "SpecificVol"   && bc_type["label"] != "Velocity"    &&
          bc_type["label"] != "Density"       && bc_type["label"] != "Symmetric"   &&
          bc_type["label"] != "scalar-f"      && bc_type["label"] != "cumulativeEnergyReleased"){
        ostringstream warn;
        warn <<"\n   ERROR: ICE::\n   The boundary condition label ("<< bc_type["label"] <<") is not valid\n"
             << " Face:  " << face["side"] << " BCType " << bc_type["label"]<< endl;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }  
      
      // specified "all" for a 1 matl problem
      if (bc_type["id"] == "all" && numAllMatls == 1){
        ostringstream warn;
        warn <<"\n__________________________________\n"   
             << "  ERROR: ICE: This is a single material problem and you've specified 'BCType id = all' \n"
             << "  The boundary condition infrastructure treats 'all' and '0' as two separate materials, \n"
             << "  setting the boundary conditions twice on each face.  Set BCType id = '0' \n" 
             << "   Face:  " << face["side"] << " BCType " << bc_type["label"]<< endl;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      
      // symmetric BCs
      if ( bc_type["label"] == "Symmetric"){
        if (numAllMatls > 1 &&  bc_type["id"] != "all") {
          ostringstream warn;
          warn <<"\n__________________________________\n"   
             << "  ERROR: ICE: This is a multimaterial problem with a symmetric boundary condition\n"
             << "  You must have the id = all instead of id = "<< bc_type["id"] <<"\n"
             << "  Face:  " << face["side"] << " BCType " << bc_type["label"]<< endl;
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
           << "  ERROR: ICE: \n"
           << "  The "<<var<<" boundary condition for one of the materials has not been set \n"
           << "  Face:  " << face["side"] <<  endl;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
    }
    
    //__________________________________
    //  Has the pressure BC been set on this face;
    int isSet  = is_press_BC_set.count( side );

    if(isSet != 1){
      ostringstream warn;
      warn <<"\n__________________________________\n"   
         << "  ERROR: ICE: \n"
         << "  The pressure boundary condition has not been set OR has been set more than once \n"
         << "  Face:  " << side <<  endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
  } //face loop
  

  //__________________________________
  
  for(int dir = 0; dir<3; dir++){

    // Are boundary conditions set on a non-periodic face?
    if( periodic_L0[dir] == 0 && ( tagFace_minus[dir] == 0 || tagFace_plus[dir] == 0)){
      ostringstream warn;
      warn <<"\n__________________________________\n "
           << "ERROR: ICE: You must specify a boundary conditions in direction "<< dir << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);   
    }

    // Are boundary conditions set on a periodic face?
    if( periodic_L0[dir] == 1 && ( tagFace_minus[dir] == 1 || tagFace_plus[dir] == 1)){
      ostringstream warn;
      warn <<"\n__________________________________\n "
           << "ERROR: ICE: A periodic AND a normal boundary condition have been specifed for \n"
           << " direction: "<< dir << "  You can only have on or the other"<< endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);   
    }
  }
  
  //__________________________________
  //  Loop over all levels and verify that extra cells are 
  //  specified on non-periodic faces
  for (int L = 0; L < grid->numLevels(); L++) {
    const LevelP level = grid->getLevel(L);
    IntVector periodic   = level->getPeriodicBoundaries();
    IntVector extraCells = level->getExtraCells();
    
    for(int dir = 0; dir<3; dir++){
      if (periodic[dir] == 1 && extraCells[dir] != 0 ){
        ostringstream warn;
        warn <<"\n__________________________________\n "
             << "  ERROR: ICE:  A periodic boundary condition has been set in \n"
             << " direction: "<< dir << "  and the extraCells are not 0."<< endl;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);    
      }
      
      if (periodic[dir] == 0 && extraCells[dir] != 1 ){             // HARDWIRE:  Assume the number of extra cells = 1
        ostringstream warn;
        warn <<"\n__________________________________\n "
             << "  ERROR: ICE: The number of extra cells != 1, dir ("<< dir << ")\n";
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

//______________________________________________________________________
//______________________________________________________________________
//      S T U B   F U N C T I O N S

void setBC(CCVariable<double>& var,     
          const std::string& type,     // so gcc compiles
          const Patch* patch,  
          MaterialManagerP& materialManager,
          const int mat_id,
          DataWarehouse* new_dw,
          const bool isNotInitialTimeStep)
{
  customBC_globalVars* globalVars = scinew customBC_globalVars();
  customBC_localVars* localVars   = scinew customBC_localVars();
  constCCVariable<double> placeHolder;
  
  setBC(var, type, placeHolder, placeHolder, patch, materialManager, 
        mat_id, new_dw, globalVars, localVars,
        isNotInitialTimeStep );
  
  delete globalVars;
  delete localVars;
} 
//__________________________________  
void setBC(CCVariable<double>& press_CC,          
         std::vector<CCVariable<double> >& rho_micro,
         std::vector<constCCVariable<double> >& sp_vol,
         const int surroundingMatl_indx,
         const std::string& whichVar, 
         const std::string& kind, 
         const Patch* p, 
         MaterialManagerP& materialManager,
         const int mat_id, 
         DataWarehouse* new_dw,
         const bool isNotInitialTimeStep) {
         
  customBC_globalVars* globalVars = scinew customBC_globalVars();
  customBC_localVars* localVars   = scinew customBC_localVars();
  
  setBC(press_CC, rho_micro, sp_vol, surroundingMatl_indx,
        whichVar, kind, p, materialManager, mat_id, new_dw, globalVars, localVars,
        isNotInitialTimeStep); 

  delete globalVars;
  delete localVars;         
}   
//__________________________________       
void setBC(CCVariable<Vector>& variable,
          const std::string& type,
          const Patch* p,
          MaterialManagerP& materialManager,
          const int mat_id,
          DataWarehouse* new_dw,
          const bool isNotInitialTimeStep)
{ 
  customBC_globalVars* globalVars  = scinew customBC_globalVars();
  customBC_localVars* localVars    = scinew customBC_localVars();
   
  setBC( variable, type, p, materialManager, mat_id, new_dw,globalVars, localVars,
         isNotInitialTimeStep);
  
  delete globalVars;
  delete localVars; 
}


}  // using namespace Uintah

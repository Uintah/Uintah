/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
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

#include <fstream>

//-- Uintah framework includes --//
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/Iterator.h>
#include <Core/Grid/Variables/Stencil4.h>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/FVStaggered.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>
#include <expression/ExpressionFactory.h>

//-- Wasatch includes --//
#include "Operators/OperatorTypes.h"
#include "FieldTypes.h"
#include "BCHelperTools.h"
#include "Expressions/BoundaryConditions/BoundaryConditions.h"
#include "Expressions/BoundaryConditions/BoundaryConditionBase.h"

/**
 * \file BCHelperTools.cc
 */

//#define WASATCH_BC_DIAGNOSTICS

namespace Wasatch {

  //****************************************************************************  
  /**
   *  \ingroup WasatchCore
   *
   *  \brief This function grabs an iterator and a value associated with a
             boundary condition set in the input file.
   *
   */
  //****************************************************************************  
  template <typename T>
  bool get_iter_bcval_bckind_bcname( const Uintah::Patch* patch,
                                     const Uintah::Patch::FaceType face,
                                     const int child,
                                     const std::string& desc,
                                     const int mat_id,
                                     T& bc_value,
                                     Uintah::Iterator& bound_ptr,
                                     std::string& bc_kind,
                                     std::string& bc_face_name,
                                     std::string& bc_functor_name )
  {
    Uintah::Iterator nu;
    const Uintah::BoundCondBase* const bc = patch->getArrayBCValues( face, mat_id, desc, bound_ptr, nu, child );
    const Uintah::BoundCond<T>* const new_bcs = dynamic_cast<const Uintah::BoundCond<T>*>(bc);

    bc_value=T(-9);
    bc_kind="NotSet";
    bc_functor_name="none";
    if (new_bcs != 0) {      // non-symmetric
      bc_value = new_bcs->getValue();
      bc_kind  = new_bcs->getBCType();
      bc_face_name = new_bcs->getBCFaceName();
    }

    // if no name was specified for the face, then create a unique identifier
    if ( bc_face_name.compare("none") == 0 ) {
      std::ostringstream intToStr;
      intToStr << face << "_" << child;     
      bc_face_name = "face_" + intToStr.str();
    }
    
    delete bc;

    // Did I find an iterator
    return( bc_kind.compare("NotSet") != 0 );
  }


  //****************************************************************************  
  /**
   *  \ingroup WasatchCore
   *
   *  \brief Updates the rhs of a poisson equation to account for BCs 
   *
   */
  //****************************************************************************      
  void update_poisson_rhs( const Expr::Tag& poissonTag,
                            Uintah::CCVariable<Uintah::Stencil4>& poissonMatrix,
                            SVolField& poissonField,
                            SVolField& poissonRHS,
                            const Uintah::Patch* patch,
                            const int material )
  {
    /*
     ALGORITHM:
     1. loop over the patches
     2. For each patch, loop over materials
     3. For each material, loop over boundary faces
     4. For each boundary face, loop over its children
     5. For each child, get the cell faces and set appropriate
     boundary conditions
     */
//    // check if we have plus boundary faces on this patch
//    bool hasPlusFace[3] = {false,false,false};
//    if (patch->getBCType(Uintah::Patch::xplus)==Uintah::Patch::None) hasPlusFace[0]=true;
//    if (patch->getBCType(Uintah::Patch::yplus)==Uintah::Patch::None) hasPlusFace[1]=true;
//    if (patch->getBCType(Uintah::Patch::zplus)==Uintah::Patch::None) hasPlusFace[2]=true;
    // get the dimensions of this patch
    using SpatialOps::IntVec;
    const SCIRun::IntVector patchDim_ = patch->getCellHighIndex();
    const IntVec patchDim(patchDim_[0],patchDim_[1],patchDim_[2]);
    const Uintah::Vector spacing = patch->dCell();
    const double dx = spacing[0];
    const double dy = spacing[1];
    const double dz = spacing[2];
    const double dx2 = dx*dx;
    const double dy2 = dy*dy;
    const double dz2 = dz*dz;

    const std::string phiName = poissonTag.name();

    std::vector<Uintah::Patch::FaceType> bndFaces;
    patch->getBoundaryFaces(bndFaces);
    std::vector<Uintah::Patch::FaceType>::const_iterator faceIterator = bndFaces.begin();

    // loop over the boundary faces
    for( ; faceIterator!=bndFaces.end(); ++faceIterator ){
      Uintah::Patch::FaceType face = *faceIterator;

      //get the number of children
      const int numChildren = patch->getBCDataArray(face)->getNumberChildren(material);

      for( int child = 0; child<numChildren; ++child ){

        double bc_value = -9;
        std::string bc_kind = "NotSet";
        std::string bc_name = "none";
        std::string bc_functor_name = "none";        
        Uintah::Iterator bound_ptr;
        const bool foundIterator = get_iter_bcval_bckind_bcname( patch, face, child, phiName, material, bc_value, bound_ptr, bc_kind,bc_name,bc_functor_name);

        SCIRun::IntVector insideCellDir = patch->faceDirection(face);
        const bool hasExtraCells = ( patch->getExtraCells() != SCIRun::IntVector(0,0,0) );
        
        IntVec bcPointGhostOffset(0,0,0);
        double denom   = 1.0;
        double spacing = 1.0;
        
        switch( face ){
          case Uintah::Patch::xminus:  bcPointGhostOffset[0] = hasExtraCells?  1 : -1;  spacing = dx; denom = dx2;  break;
          case Uintah::Patch::xplus :  bcPointGhostOffset[0] = hasExtraCells? -1 :  1;  spacing = dx; denom = dx2;  break;
          case Uintah::Patch::yminus:  bcPointGhostOffset[1] = hasExtraCells?  1 : -1;  spacing = dy; denom = dy2;  break;
          case Uintah::Patch::yplus :  bcPointGhostOffset[1] = hasExtraCells? -1 :  1;  spacing = dy; denom = dy2;  break;
          case Uintah::Patch::zminus:  bcPointGhostOffset[2] = hasExtraCells?  1 : -1;  spacing = dz; denom = dz2;  break;
          case Uintah::Patch::zplus :  bcPointGhostOffset[2] = hasExtraCells? -1 :  1;  spacing = dz; denom = dz2;  break;
          default:                                                                                                  break;
        } // switch
        
        // cell offset used to calculate local cell index with respect to patch.
        const SCIRun::IntVector patchCellOffset = patch->getCellLowIndex(0);


        if( foundIterator ){

          if( bc_kind=="Dirichlet" ){
            for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
              SCIRun::IntVector bc_point_indices(*bound_ptr);
              
              bc_point_indices = bc_point_indices - patchCellOffset;
              
              const IntVec   intCellIJK( bc_point_indices[0],
                                         bc_point_indices[1],
                                         bc_point_indices[2] );
              const IntVec ghostCellIJK( bc_point_indices[0]+bcPointGhostOffset[0],
                                         bc_point_indices[1]+bcPointGhostOffset[1],
                                         bc_point_indices[2]+bcPointGhostOffset[2] );
              
              const int iInterior = poissonField.window_without_ghost().flat_index( hasExtraCells ? ghostCellIJK : intCellIJK  );
//            const int iGhost    = poissonField.window_without_ghost().flat_index( hasExtraCells? intCellIJK   : ghostCellIJK);
//            const double ghostValue = 2.0*bc_value - poissonField[iInterior];
//            poissonRHS[iInterior] += bc_value/denom;
              poissonRHS[iInterior] += 2.0*bc_value/denom;
            }
          }
          else if( bc_kind=="Neumann" ){
            for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
              SCIRun::IntVector bc_point_indices(*bound_ptr);
              
              bc_point_indices = bc_point_indices - patchCellOffset;
                            
              const IntVec   intCellIJK( bc_point_indices[0],
                                         bc_point_indices[1],
                                         bc_point_indices[2] );

              const IntVec ghostCellIJK( bc_point_indices[0]+bcPointGhostOffset[0],
                                         bc_point_indices[1]+bcPointGhostOffset[1],
                                         bc_point_indices[2]+bcPointGhostOffset[2] );

              const int iInterior = poissonField.window_without_ghost().flat_index( hasExtraCells? ghostCellIJK : intCellIJK  );
//            const int iGhost    = poissonField.window_without_ghost().flat_index( hasExtraCells? intCellIJK   : ghostCellIJK);
//            const double ghostValue = spacing*bc_value + poissonField[iInterior];
//            poissonRHS[iInterior] += ghostValue/denom;
              poissonRHS[iInterior] += spacing*bc_value/denom;
            }
          }
          else {
            return;
          }
        }
      } // child loop
    } // face loop
  }

  //****************************************************************************  
  /**
   *  \ingroup WasatchCore
   *
   *  \brief Updates the Poisson equation coefficient matrix to account for BCs 
   *
   */
  //****************************************************************************      
  void update_poisson_matrix( const Expr::Tag& poissonTag,
                              Uintah::CCVariable<Uintah::Stencil4>& poissonMatrix,
                              const Uintah::Patch* patch,
                              const int material)
  {
    /*
     ALGORITHM:
     1. loop over the patches
     2. For each patch, loop over materials
     3. For each material, loop over boundary faces
     4. For each boundary face, loop over its children
     5. For each child, get the cell faces and set appropriate
     boundary conditions
     */
    // get the dimensions of this patch
    using SpatialOps::IntVec;
    const SCIRun::IntVector uintahPatchDim = patch->getCellHighIndex();
    const IntVec patchDim(uintahPatchDim[0],uintahPatchDim[1],uintahPatchDim[2]);
    const Uintah::Vector spacing = patch->dCell();
    const double dx = spacing[0];
    const double dy = spacing[1];
    const double dz = spacing[2];
    const double dx2 = dx*dx;
    const double dy2 = dy*dy;
    const double dz2 = dz*dz;

    std::vector<Uintah::Patch::FaceType> bndFaces;
    patch->getBoundaryFaces(bndFaces);
    std::vector<Uintah::Patch::FaceType>::const_iterator faceIterator = bndFaces.begin();

    // loop over the boundary faces
    for( ; faceIterator!=bndFaces.end(); ++faceIterator ){
      Uintah::Patch::FaceType face = *faceIterator;
      //get the number of children
      const int numChildren = patch->getBCDataArray(face)->getNumberChildren(material);

      for( int child = 0; child<numChildren; ++child ){
        Uintah::Iterator boundPtr;

        double bcValue;
        std::string bcKind;
        std::string bcName = "none";
        //get_iter_bcval_bckind_bcname( patch, face, child, poissonTag.name(), material, bc_value, bound_ptr, bc_kind, bc_name,bc_functor_name);
        getBCKind( patch, face, child, poissonTag.name(), material, bcKind, bcName );
        
        if( bcKind.compare("Dirichlet")==0 || bcKind.compare("Neumann")==0 )
          getBCValue( patch, face, child, poissonTag.name(), material, bcValue );
        
        patch->getBCDataArray(face)->getCellFaceIterator( material, boundPtr, child );
        
        SCIRun::IntVector insideCellDir = patch->faceDirection(face);
        const bool hasExtraCells = ( patch->getExtraCells() != SCIRun::IntVector(0,0,0) );

        // cell offset used to calculate local cell index with respect to patch.
        const SCIRun::IntVector patchCellOffset = patch->getCellLowIndex(0);
        
        if( bcKind == "Dirichlet" ){ // pressure Outlet BC. don't forget to update pressure_rhs also.
          for( boundPtr.reset(); !boundPtr.done(); boundPtr++ ){
            SCIRun::IntVector bcPointIndices(*boundPtr);
            Uintah::Stencil4& coefs = poissonMatrix[hasExtraCells ? bcPointIndices - insideCellDir : bcPointIndices];
            
            switch(face){
              case Uintah::Patch::xminus: coefs.w = 0.0; coefs.p +=1.0/dx2; break;
              case Uintah::Patch::yminus: coefs.s = 0.0; coefs.p +=1.0/dy2; break;
              case Uintah::Patch::zminus: coefs.b = 0.0; coefs.p +=1.0/dz2; break;
              case Uintah::Patch::xplus :                coefs.p +=1.0/dx2; break;               
              case Uintah::Patch::yplus :                coefs.p +=1.0/dy2; break;
              case Uintah::Patch::zplus :                coefs.p +=1.0/dz2; break;
              default:                                   break;
            }
          }
        } else if (bcKind == "Neumann") { // outflow bc
          for( boundPtr.reset(); !boundPtr.done(); boundPtr++ ) {
            SCIRun::IntVector bcPointIndices(*boundPtr);
            
            Uintah::Stencil4& coefs = poissonMatrix[hasExtraCells ? bcPointIndices - insideCellDir : bcPointIndices];
            
            switch(face){
              case Uintah::Patch::xminus: coefs.w = 0.0; coefs.p -=1.0/dx2; break;
              case Uintah::Patch::xplus :                coefs.p -=1.0/dx2; break;
              case Uintah::Patch::yminus: coefs.s = 0.0; coefs.p -=1.0/dy2; break;
              case Uintah::Patch::yplus :                coefs.p -=1.0/dy2; break;
              case Uintah::Patch::zminus: coefs.b = 0.0; coefs.p -=1.0/dz2; break;
              case Uintah::Patch::zplus :                coefs.p -=1.0/dz2; break;
              default:                                                      break;
            }
          }
        } else if (bcKind == "OutletBC") { // outflow bc
          for( boundPtr.reset(); !boundPtr.done(); boundPtr++ ) {
            SCIRun::IntVector bcPointIndices(*boundPtr);
            
            Uintah::Stencil4& coefs = poissonMatrix[hasExtraCells ? bcPointIndices - insideCellDir : bcPointIndices];
            
            switch(face){
              case Uintah::Patch::xminus: coefs.w = 0.0; coefs.p += 1.0/dx2; break;
              case Uintah::Patch::xplus :                coefs.p += 1.0/dx2; break;
              case Uintah::Patch::yminus: coefs.s = 0.0; coefs.p += 1.0/dy2; break;
              case Uintah::Patch::yplus :                coefs.p += 1.0/dy2; break;
              case Uintah::Patch::zminus: coefs.b = 0.0; coefs.p += 1.0/dz2; break;
              case Uintah::Patch::zplus :                coefs.p += 1.0/dz2; break;
              default:                                                       break;
            }
          }          
        } else { // when no pressure BC is specified, it implies that we have a wall/inlet.
                 // note that when the face is periodic, then bound_ptr is empty
          for( boundPtr.reset(); !boundPtr.done(); boundPtr++ ) {
            SCIRun::IntVector bcPointIndices(*boundPtr);
            
            Uintah::Stencil4& coefs = poissonMatrix[hasExtraCells ? bcPointIndices - insideCellDir : bcPointIndices];
            
            switch(face){
              case Uintah::Patch::xminus: coefs.w = 0.0; coefs.p -=1.0/dx2; break;
              case Uintah::Patch::xplus :                coefs.p -=1.0/dx2; break;
              case Uintah::Patch::yminus: coefs.s = 0.0; coefs.p -=1.0/dy2; break;
              case Uintah::Patch::yplus :                coefs.p -=1.0/dy2; break;
              case Uintah::Patch::zminus: coefs.b = 0.0; coefs.p -=1.0/dz2; break;
              case Uintah::Patch::zplus :                coefs.p -=1.0/dz2; break;
              default:                                                      break;
            }
          }          
        }
      } // child loop
    } // face loop
  }

  //****************************************************************************  
  /**
   *  \ingroup WasatchCore
   *
   *  \brief Updates the coefficient matrix of a poisson equation to account a 
             reference value
   *
   */
  //****************************************************************************      
  void set_ref_poisson_coefs( Uintah::CCVariable<Uintah::Stencil4>& poissonMatrix,
                              const Uintah::Patch* patch,
                              const SCIRun::IntVector refCell )
  {
    using SCIRun::IntVector;
    std::ostringstream msg;
    if (patch->containsCell(refCell)) {

      Uintah::Stencil4& refCoef = poissonMatrix[refCell];
      refCoef.w = 0.0;
      refCoef.s = 0.0;
      refCoef.b = 0.0;
      refCoef.p = 1.0;
      
      const SCIRun::IntVector refCellip1 = refCell + IntVector(1,0,0);
      const SCIRun::IntVector refCelljp1 = refCell + IntVector(0,1,0);
      const SCIRun::IntVector refCellkp1 = refCell + IntVector(0,0,1);
      
      SCIRun::IntVector l, h;
      patch->getLevel()->findCellIndexRange(l,h);
      
      const int nx = h.x() - l.x();
      const int ny = h.y() - l.y();
      const int nz = h.z() - l.z();
      
      // if this patch owns x+1 cell, then set that cell's coefficients appropriately
      if ( patch->containsCell(refCellip1) && nx != 1 ) {
        poissonMatrix[refCellip1].w = 0.0;
      }

      if ( patch->containsCell(refCelljp1) && ny != 1) {
        poissonMatrix[refCelljp1].s = 0.0;
      }

      if ( patch->containsCell(refCellkp1) && nz != 1) {
        poissonMatrix[refCellkp1].b = 0.0;
      }

    }
  }

  //****************************************************************************  
  /**
   *  \ingroup WasatchCore
   *
   *  \brief Updates the rhs of a poisson equation to account for reference pressure 
   *
   */
  //****************************************************************************      
  void set_ref_poisson_rhs( SVolField& poissonRHS,
                             const Uintah::Patch* patch,
                             const double refpoissonValue,
                             const SCIRun::IntVector refCell )
  {
    using SCIRun::IntVector;
    std::ostringstream msg;
    if (patch->containsCell(refCell)) {
      // NOTE: for some reason, for the [0,0,0] cell, we are able to set the RHS of the "ghost" or "extra" cells although
      // the patch reports that those cells are not contained in that patch... Hence the crazy logic in the following lines to
      // take care of the [0,0,0] cell.
      
      SCIRun::IntVector l, h;
      patch->getLevel()->findCellIndexRange(l,h);
      const int nx = h.x() - l.x();
      const int ny = h.y() - l.y();
      const int nz = h.z() - l.z();
      const int oneDx = (nx == 1) ? 0 : 1;
      const int oneDy = (ny == 1) ? 0 : 1;
      const int oneDz = (nz == 1) ? 0 : 1;

      const SCIRun::IntVector refCellip1 = refCell + IntVector(1,0,0);
      const SCIRun::IntVector refCelljp1 = refCell + IntVector(0,1,0);
      const SCIRun::IntVector refCellkp1 = refCell + IntVector(0,0,1);
      const SCIRun::IntVector refCellim1 = refCell - IntVector(1,0,0);
      const SCIRun::IntVector refCelljm1 = refCell - IntVector(0,1,0);
      const SCIRun::IntVector refCellkm1 = refCell - IntVector(0,0,1);

      const bool hasXNeighbors = patch->containsCell(refCellip1) &&
                                (patch->containsCell(refCellim1) || refCell.x() == 0);

      const bool hasYNeighbors = patch->containsCell(refCelljp1) &&
                                (patch->containsCell(refCelljm1) || refCell.y() == 0);

      const bool hasZNeighbors = patch->containsCell(refCellkp1) &&
                                (patch->containsCell(refCellkm1) || refCell.z() == 0);

      // remember, indexing is local for SpatialOps so we must offset by the patch's low index
      const SCIRun::IntVector refCellWithOffset = refCell - patch->getCellLowIndex(0);
      const SpatialOps::IntVec refCellIJK(refCellWithOffset.x(),refCellWithOffset.y(),refCellWithOffset.z());
      const int irefCell = poissonRHS.window_without_ghost().flat_index(refCellIJK);
      poissonRHS[irefCell] = refpoissonValue;
      // modify rhs for neighboring cells
      const Uintah::Vector spacing = patch->dCell();
      const double dx2 = spacing[0]*spacing[0];
      const double dy2 = spacing[1]*spacing[1];
      const double dz2 = spacing[2]*spacing[2];

      bool fault = false;
      if (nx != 1) {
        if (hasXNeighbors) {
          poissonRHS[poissonRHS.window_without_ghost().flat_index(refCellIJK + SpatialOps::IntVec(1,0,0) ) ] += refpoissonValue/dx2;
          poissonRHS[poissonRHS.window_without_ghost().flat_index(refCellIJK - SpatialOps::IntVec(1,0,0) ) ] += refpoissonValue/dx2;
        } else {
          fault = true;
        }
      }

      if (ny != 1) {
        if (hasYNeighbors) {
          poissonRHS[poissonRHS.window_without_ghost().flat_index(refCellIJK + SpatialOps::IntVec(0,1,0) ) ] += refpoissonValue/dy2;
          poissonRHS[poissonRHS.window_without_ghost().flat_index(refCellIJK - SpatialOps::IntVec(0,1,0) ) ] += refpoissonValue/dy2;
        } else {
          fault = true;
        }
      }

      if (nz != 1) {
        if (hasZNeighbors) {
          poissonRHS[poissonRHS.window_without_ghost().flat_index(refCellIJK + SpatialOps::IntVec(0,0,1) ) ] += refpoissonValue/dz2;
          poissonRHS[poissonRHS.window_without_ghost().flat_index(refCellIJK - SpatialOps::IntVec(0,0,1) ) ] += refpoissonValue/dz2;
        } else {
          fault = true;
        }
      }

      if ( fault ) {
        msg << std::endl
        << "  Invalid reference poisson cell." << std::endl
        << "  The reference poisson cell as well as its north, east, and top neighbors must be contained in the same patch." << std::endl
        << std::endl;
        throw std::runtime_error( msg.str() );
      }
    }
  }

  //****************************************************************************  
  /**
   *  \ingroup WasatchCore
   *
   *  \brief Key function to process boundary conditions for poisson equation.
   *
   */
  //****************************************************************************      
  void process_poisson_bcs( const Expr::Tag& poissonTag,
                            SVolField& poissonField,
                            const Uintah::Patch* patch,
                            const int material )
  {
//    // check if we have plus boundary faces on this patch
//    bool hasPlusFace[3] = {false,false,false};
//    if (patch->getBCType(Uintah::Patch::xplus)==Uintah::Patch::None) hasPlusFace[0]=true;
//    if (patch->getBCType(Uintah::Patch::yplus)==Uintah::Patch::None) hasPlusFace[1]=true;
//    if (patch->getBCType(Uintah::Patch::zplus)==Uintah::Patch::None) hasPlusFace[2]=true;
    // get the dimensions of this patch
    using SpatialOps::IntVec;
    const SCIRun::IntVector uintahPatchDim = patch->getCellHighIndex();
    const IntVec patchDim( uintahPatchDim[0], uintahPatchDim[1], uintahPatchDim[2] );
    const Uintah::Vector spacing = patch->dCell();
    const double dx = spacing[0];
    const double dy = spacing[1];
    const double dz = spacing[2];
    
    const std::string phiName = poissonTag.name();
    
    std::vector<Uintah::Patch::FaceType> bndFaces;
    patch->getBoundaryFaces(bndFaces);
    std::vector<Uintah::Patch::FaceType>::const_iterator faceIterator = bndFaces.begin();
    
    // loop over the boundary faces
    for( ; faceIterator!=bndFaces.end(); ++faceIterator ){
      Uintah::Patch::FaceType face = *faceIterator;
      
      //get the number of children
      const int numChildren = patch->getBCDataArray(face)->getNumberChildren(material);
      
      for( int child = 0; child<numChildren; ++child ){
        
        double bc_value = -9;
        std::string bcKind = "NotSet";
        std::string bcName = "none";
        std::string bcFunctorName = "none";
        Uintah::Iterator boundPtr;
        const bool foundIterator = get_iter_bcval_bckind_bcname( patch, face, child, phiName, material, bc_value, boundPtr, bcKind, bcName,bcFunctorName);
        
        if( foundIterator ){
          
          SCIRun::IntVector insideCellDir = patch->faceDirection(face);
          const bool hasExtraCells = ( patch->getExtraCells() != SCIRun::IntVector(0,0,0) );
          
          IntVec bcPointGhostOffset(0,0,0);
          double spacing = 1.0;
          
          switch( face ){
            case Uintah::Patch::xminus:  bcPointGhostOffset[0] = hasExtraCells?  1 : -1;  spacing = dx;  break;
            case Uintah::Patch::xplus :  bcPointGhostOffset[0] = hasExtraCells? -1 :  1;  spacing = dx;  break;
            case Uintah::Patch::yminus:  bcPointGhostOffset[1] = hasExtraCells?  1 : -1;  spacing = dy;  break;
            case Uintah::Patch::yplus :  bcPointGhostOffset[1] = hasExtraCells? -1 :  1;  spacing = dy;  break;
            case Uintah::Patch::zminus:  bcPointGhostOffset[2] = hasExtraCells?  1 : -1;  spacing = dz;  break;
            case Uintah::Patch::zplus :  bcPointGhostOffset[2] = hasExtraCells? -1 :  1;  spacing = dz;  break;
            default:                                                                                     break;
          } // switch
          
          // cell offset used to calculate local cell index with respect to patch.
          const SCIRun::IntVector patchCellOffset = patch->getCellLowIndex(0);
          
          if (bcKind=="Dirichlet") {
            for( boundPtr.reset(); !boundPtr.done(); boundPtr++ ) {
              SCIRun::IntVector bcPointIndices(*boundPtr);

              bcPointIndices = bcPointIndices - patchCellOffset;
              
              const IntVec   intCellIJK( bcPointIndices[0],
                                         bcPointIndices[1],
                                         bcPointIndices[2] );
              const IntVec ghostCellIJK( bcPointIndices[0]+bcPointGhostOffset[0],
                                         bcPointIndices[1]+bcPointGhostOffset[1],
                                         bcPointIndices[2]+bcPointGhostOffset[2] );

              const int iInterior = poissonField.window_without_ghost().flat_index( hasExtraCells? ghostCellIJK : intCellIJK  );
              const int iGhost    = poissonField.window_without_ghost().flat_index( hasExtraCells? intCellIJK   : ghostCellIJK);              
              poissonField[iGhost] = 2.0*bc_value - poissonField[iInterior];              
            }
          } else if (bcKind=="Neumann") {
            for( boundPtr.reset(); !boundPtr.done(); boundPtr++ ) {
              SCIRun::IntVector bcPointIndices(*boundPtr);

              bcPointIndices = bcPointIndices - patchCellOffset;

              const IntVec   intCellIJK( bcPointIndices[0],
                                         bcPointIndices[1],
                                         bcPointIndices[2] );
              const IntVec ghostCellIJK( bcPointIndices[0]+bcPointGhostOffset[0],
                                         bcPointIndices[1]+bcPointGhostOffset[1],
                                         bcPointIndices[2]+bcPointGhostOffset[2] );

              const int iInterior = poissonField.window_without_ghost().flat_index( hasExtraCells? ghostCellIJK : intCellIJK  );
              const int iGhost    = poissonField.window_without_ghost().flat_index( hasExtraCells? intCellIJK   : ghostCellIJK);              
              poissonField[iGhost] = spacing*bc_value + poissonField[iInterior];
            }

          } else if (bcKind=="OutletBC") {
            for( boundPtr.reset(); !boundPtr.done(); boundPtr++ ) {
              SCIRun::IntVector bcPointIndices(*boundPtr);

              bcPointIndices = bcPointIndices - patchCellOffset;

              const IntVec   intCellIJK( bcPointIndices[0],
                                         bcPointIndices[1],
                                         bcPointIndices[2] );

              const IntVec extraCellIJK( bcPointIndices[0] + bcPointGhostOffset[0],
                                         bcPointIndices[1] + bcPointGhostOffset[1],
                                         bcPointIndices[2] + bcPointGhostOffset[2] );
              
              const int iInterior  = poissonField.window_without_ghost().flat_index( hasExtraCells? extraCellIJK : intCellIJK  );
              const int iGhost     = poissonField.window_without_ghost().flat_index( hasExtraCells? intCellIJK   : extraCellIJK);
              poissonField[iGhost] = -poissonField[iInterior];
            }
            
          } else {
            return;
          }
        }
      } // child loop
    }    
  }

} // namespace wasatch

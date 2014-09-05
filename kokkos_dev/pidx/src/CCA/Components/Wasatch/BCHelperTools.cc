/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
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
#include <Core/Grid/Variables/Stencil4.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/FVStaggered.h>
#include <spatialops/structured/FVStaggeredBCTools.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>
#include <expression/ExpressionFactory.h>

//-- Wasatch includes --//
#include "Operators/OperatorTypes.h"
#include "FieldTypes.h"
#include "BCHelperTools.h"
#include "Expressions/BoundaryConditions/ConstantBC.h"
#include "Expressions/BoundaryConditions/ParabolicBC.h"
#include "Expressions/BoundaryConditions/BoundaryConditionBase.h"

//#define WASATCH_BC_DIAGNOSTICS

namespace Wasatch {

  // This function returns true if the boundary condition is applied in the same direction
  // as the staggered field. For example, xminus/xplus on a XVOL field.
  bool is_staggered_bc( const Direction staggeredLocation,
                        const Uintah::Patch::FaceType face ){
    switch (staggeredLocation) {
      case XDIR:
        return ( (face==Uintah::Patch::xminus || face==Uintah::Patch::xplus));
        break;
      case YDIR:
        return ( (face==Uintah::Patch::yminus || face==Uintah::Patch::yplus));
        break;
      case ZDIR:
        return ( (face==Uintah::Patch::zminus || face==Uintah::Patch::zplus));
        break;
      default:
        return false;
        break;
    }
    return false;
  }

  //****************************************************************************    
  /**
   *
   *  \brief helps with staggered fields.
   *
   */
  //****************************************************************************      
  void get_face_offset( const Uintah::Patch::FaceType& face,
                        const bool hasExtraCells,
                        SpatialOps::structured::IntVec& faceOffset )
  {
    namespace SS = SpatialOps::structured;
    if( hasExtraCells ){
      switch( face ){
        case Uintah::Patch::xminus: faceOffset = SS::IntVec(1,0,0);   break;
        case Uintah::Patch::xplus : faceOffset = SS::IntVec(0,0,0);   break;
        case Uintah::Patch::yminus: faceOffset = SS::IntVec(0,1,0);   break;
        case Uintah::Patch::yplus : faceOffset = SS::IntVec(0,0,0);   break;
        case Uintah::Patch::zminus: faceOffset = SS::IntVec(0,0,1);   break;
        case Uintah::Patch::zplus : faceOffset = SS::IntVec(0,0,0);   break;
        default:                                                      break;
      }
    } else {
      switch( face ){
        case Uintah::Patch::xminus: faceOffset = SS::IntVec(0,0,0);   break;
        case Uintah::Patch::xplus : faceOffset = SS::IntVec(1,0,0);   break;
        case Uintah::Patch::yminus: faceOffset = SS::IntVec(0,0,0);   break;
        case Uintah::Patch::yplus : faceOffset = SS::IntVec(0,1,0);   break;
        case Uintah::Patch::zminus: faceOffset = SS::IntVec(0,0,0);   break;
        case Uintah::Patch::zplus : faceOffset = SS::IntVec(0,0,1);   break;
        default:                                                      break;
      }
    }
  }

  //****************************************************************************    
  /**
   *  \ingroup WasatchCore
   *
   *  \brief This function determines the point(s) on which we want to set bcs.
   *
   */
  //****************************************************************************    
  void get_bc_points_ijk( const Direction staggeredLocation,
                          const Uintah::Patch::FaceType face,
                          const SpatialOps::structured::BCSide bcSide,
                          const std::string& bc_kind,
                          const SCIRun::IntVector bc_point_indices,
                          const SpatialOps::structured::IntVec faceOffset,
                          const SCIRun::IntVector insideCellDir,
                          SpatialOps::structured::IntVec& bcPointIJK,
                          SpatialOps::structured::IntVec& ghostPointIJK,
                          const bool hasExtraCells)
  {
    namespace SS = SpatialOps::structured;

    // interiorCellIJK is the ijk boundary cell index returned by uintah. Depending
    // on whether we use extra cells or not, interiorCellIJK may have different
    // meanings. Two cases arise here.
    // 1. Using ExtraCells: interiorCellIJK corresponds to the extra cell index  starting at [-1,-1,-1].
    // 2. Using GhostCells: if we have ghost cells, then interiorCellIJK corresponds to the
    //    interior cell adjacent to the boundary starting at [0,0,0].
    // NOTE: these are the indices of the scalar cells.
    // One of the caveats of using ghostcells is that we will always miss setting BCs on the corner cells.
    const SS::IntVec interiorCellIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]);


    // bndFaceIJK returns the index of boundary face starting at [0,0,0]
    // This is done by offsetting the cell index returned by uintah.
    // Two cases arise here:
    // 1. ExtraCells: When extra cells are present, we offset the MINUS-FACE cells
    //    by +1 while the PLUS-FACE cells remain the same.
    // 2. GhostCells: When using ghost cells, we offset the PLUS-FACE cells by
    //    +1 while the MINUS-FACE cells remain the same.
    const SS::IntVec bndFaceIJK = interiorCellIJK + faceOffset;

    // insideCellDir returns the face direction:
    // x-: [-1, 0, 0]
    // x+: [ 1, 0, 0]
    // y-: [ 0,-1, 0]
    // y+: [ 0, 1, 0]
    // z-: [ 0, 0,-1]
    // z+: [ 0, 0, 1]
    const SS::IntVec stgrdBndFaceIJK( bc_point_indices[0] + insideCellDir[0],
                                      bc_point_indices[1] + insideCellDir[1],
                                      bc_point_indices[2] + insideCellDir[2] );

    const SS::IntVec interiorStgrdCellIJK( bc_point_indices[0] - insideCellDir[0],
                                     bc_point_indices[1] - insideCellDir[1],
                                     bc_point_indices[2] - insideCellDir[2] );

    const SS::IntVec stgrdGhostPlusBndFaceIJK( bc_point_indices[0] + 2*insideCellDir[0],
                                     bc_point_indices[1] + 2*insideCellDir[1],
                                     bc_point_indices[2] + 2*insideCellDir[2] );

    if (is_staggered_bc(staggeredLocation,face) ) {
      switch (bcSide) {
        case SpatialOps::structured::MINUS_SIDE: {
          if (hasExtraCells) {
            bcPointIJK = bndFaceIJK;
            ghostPointIJK = interiorCellIJK;
          } else {
            // this stuff works with boundary layer cells
            bcPointIJK = interiorCellIJK;
            ghostPointIJK = stgrdBndFaceIJK;
          }
          break;
        }
        case SpatialOps::structured::PLUS_SIDE: {
          if (hasExtraCells) {
            bcPointIJK = interiorCellIJK;
            //bcPointIJK = (bc_kind.compare("Dirichlet")==0 ? interiorCellIJK : interiorCellIJK);
            ghostPointIJK = (bc_kind.compare("Dirichlet")==0 ? stgrdBndFaceIJK : interiorCellIJK);
          } else {
            bcPointIJK = bndFaceIJK;
            // this stuff works with boundary layer cells
            //bcPointIJK = (bc_kind.compare("Dirichlet")==0 ? stgrdBndFaceIJK : stgrdBndFaceIJK);
            ghostPointIJK = (bc_kind.compare("Dirichlet")==0 ? stgrdGhostPlusBndFaceIJK : stgrdBndFaceIJK);
          }
          break;
        }
        default:
          break;
      }
    } else {
      bcPointIJK = bndFaceIJK;
    }
  }

  //****************************************************************************    
  /**
   *  \ingroup WasatchCore
   *
   *  \brief This function sets the boundary condition on a collection of points. 
   this gets called from set_bc_on_face.
   *
   */
  //****************************************************************************    
  template < typename FieldT, typename BCOpT >
  void set_bc_on_points( const Uintah::Patch* const patch,
                        const GraphHelper& gh,
                        const Expr::Tag phiTag,
                        const std::string fieldName,
                        const std::vector<SpatialOps::structured::IntVec>& bcPointsIJK,
                        const std::vector<SpatialOps::structured::IntVec>& ghostPointsIJK,
                        const SpatialOps::structured::BCSide bcSide,
                        const double bcValue,
                        const SpatialOps::OperatorDatabase& opdb,
                        const bool isStaggered,
                        const std::string& bc_kind,
                        const std::string& bc_name,
                        const std::string& bc_functor_name)
  {
    using namespace SpatialOps::structured;
    Expr::ExpressionFactory& factory = *gh.exprFactory;
    
    const bool withoutGhost = true;
    SpatialOps::structured::MemoryWindow fieldWindow = get_memory_window_for_uintah_field<FieldT>(patch, withoutGhost);        
    BCOpT bcOp(fieldWindow, bcPointsIJK, bcSide, 0, opdb );
    double cg = bcOp.getGhostCoef();
    double ci = bcOp.getInteriorCoef();
    std::vector<int> flatGhostPoints;   // = bcOp.getFlatGhostPoints();
    std::vector<int> flatInteriorPoints;// = bcOp.getFlatInteriorPoints();

    // construct flat indices for staggered fields
    // NOTE ON STAGGERED FIELDS: To avoid random values in the ghost (extra) cell
    // layer, we also set the SAME BC on those faces if we have Dirichlet conditions.
    if (isStaggered && bc_kind.compare("Dirichlet")==0) {
      for( std::vector<IntVec>::const_iterator interiorIJKIter = bcPointsIJK.begin(),
          ghostIJKIter = ghostPointsIJK.begin();
          interiorIJKIter != bcPointsIJK.end() && ghostIJKIter != ghostPointsIJK.end();
          ++interiorIJKIter, ++ghostIJKIter )
      {
        flatInteriorPoints.push_back(fieldWindow.flat_index(*interiorIJKIter));
        flatInteriorPoints.push_back(fieldWindow.flat_index(*ghostIJKIter));

        flatGhostPoints.push_back(fieldWindow.flat_index(*interiorIJKIter));
        flatGhostPoints.push_back(fieldWindow.flat_index(*ghostIJKIter));
      }
      
      cg = 1.0;
      ci = 0.0;
      
    } else {
      flatGhostPoints = bcOp.getFlatGhostPoints();
      flatInteriorPoints = bcOp.getFlatInteriorPoints();
    }
    
    assert( flatInteriorPoints.size() == flatGhostPoints.size() );
        
    // create unique names for the modifier expressions
    std::string strPatchID;
    std::ostringstream intToStr;
    intToStr << patch->getID();
    strPatchID = intToStr.str();
    
    Expr::Tag modTag;
    Expr::Tag modTagGhost;
    
    Expr::ExpressionBuilder* builder = NULL;

    // create constant bc expressions. These are not created from the input file.
    if (bc_functor_name.compare("none")==0) { // constant bc
      modTag = Expr::Tag(fieldName + "_bc_" + bc_name + "_patch_" + strPatchID,Expr::STATE_NONE);
      builder = new typename ConstantBC<FieldT>::Builder(modTag, bcValue);
      factory.register_expression( builder, true );
    } else { // expression
      modTag = Expr::Tag(bc_functor_name,Expr::STATE_NONE);
    }
    // attach the modifier expression to the target expression
    factory.attach_modifier_expression( modTag, phiTag, patch->getID(), true );
    
    // now retrieve the modifier expression and set the ghost and interior points
    BoundaryConditionBase<FieldT>& modExpr =
      dynamic_cast<BoundaryConditionBase<FieldT>&>( factory.retrieve_modifier_expression( modTag, patch->getID(), false ) );

    
    // this is needed for bc expressions that require global uintah indexing, e.g. TurbulentInletBC
    const SCIRun::IntVector sciPatchCellOffset = patch->getCellLowIndex(0);
    SpatialOps::structured::IntVec patchCellOffset(sciPatchCellOffset.x(), sciPatchCellOffset.y(), sciPatchCellOffset.z());
    modExpr.set_patch_cell_offset(patchCellOffset);
    
    // set the ghost and interior points as well as coefficients
    modExpr.set_ghost_coef(cg);
    modExpr.set_ghost_points(flatGhostPoints);
    modExpr.set_interior_coef(ci);
    modExpr.set_interior_points(flatInteriorPoints);
  }
  
  //****************************************************************************    
  /**
   *  @struct BCOpTypeSelectorBase
   *
   *  @brief This templated struct is used to simplify boundary
   *         condition operator selection.
   */
  //****************************************************************************    
  template< typename FieldT, typename BCEvalT>
  struct BCOpTypeSelectorBase
  {
  private:
    typedef OpTypes<FieldT> Ops;

  public:
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::InterpC2FX, BCEvalT >   DirichletX;
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::InterpC2FY, BCEvalT >   DirichletY;
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::InterpC2FZ, BCEvalT >   DirichletZ;

    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::GradX,      BCEvalT >   NeumannX;
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::GradY,      BCEvalT >   NeumannY;
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::GradZ,      BCEvalT >   NeumannZ;
  };
  //
  template< typename FieldT, typename BCEvalT>
  struct BCOpTypeSelector : public BCOpTypeSelectorBase<FieldT, BCEvalT>
  { };
  // partial specialization with inheritance for ZVolFields
  template< typename BCEvalT>
  struct BCOpTypeSelector<XVolField,BCEvalT> : public BCOpTypeSelectorBase<XVolField,BCEvalT>
  {
  private:
    typedef OpTypes<XVolField> Ops;

  public:
    typedef typename SpatialOps::structured::BoundaryConditionOp< typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::GradientX, XVolField, XVolField >::type, BCEvalT> NeumannX;
  };
  // partial specialization with inheritance for YVolFields
  template< typename BCEvalT>
  struct BCOpTypeSelector<YVolField,BCEvalT> : public BCOpTypeSelectorBase<YVolField,BCEvalT>
  {
  private:
    typedef OpTypes<YVolField> Ops;

  public:
    typedef typename SpatialOps::structured::BoundaryConditionOp< typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::GradientY, YVolField, YVolField >::type, BCEvalT> NeumannY;
  };
  // partial specialization with inheritance for ZVolFields
  template< typename BCEvalT>
  struct BCOpTypeSelector<ZVolField,BCEvalT> : public BCOpTypeSelectorBase<ZVolField,BCEvalT>
  {
  private:
    typedef OpTypes<ZVolField> Ops;

  public:
    typedef typename SpatialOps::structured::BoundaryConditionOp< typename SpatialOps::structured::OperatorTypeBuilder<SpatialOps::GradientZ, ZVolField, ZVolField >::type, BCEvalT> NeumannZ;
  };
  //
  template< typename BCEvalT >
  struct BCOpTypeSelector<FaceTypes<XVolField>::XFace,BCEvalT>
  {
  public:
    typedef typename SpatialOps::structured::BoundaryConditionOp< typename SpatialOps::structured::OperatorTypeBuilder<Interpolant, SpatialOps::structured::XSurfXField, SpatialOps::structured::XVolField >::type, BCEvalT> DirichletX;
    typedef typename SpatialOps::structured::BoundaryConditionOp< typename SpatialOps::structured::OperatorTypeBuilder<Divergence, SpatialOps::structured::XSurfXField, SpatialOps::structured::XVolField >::type, BCEvalT> NeumannX;
  };
  //
  template< typename BCEvalT >
  struct BCOpTypeSelector<FaceTypes<YVolField>::YFace,BCEvalT>
  {
  public:
    typedef typename SpatialOps::structured::BoundaryConditionOp< typename SpatialOps::structured::OperatorTypeBuilder<Interpolant, SpatialOps::structured::YSurfYField, SpatialOps::structured::YVolField >::type, BCEvalT> DirichletY;
    typedef typename SpatialOps::structured::BoundaryConditionOp< typename SpatialOps::structured::OperatorTypeBuilder<Divergence, SpatialOps::structured::YSurfYField, SpatialOps::structured::YVolField >::type, BCEvalT> NeumannY;
  };
  //
  template< typename BCEvalT >
  struct BCOpTypeSelector<FaceTypes<ZVolField>::ZFace,BCEvalT>
  {
  public:
    typedef typename SpatialOps::structured::BoundaryConditionOp< typename SpatialOps::structured::OperatorTypeBuilder<Interpolant, SpatialOps::structured::ZSurfZField, SpatialOps::structured::ZVolField >::type, BCEvalT> DirichletZ;
    typedef typename SpatialOps::structured::BoundaryConditionOp< typename SpatialOps::structured::OperatorTypeBuilder<Divergence, SpatialOps::structured::ZSurfZField, SpatialOps::structured::ZVolField >::type, BCEvalT> NeumannZ;
  };

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
                             SCIRun::Iterator& bound_ptr,
                             std::string& bc_kind,
                             std::string& bc_face_name,
                             std::string& bc_functor_name)
  {
    SCIRun::Iterator nu;
    const Uintah::BoundCondBase* const bc = patch->getArrayBCValues( face, mat_id, desc, bound_ptr, nu, child );
    const Uintah::BoundCond<T>* const new_bcs = dynamic_cast<const Uintah::BoundCond<T>*>(bc);

    bc_value=T(-9);
    bc_kind="NotSet";
    bc_functor_name="none";
    if (new_bcs != 0) {      // non-symmetric
      bc_value = new_bcs->getValue();
      bc_kind =  new_bcs->getBCType__NEW();
      bc_face_name = new_bcs->getBCFaceName();
      bc_functor_name = new_bcs->getFunctorName();
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
   *  \brief This function sets the boundary conditions on every point of a
             given face.
   *
   */
  //****************************************************************************    
  template < typename FieldT, typename BcT >
  void set_bcs_on_face (SCIRun::Iterator& bound_ptr,
                           const Uintah::Patch::FaceType& face,
                           const Direction staggeredLocation,
                           const Uintah::Patch* const patch,
                           const GraphHelper& graphHelper,
                           const Expr::Tag phiTag,
                           const std::string fieldName,
                           const double bc_value,
                           const SpatialOps::OperatorDatabase& opdb,
                           const std::string& bc_kind,
                           const std::string& bc_name,
                           const std::string& bc_functor_name,
                           const SpatialOps::structured::BCSide bcSide,
                           const SpatialOps::structured::IntVec& faceOffset,
                           const bool hasExtraCells)
  {
    namespace SS = SpatialOps::structured;
    typedef SS::ConstValEval BCEvalT; // basic functor for constant functions.
    SCIRun::IntVector patchCellOffset = patch->getCellLowIndex(0); // cell offset used to calculate local cell index with respect to patch.
    SCIRun::IntVector insideCellDir = patch->faceDirection(face);
#ifdef WASATCH_BC_DIAGNOSTICS    
    std::cout << "SETTING BOUNDARY CONDITION ON "<< fieldName << " FACE:" << face << " ON PATCH " << patch->getID() << std::endl;
#endif
    
    std::vector<SS::IntVec> bcPointsIJK;
    std::vector<SS::IntVec> ghostPointsIJK;
    
    for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
      SCIRun::IntVector bc_point_indices(*bound_ptr);
      //std::cout << "bc point indices " << bc_point_indices << std::endl << std::endl;
      bc_point_indices = bc_point_indices - patchCellOffset; // get local cell index (with respect to patch). SpatialOps needs local cell indices.
      SS::IntVec bcPointIJK;
      SS::IntVec ghostPointIJK;
      get_bc_points_ijk ( staggeredLocation, face, bcSide, bc_kind, bc_point_indices, faceOffset, insideCellDir, bcPointIJK,ghostPointIJK, hasExtraCells);
    
      bcPointsIJK.push_back(bcPointIJK);
      ghostPointsIJK.push_back(ghostPointIJK);
    }
    
      set_bc_on_points< FieldT, BcT >( patch, graphHelper, phiTag,fieldName,
                                     bcPointsIJK, ghostPointsIJK, bcSide, bc_value,
                                     opdb, is_staggered_bc(staggeredLocation, face),
                                     bc_kind, bc_name, bc_functor_name);
    
  }

  //****************************************************************************  
  /**
   *  \ingroup WasatchCore
   *
   *  \brief Given a boundary face (xminus for example), this function does
   *         the preprocessing for setting the boundary conditions on that face.
             This function will subsequently call set_bc_on_face.
   *
   *  \param bound_ptr This is the boundary pointer passed by Uintah
   *
   *  \param face This is the face location passed by Uintah (xminus, xplus...)
   *
   *  \param staggeredLocation This is the staggered location of the field on
   *         which the boundary condition is applied
   *
   *  \param patch a const pointer to a const Uintah::Patch
   *
   *  \param graphHelper The graph which contains the expression associated with
             the field we are setting the bc on
   *
   *  \param phiTag An Expr::Tag that has contains the tag of the expression
             we're setting the bc on
   *  \param fieldName A string containing the name of the field for which we
             want to apply the bc. This is needed for expressions that compute
             multiple fields such as the pressure.
   *  \param bc_value A double containing the value of the boundary condition
   *  \param opdb The operators databse
   *  \param bc_kind The type of bc: Dirichlet or Neumann
   */
  //****************************************************************************  
  template < typename FieldT >
  void process_bcs_on_face( SCIRun::Iterator& bound_ptr,
                            const Uintah::Patch::FaceType& face,
                            const Direction staggeredLocation,
                            const Uintah::Patch* const patch,
                            const GraphHelper& graphHelper,
                            const Expr::Tag& phiTag,
                            const std::string& fieldName,
                            const double bc_value,
                            const SpatialOps::OperatorDatabase& opdb,
                            const std::string& bc_kind,
                            const std::string& bc_name,
                            const std::string& bc_functor_name)
  {
    namespace SS = SpatialOps::structured;
    typedef SS::ConstValEval BCEvalT; // basic functor for constant functions.
    typedef BCOpTypeSelector<FieldT,BCEvalT> BCOpT;
    SS::IntVec faceOffset(0,0,0);
    SCIRun::IntVector insideCellDir = patch->faceDirection(face);
    const bool hasExtraCells = ( patch->getExtraCells() != SCIRun::IntVector(0,0,0) );
    get_face_offset( face, hasExtraCells, faceOffset );

    if( bc_kind.compare("Dirichlet")==0 ){
      switch( face ){
        case Uintah::Patch::xminus:
          set_bcs_on_face<FieldT,typename BCOpT::DirichletX>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name,bc_functor_name, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::xplus:
          set_bcs_on_face<FieldT,typename BCOpT::DirichletX>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yminus:
          set_bcs_on_face<FieldT,typename BCOpT::DirichletY>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yplus:
          set_bcs_on_face<FieldT,typename BCOpT::DirichletY>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::zminus:
          set_bcs_on_face<FieldT,typename BCOpT::DirichletZ>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::zplus:
          set_bcs_on_face<FieldT,typename BCOpT::DirichletZ>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        default:
          break;
      }

    } else if (bc_kind.compare("Neumann")==0 ){
      switch( face ){
        case Uintah::Patch::xminus:
          set_bcs_on_face<FieldT,typename BCOpT::NeumannX>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::xplus:
          set_bcs_on_face<FieldT,typename BCOpT::NeumannX>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yminus:
          set_bcs_on_face<FieldT,typename BCOpT::NeumannY>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yplus:
          set_bcs_on_face<FieldT,typename BCOpT::NeumannY>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::zminus:
          set_bcs_on_face<FieldT,typename BCOpT::NeumannZ>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::zplus:
          set_bcs_on_face<FieldT,typename BCOpT::NeumannZ>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        default:
          break;
      }
    }

  }

  //****************************************************************************  
  // Specialization for normal stress and convective flux for xvol fields
  //****************************************************************************    
  template<>
  void process_bcs_on_face<FaceTypes<XVolField>::XFace>( SCIRun::Iterator& bound_ptr,
                                                         const Uintah::Patch::FaceType& face,
                                                         const Direction staggeredLocation,
                                                         const Uintah::Patch* const patch,
                                                         const GraphHelper& graphHelper,
                                                         const Expr::Tag& phiTag,
                                                         const std::string& fieldName,
                                                         const double bc_value,
                                                         const SpatialOps::OperatorDatabase& opdb,
                                                         const std::string& bc_kind,
                                                         const std::string& bc_name,
                                                         const std::string& bc_functor_name)
  {
    namespace SS = SpatialOps::structured;
    typedef FaceTypes<XVolField>::XFace FieldT;
    typedef SS::ConstValEval BCEvalT; // basic functor for constant functions.
    typedef BCOpTypeSelector<FieldT,BCEvalT> BCOpT;

    SS::IntVec faceOffset(0,0,0);
    SCIRun::IntVector insideCellDir = patch->faceDirection(face);
    const bool hasExtraCells = (patch->getExtraCells() != SCIRun::IntVector(0,0,0));
    get_face_offset(face, hasExtraCells, faceOffset);

    if( bc_kind.compare("Dirichlet")==0 ){
      switch( face ){
        case Uintah::Patch::xminus:
          set_bcs_on_face<FieldT, BCOpT::DirichletX>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::xplus:
          set_bcs_on_face<FieldT, BCOpT::DirichletX>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yminus:  case Uintah::Patch::yplus:
        case Uintah::Patch::zminus:  case Uintah::Patch::zplus:
          throw Uintah::ProblemSetupException( "Invalid face", __FILE__, __LINE__ );
          break;
        default:
          break;
      }

    } else if (bc_kind.compare("Neumann")==0 ){
      switch( face ){
        case Uintah::Patch::xminus:
          set_bcs_on_face<FieldT, BCOpT::NeumannX>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::xplus:
          set_bcs_on_face<FieldT, BCOpT::NeumannX>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yminus:  case Uintah::Patch::yplus:
        case Uintah::Patch::zminus:  case Uintah::Patch::zplus:
          throw Uintah::ProblemSetupException( "Invalid face", __FILE__, __LINE__ );
          break;
        default:
          break;
      }
    }

  }

  //****************************************************************************  
  // Specialization for normal stress and convective flux for yvol fields
  //****************************************************************************    
  template<>
  void process_bcs_on_face<FaceTypes<YVolField>::YFace>( SCIRun::Iterator& bound_ptr,
                                                         const Uintah::Patch::FaceType& face,
                                                         const Direction staggeredLocation,
                                                         const Uintah::Patch* const patch,
                                                         const GraphHelper& graphHelper,
                                                         const Expr::Tag& phiTag,
                                                         const std::string& fieldName,
                                                         const double bc_value,
                                                         const SpatialOps::OperatorDatabase& opdb,
                                                         const std::string& bc_kind,
                                                         const std::string& bc_name,
                                                         const std::string& bc_functor_name)
  {
    namespace SS = SpatialOps::structured;
    typedef FaceTypes<YVolField>::YFace FieldT;
    typedef SS::ConstValEval BCEvalT; // basic functor for constant functions.
    typedef BCOpTypeSelector<FieldT,BCEvalT> BCOpT;

    SS::IntVec faceOffset(0,0,0);
    SCIRun::IntVector insideCellDir = patch->faceDirection(face);
    const bool hasExtraCells = (patch->getExtraCells() != SCIRun::IntVector(0,0,0));
    get_face_offset(face, hasExtraCells, faceOffset);

    if( bc_kind.compare("Dirichlet")==0 ){
      switch( face ){
        case Uintah::Patch::yminus:
          set_bcs_on_face<FieldT, BCOpT::DirichletY>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yplus:
          set_bcs_on_face<FieldT, BCOpT::DirichletY>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::xminus:  case Uintah::Patch::xplus:
        case Uintah::Patch::zminus:  case Uintah::Patch::zplus:
          throw Uintah::ProblemSetupException( "Invalid face", __FILE__, __LINE__ );
          break;
        default:
          break;
      }

    } else if (bc_kind.compare("Neumann")==0 ){
      switch( face ){
        case Uintah::Patch::yminus:
          set_bcs_on_face<FieldT, BCOpT::NeumannY>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yplus:
          set_bcs_on_face<FieldT, BCOpT::NeumannY>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::xminus:  case Uintah::Patch::xplus:
        case Uintah::Patch::zminus:  case Uintah::Patch::zplus:
          throw Uintah::ProblemSetupException( "Invalid face", __FILE__, __LINE__ );
          break;
        default:
          break;
      }
    }
  }

  //****************************************************************************  
  // Specialization for normal stress and convective flux for zvol fields
  //****************************************************************************    
  template<>
  void process_bcs_on_face<FaceTypes<ZVolField>::ZFace>( SCIRun::Iterator& bound_ptr,
                                                         const Uintah::Patch::FaceType& face,
                                                         const Direction staggeredLocation,
                                                         const Uintah::Patch* const patch,
                                                         const GraphHelper& graphHelper,
                                                         const Expr::Tag& phiTag,
                                                         const std::string& fieldName,
                                                         const double bc_value,
                                                         const SpatialOps::OperatorDatabase& opdb,
                                                         const std::string& bc_kind,
                                                         const std::string& bc_name,
                                                         const std::string& bc_functor_name)
  {
    namespace SS = SpatialOps::structured;
    typedef FaceTypes<ZVolField>::ZFace FieldT;
    typedef SS::ConstValEval BCEvalT; // basic functor for constant functions.
    typedef BCOpTypeSelector<FieldT,BCEvalT> BCOpT;
    SS::IntVec faceOffset(0,0,0);
    SCIRun::IntVector insideCellDir = patch->faceDirection(face);
    const bool hasExtraCells = (patch->getExtraCells() != SCIRun::IntVector(0,0,0));
    get_face_offset(face, hasExtraCells, faceOffset);

    if( bc_kind.compare("Dirichlet")==0 ){
      switch( face ){
        case Uintah::Patch::zminus:
          set_bcs_on_face<FieldT, BCOpT::DirichletZ>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::zplus:
          set_bcs_on_face<FieldT, BCOpT::DirichletZ>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yminus:  case Uintah::Patch::yplus:
        case Uintah::Patch::xminus:  case Uintah::Patch::xplus:
          throw Uintah::ProblemSetupException( "Invalid face", __FILE__, __LINE__ );
          break;
        default:
          break;
      }

    } else if (bc_kind.compare("Neumann")==0 ){
      switch( face ){
        case Uintah::Patch::zminus:
          set_bcs_on_face<FieldT, BCOpT::NeumannZ>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::zplus:
          set_bcs_on_face<FieldT, BCOpT::NeumannZ>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yminus:  case Uintah::Patch::yplus:
        case Uintah::Patch::xminus:  case Uintah::Patch::xplus:
          throw Uintah::ProblemSetupException( "Invalid face", __FILE__, __LINE__ );
          break;
        default:
          break;
      }
    }
  }

  //****************************************************************************  
  /**
   *  \ingroup WasatchCore
   *
   *  \brief This is the function to be called to process boundary conditions 
   *         from input
   *
   *  \param phiTag An Expr::Tag that has contains the tag of the expression
   we're setting the bc on
   *
   *  \param fieldName A string containing the name of the field for which we
   want to apply the bc. This is needed for expressions that compute
   multiple fields such as the pressure.
   *
   *  \param staggeredLocation This is the staggered location of the field on
   *         which the boundary condition is applied
   *
   *  \param graphHelper The graph which contains the expression associated with
   the field we are setting the bc on
   *
   *  \param localPatches The set of local patches on which the BCs are applied
   *
   *  \param patchInfoMap The Wasatch patchInfoMap
   *
   *  \param materials The material subset on which BCs are to be applied
   */
  //****************************************************************************    
  template < typename FieldT >
  void process_boundary_conditions( const Expr::Tag& phiTag,
                                    const std::string& fieldName,
                                    const Direction staggeredLocation,
                                    const GraphHelper& graphHelper,
                                    const Uintah::PatchSet* const localPatches,
                                    const PatchInfoMap& patchInfoMap,
                                    const Uintah::MaterialSubset* const materials,
                                    const std::map<std::string, std::set<std::string> >& bcFunctorMap,
                                    std::string useFieldForBCIterator,
                                    double useBCValue,
                                    std::string useBCKind,
                                    std::string useBCFunctorName)
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
    // loop over all patches, and for each patch set boundary conditions

    namespace SS = SpatialOps::structured;
    typedef SS::ConstValEval BCEvaluator; // basic functor for constant functions.
    const std::string phiName = phiTag.name();
    
    bool useOtherField = useFieldForBCIterator.empty() ? false : true;
    
    if (useFieldForBCIterator.empty()) {
      useFieldForBCIterator = fieldName;
    }
    
    // loop over local patches
    for( int ip=0; ip<localPatches->size(); ++ip ){

      // get the patch subset
      const Uintah::PatchSubset* const patches = localPatches->getSubset(ip);

      // loop over every patch in the patch subset
      for( int ipss=0; ipss<patches->size(); ++ipss ){

        // get a pointer to the current patch
        const Uintah::Patch* const patch = patches->get(ipss);

        // get the patch info from which we can get the operators database
        const PatchInfoMap::const_iterator ipi = patchInfoMap.find( patch->getID() );
        assert( ipi != patchInfoMap.end() );
        const SpatialOps::OperatorDatabase& opdb = *(ipi->second.operators);

        // loop over materials
        for( int im=0; im<materials->size(); ++im ){

          // process functors... this is an ugly part of the code but we
          // have to deal with this because Uintah doesn't allow a given task
          // to run on different patches with different requires. We also can't
          // get a bc graph to play nicely with the time advance graphs.
          // this block will essentially enforce the same dependencies across
          // all patches by adding functor expressions on them. those patches
          // that do NOT have that functor associated with any of their boundaries
          // will just expose the dependencies advertised by the functor but will
          // accomplish nothing else because that functor doesn't have any bc points
          // associated with it.
          
          Expr::ExpressionFactory& factory = *graphHelper.exprFactory;
          std::map< std::string, std::set<std::string> >::const_iterator iter = bcFunctorMap.begin();
          
          while ( iter != bcFunctorMap.end() ) {
            std::string functorPhiName = (*iter).first;
            if ( functorPhiName.compare(fieldName) == 0 ) {
              // get the functor set associated with this field
              std::set<std::string>::iterator functorIter = (*iter).second.begin();
              while (functorIter != (*iter).second.end() ) {
                std::string functorName = *functorIter;
                Expr::Tag modTag = Expr::Tag(functorName,Expr::STATE_NONE);
                factory.attach_modifier_expression( modTag, phiTag, patch->getID(), true );
                ++functorIter;
              }
            }
            ++iter;
          }

          const int materialID = materials->get(im);

          std::vector<Uintah::Patch::FaceType> bndFaces;
          patch->getBoundaryFaces(bndFaces);
          std::vector<Uintah::Patch::FaceType>::const_iterator faceIterator = bndFaces.begin();

          // loop over the boundary faces
          for( ; faceIterator!=bndFaces.end(); ++faceIterator ){
            Uintah::Patch::FaceType face = *faceIterator;

            //get the face direction
            SCIRun::IntVector insideCellDir = patch->faceDirection(face);
            //std::cout << "Inside Cell Dir: \n" << insideCellDir << std::endl;

            // get the number of children
            // jcs note that we need to do some error checking here.
            // If the BC has not been set then we get a cryptic error
            // from Uintah.
            const int numChildren = patch->getBCDataArray(face)->getNumberChildren(materialID);

            for( int child = 0; child<numChildren; ++child ){

              double bc_value = -9;
              std::string bc_kind = "NotSet";
              std::string bc_name = "none";
              std::string bc_functor_name = "none";
              SCIRun::Iterator bound_ptr;
              //
              // TSAAD NOTE TO SELF:
              // In the abscence of extra cells, the uintah bc iterator will return the indices of interior scalar cells adjacent to the boundary. These will be zero based.
              // In the presence of extra cells, the uintah bc iterator will return the indices of the scalar extra cells. These will start with [-1,-1,-1].
              //
              // ALSO NOTE: that even with staggered scalar Wasatch fields, there is NO additional ghost cell on the x+ face. So
              // nx_staggered = nx_scalar
              //
              bool foundIterator = get_iter_bcval_bckind_bcname( patch, face, child, useFieldForBCIterator, materialID, bc_value, bound_ptr, bc_kind, bc_name, bc_functor_name);
              if (useOtherField) {
                bc_value = useBCValue;
                bc_kind = useBCKind;
                bc_functor_name = useBCFunctorName;
              }
              SS::IntVec faceOffset(0,0,0);
              if (foundIterator) {
                process_bcs_on_face<FieldT> (bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, bc_name, bc_functor_name);
              }
            } // child loop
          } // face loop
        } // material loop
      } // patch subset loop
    } // local patch loop
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
//    // check if we have plus boundary faces on this patch
//    bool hasPlusFace[3] = {false,false,false};
//    if (patch->getBCType(Uintah::Patch::xplus)==Uintah::Patch::None) hasPlusFace[0]=true;
//    if (patch->getBCType(Uintah::Patch::yplus)==Uintah::Patch::None) hasPlusFace[1]=true;
//    if (patch->getBCType(Uintah::Patch::zplus)==Uintah::Patch::None) hasPlusFace[2]=true;
    // get the dimensions of this patch
    namespace SS = SpatialOps::structured;
    const SCIRun::IntVector patchDim_ = patch->getCellHighIndex();
    const SS::IntVec patchDim(patchDim_[0],patchDim_[1],patchDim_[2]);
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
        SCIRun::Iterator bound_ptr;
        const bool foundIterator = get_iter_bcval_bckind_bcname( patch, face, child, phiName, material, bc_value, bound_ptr, bc_kind,bc_name,bc_functor_name);

        if (foundIterator) {

          SCIRun::IntVector insideCellDir = patch->faceDirection(face);
          const bool hasExtraCells = ( patch->getExtraCells() != SCIRun::IntVector(0,0,0) );

          SS::IntVec bcPointGhostOffset(0,0,0);
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

          if (bc_kind=="Dirichlet") {
            for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
              SCIRun::IntVector bc_point_indices(*bound_ptr);
              
              bc_point_indices = bc_point_indices - patchCellOffset;
              
              const SS::IntVec   intCellIJK( bc_point_indices[0],
                                            bc_point_indices[1],
                                            bc_point_indices[2] );
              const SS::IntVec ghostCellIJK( bc_point_indices[0]+bcPointGhostOffset[0],
                                            bc_point_indices[1]+bcPointGhostOffset[1],
                                            bc_point_indices[2]+bcPointGhostOffset[2] );
              
              const int iInterior = poissonField.window_without_ghost().flat_index( hasExtraCells? ghostCellIJK : intCellIJK  );
//            const int iGhost    = poissonField.window_without_ghost().flat_index( hasExtraCells? intCellIJK   : ghostCellIJK);
//            const double ghostValue = 2.0*bc_value - poissonField[iInterior];
//            poissonRHS[iInterior] += bc_value/denom;
              poissonRHS[iInterior] += 2.0*bc_value/denom;
            }
          } else if (bc_kind=="Neumann") {
            for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
              SCIRun::IntVector bc_point_indices(*bound_ptr);
              
              bc_point_indices = bc_point_indices - patchCellOffset;
                            
              const SS::IntVec   intCellIJK( bc_point_indices[0],
                                            bc_point_indices[1],
                                            bc_point_indices[2] );
              const SS::IntVec ghostCellIJK( bc_point_indices[0]+bcPointGhostOffset[0],
                                            bc_point_indices[1]+bcPointGhostOffset[1],
                                            bc_point_indices[2]+bcPointGhostOffset[2] );
              
              const int iInterior = poissonField.window_without_ghost().flat_index( hasExtraCells? ghostCellIJK : intCellIJK  );
//            const int iGhost    = poissonField.window_without_ghost().flat_index( hasExtraCells? intCellIJK   : ghostCellIJK);
//            const double ghostValue = spacing*bc_value + poissonField[iInterior];
//            poissonRHS[iInterior] += ghostValue/denom;
              poissonRHS[iInterior] += spacing*bc_value/denom;
            }
          } else {
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
    namespace SS = SpatialOps::structured;
    const SCIRun::IntVector patchDim_ = patch->getCellHighIndex();
    const SS::IntVec patchDim(patchDim_[0],patchDim_[1],patchDim_[2]);
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
        SCIRun::Iterator bound_ptr;

        //patch->getBCDataArray(face)->getCellFaceIterator(material, bound_ptr, child);
        double bc_value;
        std::string bc_kind;
        std::string bc_name = "none";
        std::string bc_functor_name = "none";        
        get_iter_bcval_bckind_bcname( patch, face, child, poissonTag.name(), material, bc_value, bound_ptr, bc_kind, bc_name,bc_functor_name);
        
        SCIRun::IntVector insideCellDir = patch->faceDirection(face);
        const bool hasExtraCells = ( patch->getExtraCells() != SCIRun::IntVector(0,0,0) );

        // cell offset used to calculate local cell index with respect to patch.
        const SCIRun::IntVector patchCellOffset = patch->getCellLowIndex(0);
        
        if (bc_kind == "Dirichlet") { // pressure Outlet BC. don't forget to update pressure_rhs also.
          for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
            SCIRun::IntVector bc_point_indices(*bound_ptr);
            Uintah::Stencil4& coefs = poissonMatrix[hasExtraCells ? bc_point_indices - insideCellDir : bc_point_indices];
            
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
        } else if (bc_kind == "Neumann") { // outflow bc
          for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
            SCIRun::IntVector bc_point_indices(*bound_ptr);
            
            Uintah::Stencil4& coefs = poissonMatrix[hasExtraCells ? bc_point_indices - insideCellDir : bc_point_indices];
            
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
        } else { // when no pressure BC is specified, it implies that we have wall. 
                 // note that when the face is periodic, then bound_ptr is empty
          for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
            SCIRun::IntVector bc_point_indices(*bound_ptr);
            
            Uintah::Stencil4& coefs = poissonMatrix[hasExtraCells ? bc_point_indices - insideCellDir : bc_point_indices];
            
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
      const bool containsAllNeighbors = patch->containsCell(refCell + IntVector(1,0,0)) &&
                                        patch->containsCell(refCell + IntVector(0,1,0)) &&
                                        patch->containsCell(refCell + IntVector(0,0,1));
      // check if all cell neighbors are contained in this patch:
      if ( !containsAllNeighbors ) {
        msg << std::endl
        << "  Invalid reference cell specified for poisson system." << std::endl
        << "  The reference cell, as well as its north, east, and top neighbors must be in the same patch." << std::endl
        << std::endl;
        throw std::runtime_error( msg.str() );
      }
      Uintah::Stencil4& refCoef = poissonMatrix[refCell];
      refCoef.w = 0.0;
      refCoef.s = 0.0;
      refCoef.b = 0.0;
      refCoef.p = 1.0;
      poissonMatrix[refCell + IntVector(1,0,0)].w = 0.0;
      poissonMatrix[refCell + IntVector(0,1,0)].s = 0.0;
      poissonMatrix[refCell + IntVector(0,0,1)].b = 0.0;
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
      const bool containsAllNeighbors = patch->containsCell(refCell + IntVector(1,0,0)) &&
                                        patch->containsCell(refCell + IntVector(0,1,0)) &&
                                        patch->containsCell(refCell + IntVector(0,0,1)) &&
                                       (patch->containsCell(refCell + IntVector(-1,0,0)) || refCell.x() == 0) &&
                                       (patch->containsCell(refCell + IntVector(0,-1,0)) || refCell.y() == 0) &&
                                       (patch->containsCell(refCell + IntVector(0,0,-1)) || refCell.z() == 0) ;
      // check if all cell neighbors are contained in this patch:
      if ( !containsAllNeighbors ) {
        msg << std::endl
        << "  Invalid reference poisson cell." << std::endl
        << "  The reference poisson cell as well as its north, east, and top neighbors must be contained in the same patch." << std::endl
        << std::endl;
        throw std::runtime_error( msg.str() );
      }
      // remember, indexing is local for SpatialOps so we must offset by the patch's low index
      const SCIRun::IntVector refCellWithOffset = refCell - patch->getCellLowIndex(0);
      const SpatialOps::structured::IntVec refCellIJK(refCellWithOffset.x(),refCellWithOffset.y(),refCellWithOffset.z());
      const int irefCell = poissonRHS.window_without_ghost().flat_index(refCellIJK);
      poissonRHS[irefCell] = refpoissonValue;

      // modify rhs for neighboring cells
      const Uintah::Vector spacing = patch->dCell();
      const double dx2 = spacing[0]*spacing[0];
      const double dy2 = spacing[1]*spacing[1];
      const double dz2 = spacing[2]*spacing[2];
      poissonRHS[poissonRHS.window_without_ghost().flat_index(refCellIJK + SpatialOps::structured::IntVec(1,0,0) ) ] += refpoissonValue/dx2;
      poissonRHS[poissonRHS.window_without_ghost().flat_index(refCellIJK + SpatialOps::structured::IntVec(0,1,0) ) ] += refpoissonValue/dy2;
      poissonRHS[poissonRHS.window_without_ghost().flat_index(refCellIJK + SpatialOps::structured::IntVec(0,0,1) ) ] += refpoissonValue/dz2;
      poissonRHS[poissonRHS.window_without_ghost().flat_index(refCellIJK + SpatialOps::structured::IntVec(-1,0,0) )] += refpoissonValue/dx2;
      poissonRHS[poissonRHS.window_without_ghost().flat_index(refCellIJK + SpatialOps::structured::IntVec(0,-1,0) )] += refpoissonValue/dy2;
      poissonRHS[poissonRHS.window_without_ghost().flat_index(refCellIJK + SpatialOps::structured::IntVec(0,0,-1) )] += refpoissonValue/dz2;
    }
  }

  //****************************************************************************  
  /**
   *  \ingroup WasatchCore
   *
   *  \brief Key function to prcess boundary conditions for poisson equation.
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
    namespace SS = SpatialOps::structured;
    const SCIRun::IntVector patchDim_ = patch->getCellHighIndex();
    const SS::IntVec patchDim(patchDim_[0],patchDim_[1],patchDim_[2]);
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
        std::string bc_kind = "NotSet";
        std::string bc_name = "none";
        std::string bc_functor_name = "none";
        SCIRun::Iterator bound_ptr;
        const bool foundIterator = get_iter_bcval_bckind_bcname( patch, face, child, phiName, material, bc_value, bound_ptr, bc_kind, bc_name,bc_functor_name);
        
        if (foundIterator) {
          
          SCIRun::IntVector insideCellDir = patch->faceDirection(face);
          const bool hasExtraCells = ( patch->getExtraCells() != SCIRun::IntVector(0,0,0) );
          
          SS::IntVec bcPointGhostOffset(0,0,0);
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
          
          if (bc_kind=="Dirichlet") {
            for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
              SCIRun::IntVector bc_point_indices(*bound_ptr);

              bc_point_indices = bc_point_indices - patchCellOffset;
              
              const SS::IntVec   intCellIJK( bc_point_indices[0],
                                            bc_point_indices[1],
                                            bc_point_indices[2] );
              const SS::IntVec ghostCellIJK( bc_point_indices[0]+bcPointGhostOffset[0],
                                            bc_point_indices[1]+bcPointGhostOffset[1],
                                            bc_point_indices[2]+bcPointGhostOffset[2] );
              
              const int iInterior = poissonField.window_without_ghost().flat_index( hasExtraCells? ghostCellIJK : intCellIJK  );
              const int iGhost    = poissonField.window_without_ghost().flat_index( hasExtraCells? intCellIJK   : ghostCellIJK);              
              poissonField[iGhost] = 2.0*bc_value - poissonField[iInterior];              
            }
          } else if (bc_kind=="Neumann") {
            for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
              SCIRun::IntVector bc_point_indices(*bound_ptr);
              
              bc_point_indices = bc_point_indices - patchCellOffset;
              
              const SS::IntVec   intCellIJK( bc_point_indices[0],
                                            bc_point_indices[1],
                                            bc_point_indices[2] );
              const SS::IntVec ghostCellIJK( bc_point_indices[0]+bcPointGhostOffset[0],
                                            bc_point_indices[1]+bcPointGhostOffset[1],
                                            bc_point_indices[2]+bcPointGhostOffset[2] );
              
              const int iInterior = poissonField.window_without_ghost().flat_index( hasExtraCells? ghostCellIJK : intCellIJK  );
              const int iGhost    = poissonField.window_without_ghost().flat_index( hasExtraCells? intCellIJK   : ghostCellIJK);              
              poissonField[iGhost] = spacing*bc_value + poissonField[iInterior];
            }
          } else {
            return;
          }
        }
      } // child loop
    }    
  }
  
  //==================================================================
  // Explicit template instantiation
  #include <CCA/Components/Wasatch/FieldTypes.h>
  using namespace SpatialOps::structured;

  #define INSTANTIATE_PROCESS_BCS_ON_FACE( FIELDT )                                   \
  template void process_bcs_on_face<FIELDT>( SCIRun::Iterator& bound_ptr,             \
                                             const Uintah::Patch::FaceType& face,     \
                                             const Direction staggeredLocation,       \
                                             const Uintah::Patch* const patch,        \
                                             const GraphHelper& graphHelper,          \
                                             const Expr::Tag& phiTag,                 \
                                             const std::string& fieldName,            \
                                             const double bc_value,                   \
                                             const SpatialOps::OperatorDatabase& opdb,\
                                             const std::string& bc_kind,              \
                                             const std::string& bc_name,              \
                                             const std::string& bc_functor_name);

  INSTANTIATE_PROCESS_BCS_ON_FACE(SVolField);
  INSTANTIATE_PROCESS_BCS_ON_FACE(XVolField);
  INSTANTIATE_PROCESS_BCS_ON_FACE(FaceTypes<XVolField>::XFace);
  INSTANTIATE_PROCESS_BCS_ON_FACE(YVolField);
  INSTANTIATE_PROCESS_BCS_ON_FACE(FaceTypes<YVolField>::YFace);
  INSTANTIATE_PROCESS_BCS_ON_FACE(ZVolField);
  INSTANTIATE_PROCESS_BCS_ON_FACE(FaceTypes<ZVolField>::ZFace);


  #define INSTANTIATE_PROCESS_BOUNDARY_CONDITIONS( FIELDT )                                           \
  template void process_boundary_conditions< FIELDT >( const Expr::Tag& phiTag,                       \
                                                       const std::string& fieldName,                  \
                                                       const Direction staggeredLocation,             \
                                                       const GraphHelper& graphHelper,                \
                                                       const Uintah::PatchSet* const localPatches,    \
                                                       const PatchInfoMap& patchInfoMap,              \
                                                       const Uintah::MaterialSubset* const materials, \
                                                       const std::map<std::string, std::set<std::string> >& bcFunctorMap, \
                                                       std::string useFieldForBCIterator,              \
                                                       double useBCValue,                              \
                                                       std::string useBCKind,                          \
                                                       std::string useBCFunctorName);                 


  INSTANTIATE_PROCESS_BOUNDARY_CONDITIONS(SVolField);
  INSTANTIATE_PROCESS_BOUNDARY_CONDITIONS(XVolField);
  INSTANTIATE_PROCESS_BOUNDARY_CONDITIONS(FaceTypes<XVolField>::XFace);
  INSTANTIATE_PROCESS_BOUNDARY_CONDITIONS(YVolField);
  INSTANTIATE_PROCESS_BOUNDARY_CONDITIONS(FaceTypes<YVolField>::YFace);
  INSTANTIATE_PROCESS_BOUNDARY_CONDITIONS(ZVolField);
  INSTANTIATE_PROCESS_BOUNDARY_CONDITIONS(FaceTypes<ZVolField>::ZFace);

  //==================================================================

} // namespace wasatch

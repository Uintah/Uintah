/*
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

//-- Wasatch includes --//
#include "Operators/OperatorTypes.h"
#include "FieldTypes.h"
#include "BCHelperTools.h"


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

  //-----------------------------------------------------------------------------

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
  /**
   *  \ingroup WasatchCore
   *
   *  \brief This function determines the point(s) on which we want to set bcs.
   *
   */
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
    const SS::IntVec interiorCellIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]);
    const SS::IntVec bndFaceIJK = interiorCellIJK + faceOffset;
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
            bcPointIJK = interiorStgrdCellIJK;
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
            bcPointIJK = (bc_kind.compare("Dirichlet")==0 ? interiorCellIJK : interiorStgrdCellIJK);
            ghostPointIJK = (bc_kind.compare("Dirichlet")==0 ? stgrdBndFaceIJK : interiorCellIJK);
          } else {
            // this stuff works with boundary layer cells
            bcPointIJK = (bc_kind.compare("Dirichlet")==0 ? stgrdBndFaceIJK : interiorCellIJK);
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

  //-----------------------------------------------------------------------------

  /**
   *  \ingroup WasatchCore
   *
   *  \brief This function sets the boundary condition on a point. this gets
             called from set_bc_on_face.
   *
   */
  template < typename FieldT, typename BCOpT >
  void set_bc_on_point( const Uintah::Patch* const patch,
                        const GraphHelper& gh,
                        const Expr::Tag phiTag,
                        const std::string fieldName,
                        const SpatialOps::structured::IntVec& bcPointIndex,
                        const SpatialOps::structured::IntVec& ghostPointIJK,
                        const SpatialOps::structured::BCSide bcSide,
                        const double bcValue,
                        const SpatialOps::OperatorDatabase& opdb,
                        const bool isStaggered,
                        const std::string& bc_kind )
  {
    typedef SpatialOps::structured::ConstValEval BCVal;
    typedef SpatialOps::structured::BoundaryCondition<FieldT,BCVal> BC;
    typedef typename BCOpT::BCEvalT BCEvaluator;
    Expr::ExpressionFactory& factory = *gh.exprFactory;
    const Expr::ExpressionID phiID = factory.get_id(phiTag);
    Expr::Expression<FieldT>& phiExpr = dynamic_cast<Expr::Expression<FieldT>&>( factory.retrieve_expression( phiID, patch->getID(), true ) );

    //FieldT& phiField = phiExpr.value();
    //const int iGhost = phiField.window_without_ghost().flat_index( ghostPointIJK );
    //const double ghostBCValue = 2*bcValue - interiorValue

    if (isStaggered) {
      if (bc_kind.compare("Dirichlet")==0) {
        BC bound_cond(bcPointIndex, BCVal(bcValue));
        phiExpr.process_after_evaluate( fieldName, bound_cond );
        BC bound_cond_ghost(ghostPointIJK, BCVal(bcValue));
        phiExpr.process_after_evaluate( fieldName, bound_cond_ghost );
      } else {
        BCOpT bcOp( bcPointIndex, bcSide, BCEvaluator(bcValue), opdb );
        phiExpr.process_after_evaluate(fieldName, bcOp );
        BCOpT bcOp_ghost( ghostPointIJK, bcSide, BCEvaluator(bcValue), opdb );
        phiExpr.process_after_evaluate(fieldName, bcOp_ghost );
      }
    } else {
      BCOpT bcOp( bcPointIndex, bcSide, BCEvaluator(bcValue), opdb );
      phiExpr.process_after_evaluate(fieldName, bcOp );
    }
  }

  //-----------------------------------------------------------------------------

  /**
   *  @struct BCOpTypeSelector
   *
   *  @brief This templated struct is used to simplify boundary
   *         condition operator selection.
   */
  template< typename FieldT, typename BCEvalT>
  struct BCOpTypeSelector
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

  //-----------------------------------------------------------------------------

  /**
   *  \ingroup WasatchCore
   *
   *  \brief This function grabs an iterator and a value associated with a
             boundary condition set in the input file.
   *
   */
  template <typename T>
  bool get_iter_bcval_bckind( const Uintah::Patch* patch,
                             const Uintah::Patch::FaceType face,
                             const int child,
                             const std::string& desc,
                             const int mat_id,
                             T& bc_value,
                             SCIRun::Iterator& bound_ptr,
                             std::string& bc_kind )
  {
    SCIRun::Iterator nu;
    const Uintah::BoundCondBase* const bc = patch->getArrayBCValues( face, mat_id, desc, bound_ptr, nu, child );
    const Uintah::BoundCond<T>* const new_bcs = dynamic_cast<const Uintah::BoundCond<T>*>(bc);

    bc_value=T(-9);
    bc_kind="NotSet";
    if (new_bcs != 0) {      // non-symmetric
      bc_value = new_bcs->getValue();
      bc_kind =  new_bcs->getBCType__NEW();
    }

    delete bc;

    // Did I find an iterator
    return( bc_kind.compare("NotSet") != 0 );
  }

  //---------------------------------------------------------------------------

  /**
   *  \ingroup WasatchCore
   *
   *  \brief This function sets the boundary conditions on every point of a
             given face.
   *
   */
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
                           const SpatialOps::structured::BCSide bcSide,
                           const SpatialOps::structured::IntVec& faceOffset,
                           const bool hasExtraCells)
  {
    namespace SS = SpatialOps::structured;
    typedef SS::ConstValEval BCEvalT; // basic functor for constant functions.
    SCIRun::IntVector patchCellOffset = patch->getCellLowIndex(0); // cell offset used to calculate local cell index with respect to patch.
    SCIRun::IntVector insideCellDir = patch->faceDirection(face);
    proc0cout << "SETTING BOUNDARY CONDITION ON "<< fieldName << " FACE:" << face << std::endl;
    for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
      SCIRun::IntVector bc_point_indices(*bound_ptr);
      //std::cout << "bc point indices " << bc_point_indices << std::endl << std::endl;
      bc_point_indices = bc_point_indices - patchCellOffset; // get local cell index (with respect to patch). SpatialOps needs local cell indices.
      SS::IntVec bcPointIJK;
      SS::IntVec ghostPointIJK;
      get_bc_points_ijk ( staggeredLocation, face, bcSide, bc_kind, bc_point_indices, faceOffset, insideCellDir, bcPointIJK,ghostPointIJK, hasExtraCells);
      set_bc_on_point< FieldT, BcT >( patch, graphHelper, phiTag,fieldName,
                                     bcPointIJK, ghostPointIJK, bcSide, bc_value,
                                     opdb, is_staggered_bc(staggeredLocation, face),
                                     bc_kind);
    }
  }

  //-----------------------------------------------------------------------------
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
                            const std::string& bc_kind )
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
          set_bcs_on_face<FieldT,typename BCOpT::DirichletX>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::xplus:
          set_bcs_on_face<FieldT,typename BCOpT::DirichletX>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yminus:
          set_bcs_on_face<FieldT,typename BCOpT::DirichletY>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yplus:
          set_bcs_on_face<FieldT,typename BCOpT::DirichletY>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::zminus:
          set_bcs_on_face<FieldT,typename BCOpT::DirichletZ>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::zplus:
          set_bcs_on_face<FieldT,typename BCOpT::DirichletZ>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::numFaces:
          throw Uintah::ProblemSetupException( "An invalid face Patch::numFaces was encountered while setting boundary conditions", __FILE__, __LINE__ );
          break;
        case Uintah::Patch::invalidFace:
          throw Uintah::ProblemSetupException( "An invalid face Patch::invalidFace was encountered while setting boundary conditions", __FILE__, __LINE__ );
          break;
        default:
          break;
      }

    } else if (bc_kind.compare("Neumann")==0 ){
      switch( face ){
        case Uintah::Patch::xminus:
          set_bcs_on_face<FieldT,typename BCOpT::NeumannX>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::xplus:
          set_bcs_on_face<FieldT,typename BCOpT::NeumannX>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yminus:
          set_bcs_on_face<FieldT,typename BCOpT::NeumannY>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yplus:
          set_bcs_on_face<FieldT,typename BCOpT::NeumannY>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::zminus:
          set_bcs_on_face<FieldT,typename BCOpT::NeumannZ>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::zplus:
          set_bcs_on_face<FieldT,typename BCOpT::NeumannZ>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::numFaces:
          throw Uintah::ProblemSetupException( "An invalid face Patch::numFaces was encountered while setting boundary conditions", __FILE__, __LINE__ );
          break;
        case Uintah::Patch::invalidFace:
          throw Uintah::ProblemSetupException( "An invalid face Patch::invalidFace was encountered while setting boundary conditions", __FILE__, __LINE__ );
          break;
        default:
          break;
      }
    }

  }

  //-----------------------------------------------------------------------------
  // Specialization for normal stress and convective flux for xvol fields
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
                                                         const std::string& bc_kind )
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
          set_bcs_on_face<FieldT, BCOpT::DirichletX>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::xplus:
          set_bcs_on_face<FieldT, BCOpT::DirichletX>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yminus:  case Uintah::Patch::yplus:
        case Uintah::Patch::zminus:  case Uintah::Patch::zplus:
          throw Uintah::ProblemSetupException( "Invalid face", __FILE__, __LINE__ );
          break;
        case Uintah::Patch::numFaces:
          throw Uintah::ProblemSetupException( "An invalid face Patch::numFaces was encountered while setting boundary conditions", __FILE__, __LINE__ );
          break;
        case Uintah::Patch::invalidFace:
          throw Uintah::ProblemSetupException( "An invalid face Patch::invalidFace was encountered while setting boundary conditions", __FILE__, __LINE__ );
          break;
      }

    } else if (bc_kind.compare("Neumann")==0 ){
      switch( face ){
        case Uintah::Patch::xminus:
          set_bcs_on_face<FieldT, BCOpT::NeumannX>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::xplus:
          set_bcs_on_face<FieldT, BCOpT::NeumannX>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yminus:  case Uintah::Patch::yplus:
        case Uintah::Patch::zminus:  case Uintah::Patch::zplus:
          throw Uintah::ProblemSetupException( "Invalid face", __FILE__, __LINE__ );
          break;
        case Uintah::Patch::numFaces:
          throw Uintah::ProblemSetupException( "An invalid face Patch::numFaces was encountered while setting boundary conditions", __FILE__, __LINE__ );
          break;
        case Uintah::Patch::invalidFace:
          throw Uintah::ProblemSetupException( "An invalid face Patch::invalidFace was encountered while setting boundary conditions", __FILE__, __LINE__ );
          break;
      }
    }

  }

  //-----------------------------------------------------------------------------
  // Specialization for normal stress and convective flux for yvol fields
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
                                                         const std::string& bc_kind )
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
          set_bcs_on_face<FieldT, BCOpT::DirichletY>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yplus:
          set_bcs_on_face<FieldT, BCOpT::DirichletY>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::xminus:  case Uintah::Patch::xplus:
        case Uintah::Patch::zminus:  case Uintah::Patch::zplus:
          throw Uintah::ProblemSetupException( "Invalid face", __FILE__, __LINE__ );
          break;
        case Uintah::Patch::numFaces:
          throw Uintah::ProblemSetupException( "An invalid face Patch::numFaces was encountered while setting boundary conditions", __FILE__, __LINE__ );
          break;
        case Uintah::Patch::invalidFace:
          throw Uintah::ProblemSetupException( "An invalid face Patch::invalidFace was encountered while setting boundary conditions", __FILE__, __LINE__ );
          break;
      }

    } else if (bc_kind.compare("Neumann")==0 ){
      switch( face ){
        case Uintah::Patch::yminus:
          set_bcs_on_face<FieldT, BCOpT::NeumannY>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yplus:
          set_bcs_on_face<FieldT, BCOpT::NeumannY>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::xminus:  case Uintah::Patch::xplus:
        case Uintah::Patch::zminus:  case Uintah::Patch::zplus:
          throw Uintah::ProblemSetupException( "Invalid face", __FILE__, __LINE__ );
          break;
       case Uintah::Patch::numFaces:
          throw Uintah::ProblemSetupException( "An invalid face Patch::numFaces was encountered while setting boundary conditions", __FILE__, __LINE__ );
          break;
        case Uintah::Patch::invalidFace:
          throw Uintah::ProblemSetupException( "An invalid face Patch::invalidFace was encountered while setting boundary conditions", __FILE__, __LINE__ );
          break;
      }
    }
  }

  //-----------------------------------------------------------------------------
  // Specialization for normal stress and convective flux for zvol fields
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
                                                         const std::string& bc_kind )
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
          set_bcs_on_face<FieldT, BCOpT::DirichletZ>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::zplus:
          set_bcs_on_face<FieldT, BCOpT::DirichletZ>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yminus:  case Uintah::Patch::yplus:
        case Uintah::Patch::xminus:  case Uintah::Patch::xplus:
          throw Uintah::ProblemSetupException( "Invalid face", __FILE__, __LINE__ );
          break;
        case Uintah::Patch::numFaces:
          throw Uintah::ProblemSetupException( "An invalid face Patch::numFaces was encountered while setting boundary conditions", __FILE__, __LINE__ );
          break;
        case Uintah::Patch::invalidFace:
          throw Uintah::ProblemSetupException( "An invalid face Patch::invalidFace was encountered while setting boundary conditions", __FILE__, __LINE__ );
          break;
      }

    } else if (bc_kind.compare("Neumann")==0 ){
      switch( face ){
        case Uintah::Patch::zminus:
          set_bcs_on_face<FieldT, BCOpT::NeumannZ>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::MINUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::zplus:
          set_bcs_on_face<FieldT, BCOpT::NeumannZ>(bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind, SpatialOps::structured::PLUS_SIDE,faceOffset, hasExtraCells);
          break;
        case Uintah::Patch::yminus:  case Uintah::Patch::yplus:
        case Uintah::Patch::xminus:  case Uintah::Patch::xplus:
          throw Uintah::ProblemSetupException( "Invalid face", __FILE__, __LINE__ );
          break;
        case Uintah::Patch::numFaces:
          throw Uintah::ProblemSetupException( "An invalid face Patch::numFaces was encountered while setting boundary conditions", __FILE__, __LINE__ );
          break;
        case Uintah::Patch::invalidFace:
          throw Uintah::ProblemSetupException( "An invalid face Patch::invalidFace was encountered while setting boundary conditions", __FILE__, __LINE__ );
          break;
      }
    }
  }

  //-----------------------------------------------------------------------------

  template < typename FieldT >
  void process_boundary_conditions( const Expr::Tag& phiTag,
                                    const std::string& fieldName,
                                    const Direction staggeredLocation,
                                    const GraphHelper& graphHelper,
                                    const Uintah::PatchSet* const localPatches,
                                    const PatchInfoMap& patchInfoMap,
                                    const Uintah::MaterialSubset* const materials )
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

            //get the number of children
            // jcs note that we need to do some error checking here.
            // If the BC has not been set then we get a cryptic error
            // from Uintah.
            const int numChildren = patch->getBCDataArray(face)->getNumberChildren(materialID);

            for( int child = 0; child<numChildren; ++child ){

              double bc_value = -9;
              std::string bc_kind = "NotSet";
              SCIRun::Iterator bound_ptr;
              //
              // TSAAD NOTE TO SELF:
              // In the abscence of extra cells, the uintah bc iterator will return the indices of interior scalar cells adjacent to the boundary. These will be zero based.
              // In the presence of extra cells, the uintah bc iterator will return the indices of the scalar extra cells. These will start with [-1,-1,-1].
              //
              // ALSO NOTE: that even with staggered scalar Wasatch fields, there is NO additional ghost cell on the x+ face. So
              // nx_staggered = nx_scalar
              //
              bool foundIterator = get_iter_bcval_bckind( patch, face, child, fieldName, materialID, bc_value, bound_ptr, bc_kind);
              SS::IntVec faceOffset(0,0,0);
              if (foundIterator) {
                process_bcs_on_face<FieldT> (bound_ptr,face,staggeredLocation,patch,graphHelper,phiTag,fieldName,bc_value,opdb,bc_kind);
              }
            } // child loop
          } // face loop
        } // material loop
      } // patch subset loop
    } // local patch loop
  }


  //-----------------------------------------------------------------------------

  void update_pressure_rhs( const Expr::Tag& pressureTag,
                            Uintah::CCVariable<Uintah::Stencil4>& pressureMatrix,
                            SVolField& pressureField,
                            SVolField& pressureRHS,
                            const Uintah::Patch* patch)
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
    // check if we have plus boundary faces on this patch
    bool hasPlusFace[3] = {false,false,false};
    if (patch->getBCType(Uintah::Patch::xplus)==Uintah::Patch::None) hasPlusFace[0]=true;
    if (patch->getBCType(Uintah::Patch::yplus)==Uintah::Patch::None) hasPlusFace[1]=true;
    if (patch->getBCType(Uintah::Patch::zplus)==Uintah::Patch::None) hasPlusFace[2]=true;
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
    const int materialID = 0;

    const std::string phiName = pressureTag.name();

    std::vector<Uintah::Patch::FaceType> bndFaces;
    patch->getBoundaryFaces(bndFaces);
    std::vector<Uintah::Patch::FaceType>::const_iterator faceIterator = bndFaces.begin();

    // loop over the boundary faces
    for( ; faceIterator!=bndFaces.end(); ++faceIterator ){
      Uintah::Patch::FaceType face = *faceIterator;

      //get the number of children
      const int numChildren = patch->getBCDataArray(face)->getNumberChildren(materialID);

      for( int child = 0; child<numChildren; ++child ){

        double bc_value = -9;
        std::string bc_kind = "NotSet";
        SCIRun::Iterator bound_ptr;
        const bool foundIterator = get_iter_bcval_bckind( patch, face, child, phiName, materialID, bc_value, bound_ptr, bc_kind);

        if (foundIterator) {

            SS::IntVec bcPointGhostOffset(0,0,0);
            double denom = 0.0;
            switch( face ){
              case Uintah::Patch::xminus:  bcPointGhostOffset[0] = -1;  denom = dx2;  break;
              case Uintah::Patch::xplus :  bcPointGhostOffset[0] =  1;  denom = dx2;  break;
              case Uintah::Patch::yminus:  bcPointGhostOffset[1] = -1;  denom = dy2;  break;
              case Uintah::Patch::yplus :  bcPointGhostOffset[1] =  1;  denom = dy2;  break;
              case Uintah::Patch::zminus:  bcPointGhostOffset[2] = -1;  denom = dz2;  break;
              case Uintah::Patch::zplus :  bcPointGhostOffset[2] =  1;  denom = dz2;  break;
              case Uintah::Patch::numFaces:
                throw Uintah::ProblemSetupException( "An invalid face of type Patch::numFaces was encountered while setting boundary conditions", __FILE__, __LINE__ );
                break;
              case Uintah::Patch::invalidFace:
                throw Uintah::ProblemSetupException( "An invalid face of type Patch::invalidFace was encountered while setting boundary conditions", __FILE__, __LINE__ );
                break;
            } // switch

            // cell offset used to calculate local cell index with respect to patch.
            const SCIRun::IntVector patchCellOffset = patch->getCellLowIndex(0);

            for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
              SCIRun::IntVector bc_point_indices(*bound_ptr);
              Uintah::Stencil4& coefs = pressureMatrix[bc_point_indices];

              switch(face){
              case Uintah::Patch::xminus: coefs.w = 0.0; break;
              //case Uintah::Patch::xplus : coefs.e = 0.0; break;
              case Uintah::Patch::yminus: coefs.s = 0.0; break;
              //case Uintah::Patch::yplus : coefs.n = 0.0; break;
              case Uintah::Patch::zminus: coefs.b = 0.0; break;
              //case Uintah::Patch::zplus : coefs.t = 0.0; break;
              }

              bc_point_indices = bc_point_indices - patchCellOffset;

              const SS::IntVec   intCellIJK( bc_point_indices[0],
                                             bc_point_indices[1],
                                             bc_point_indices[2] );
              const SS::IntVec ghostCellIJK( bc_point_indices[0]+bcPointGhostOffset[0],
                                             bc_point_indices[1]+bcPointGhostOffset[1],
                                             bc_point_indices[2]+bcPointGhostOffset[2] );

              const int iInterior = pressureField.window_without_ghost().flat_index(   intCellIJK );
              const int iGhost    = pressureField.window_without_ghost().flat_index( ghostCellIJK );

              pressureRHS[iInterior] -= pressureField[iGhost]/denom;
            }
        }
      } // child loop
    } // face loop
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
                                             const std::string& bc_kind);

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
                                                       const Uintah::MaterialSubset* const materials);

  INSTANTIATE_PROCESS_BOUNDARY_CONDITIONS(SVolField);
  INSTANTIATE_PROCESS_BOUNDARY_CONDITIONS(XVolField);
  INSTANTIATE_PROCESS_BOUNDARY_CONDITIONS(FaceTypes<XVolField>::XFace);
  INSTANTIATE_PROCESS_BOUNDARY_CONDITIONS(YVolField);
  INSTANTIATE_PROCESS_BOUNDARY_CONDITIONS(FaceTypes<YVolField>::YFace);
  INSTANTIATE_PROCESS_BOUNDARY_CONDITIONS(ZVolField);
  INSTANTIATE_PROCESS_BOUNDARY_CONDITIONS(FaceTypes<ZVolField>::ZFace);

  //==================================================================

} // namespace wasatch

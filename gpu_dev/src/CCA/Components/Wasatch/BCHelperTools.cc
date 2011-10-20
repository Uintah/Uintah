//-- Uintah framework includes --//
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/Variables/Stencil7.h>
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
  
  //
  // this macro will be unwrapped inside the process_boundary_conditions method.  The
  // variable names correspond to those defined within the appropriate
  // scope of that method.
  //
  
  #define SET_BC( BCEvalT,  /* type of bc evaluator                    */                                                            \
                  BCT,      /* type of BC                              */                                                            \
                  SIDE      /* side of the cell on which the BC is set */ )                                                          \
    proc0cout << "SETTING BOUNDARY CONDITION ON "<< phiName << std::endl;                                                            \
    for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {                                                                       \
      const SCIRun::IntVector bc_point_indices(*bound_ptr);                                                                          \
      const SS::IntVec interiorCellIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]);                                 \
      const SS::IntVec bndFaceIJK = interiorCellIJK + faceOffset;                                                                    \
      const SS::IntVec stgrdBndFaceIJK( bc_point_indices[0] + insideCellDir[0],                                                      \
                                        bc_point_indices[1] + insideCellDir[1],                                                      \
                                        bc_point_indices[2] + insideCellDir[2]);                                                     \
      switch (staggeredLocation) {                                                                                                   \
      case XDIR:                                                                                                                     \
        typedef SS::XVolField  XFieldT;                                                                                              \
        if( bc_kind.compare("Dirichlet")==0 && (face==Uintah::Patch::xminus || face==Uintah::Patch::xplus) )                                     \
          set_bc_staggered<XFieldT> (patch,graphHelper,phiTag,bndFaceIJK,bc_value);                                                  \
        else                                                                                                                         \
          set_bc< XFieldT, BCOpTypeSelector<XFieldT,BCEvalT>::BCT >( patch, graphHelper, phiTag, bndFaceIJK, SIDE, bc_value, opdb ); \
        break;                                                                                                                       \
      case YDIR:                                                                                                                     \
        typedef SS::YVolField  YFieldT;                                                                                              \
        if (bc_kind.compare("Dirichlet")==0 && (face==Uintah::Patch::yminus || face==Uintah::Patch::yplus))                                      \
          set_bc_staggered<YFieldT> (patch,graphHelper,phiTag,bndFaceIJK,bc_value);                                                  \
        else                                                                                                                         \
          set_bc< YFieldT, BCOpTypeSelector<YFieldT,BCEvalT>::BCT >( patch, graphHelper, phiTag, bndFaceIJK, SIDE, bc_value, opdb ); \
        break;                                                                                                                       \
      case ZDIR:                                                                                                                     \
        typedef SS::ZVolField  ZFieldT;                                                                                              \
        if (bc_kind.compare("Dirichlet")==0 && (face==Uintah::Patch::zminus || face==Uintah::Patch::zplus))                                      \
          set_bc_staggered<ZFieldT> (patch,graphHelper,phiTag,bndFaceIJK,bc_value);                                                  \
        else                                                                                                                         \
          set_bc< ZFieldT, BCOpTypeSelector<ZFieldT,BCEvalT>::BCT >( patch, graphHelper, phiTag, bndFaceIJK, SIDE, bc_value, opdb ); \
        break;		                                                                                                             \
      case NODIR:	                                                                                                             \
        typedef SS::SVolField  SFieldT;                                                                                              \
        set_bc< SFieldT, BCOpTypeSelector<SFieldT,BCEvalT>::BCT >( patch, graphHelper, phiTag, bndFaceIJK, SIDE, bc_value, opdb );   \
        break;                                                                                                                       \
      default:                                                                                                                       \
        break;                                                                                                                       \
    }                                                                                                                                \
  }
  
  //-----------------------------------------------------------------------------
  // Sets Dirichlet values on the field directly
  template < typename FieldT >
  void set_bc_staggered( const Uintah::Patch* const patch,
              const GraphHelper& gh,
              const Expr::Tag phiTag,
              const SpatialOps::structured::IntVec& bcPointIndex,
              const double bcValue)
  {
    typedef SpatialOps::structured::ConstValEval BCVal;
    typedef SpatialOps::structured::BoundaryCondition<FieldT,BCVal> BC;
    Expr::ExpressionFactory& factory = *gh.exprFactory;
    const Expr::ExpressionID phiID = factory.get_registry().get_id(phiTag);
    Expr::Expression<FieldT>& phiExpr = dynamic_cast<Expr::Expression<FieldT>&>( factory.retrieve_expression( phiID, patch->getID(), true ) );    
    BC bound_cond(bcPointIndex, BCVal(bcValue));
    phiExpr.process_after_evaluate( bound_cond );                      
  }
  
  #define SET_BC_TAU( BCEvalT,      /* type of bc evaluator */                                  \
                      BCT )         /* type of BC */                                            \
  std::cout<<"SETTING BOUNDARY CONDITION ON "<< phiName <<std::endl;                            \
  for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {                                    \
    SCIRun::IntVector bc_point_indices(*bound_ptr);                                             \
    const SS::IntVec bcPointIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]);   \
    std::cout<<bc_point_indices<<std::endl;                                                     \
    switch (staggeredLocation) {                                                                \
    case XDIR:                                                                                  \
      typedef SS::XVolField  XFieldT;                                                           \
      typedef SS::FaceTypes<SS::XVolField>::XFace  XFaceXFieldT;                                \
      set_bc< XFaceXFieldT, BCOpTypeSelector<XFieldT,BCEvalT>::NeumannTauX >( patch, graphHelper, phiTag, bcPointIJK, bc_value, opdb ); \
      break;                                                                                    \
    case YDIR:                                                                                  \
      typedef SS::YVolField  YFieldT;                                                           \
      typedef SS::FaceTypes<SS::YVolField>::YFace  YFaceYFieldT;                                \
      set_bc< YFaceYFieldT, BCOpTypeSelector<YFieldT,BCEvalT>::NeumannTauY >( patch, graphHelper, phiTag, bcPointIJK, bc_value, opdb ); \
      break;                                                                                    \
    case ZDIR:                                                                                  \
      typedef SS::ZVolField  ZFieldT;                                                           \
      typedef SS::FaceTypes<SS::ZVolField>::ZFace  ZFaceZFieldT;                                \
      set_bc< ZFaceZFieldT, BCOpTypeSelector<ZFieldT,BCEvalT>::NeumannTauZ >( patch, graphHelper, phiTag, bcPointIJK, bc_value, opdb ); \
      break;                                                                                    \
    case NODIR:                                                                                 \
      break;                                                                                    \
    default:                                                                                    \
      break;                                                                                    \
    } // switch                                                                                 \
  }
  
  //
  // this macro will be unwrapped inside the build_bcs method.  The
  // variable names correspond to those defined within the appropriate
  // scope of that method.
  //
  
//	#define SET_BC_EXPR( BCEvalT,  /* type of bc evaluator */                                                               \
//                           BCT,      /* type of BC */                                                                         \
//                           SIDE )                                                                                             \
//	for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {                                                              \
//        SCIRun::IntVector bc_point_indices(*bound_ptr);                                                                       \
//        const SS::IntVec bcPointIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]);                             \
//        if (isNormalStress) {										                        \
//          switch (staggeredLocation) {                                                                                        \
//          case XDIR:                                                                                                          \
//	      typedef SS::FaceTypes<SS::XVolField>::XFace  XFieldT;                                                             \
//            set_bc< XFieldT, BCOpTypeSelector<XFieldT,BCEvalT>::NeumannTauX >( fieldExpr, bcPointIJK, SIDE, bc_value, opdb ); \
// 	      break;                                                                                                            \
//          case YDIR:                                                                                                          \
//            typedef SS::FaceTypes<SS::YVolField>::YFace  YFieldT;                                                             \
//            set_bc< YFieldT, BCOpTypeSelector<YFieldT,BCEvalT>::NeumannTauY >( fieldExpr, bcPointIJK, SIDE, bc_value, opdb ); \
//            break;                                                                                                            \
//          case ZDIR:                                                                                                          \
//            typedef SS::FaceTypes<SS::ZVolField>::ZFace  ZFieldT;                                                             \
//            set_bc< ZFieldT, BCOpTypeSelector<ZFieldT,BCEvalT>::NeumannTauZ >( fieldExpr, bcPointIJK, SIDE, bc_value, opdb ); \
//            break;                                                                                                            \
//          case NODIR:	                                                                                                        \
//            break;                                                                                                            \
//          default:                                                                                                            \
//            break;                                                                                                            \
//          } // switch( staggeredLocation )                                                                                    \
//	  }                                                                                                                     \
//        else {                                                                                                                \
//          switch (staggeredLocation) {                                                                                        \
//          case XDIR:                                                                                                          \
//	      typedef SS::XVolField  XFieldT;                                                                                   \
//	      set_bc< XFieldT, BCOpTypeSelector<XFieldT,BCEvalT>::BCT >( fieldExpr, bcPointIJK, SIDE, bc_value, opdb );         \
//            break;                                                                                                            \
//          case YDIR:                                                                                                          \
//            typedef SS::YVolField  YFieldT;                                                                                   \
//            set_bc< YFieldT, BCOpTypeSelector<YFieldT,BCEvalT>::BCT >( fieldExpr, bcPointIJK, SIDE, bc_value, opdb );         \
//            break;                                                                                                            \
//          case ZDIR:                                                                                                          \
//            typedef SS::ZVolField  ZFieldT;                                                                                   \
//            set_bc< ZFieldT, BCOpTypeSelector<ZFieldT,BCEvalT>::BCT >( fieldExpr, bcPointIJK, SIDE, bc_value, opdb );         \
//            break;                                                                                                            \
//          case NODIR:	                                                                                                        \
//            typedef SS::SVolField  SFieldT;                                                                                   \
//            set_bc< SFieldT, BCOpTypeSelector<SFieldT,BCEvalT>::BCT >( fieldExpr, bcPointIJK, SIDE, bc_value, opdb );         \
//            break;                                                                                                            \
//          default:                                                                                                            \
//            break;                                                                                                            \
//          }                                                                                                                   \
//        }                                                                                                                     \
//	}
  
  //-----------------------------------------------------------------------------
  
  template < typename FieldT, typename BCOpT >
  void set_bc( const Uintah::Patch* const patch,
               const GraphHelper& gh,
               const Expr::Tag phiTag,
               const SpatialOps::structured::IntVec& bcPointIndex,
               const SpatialOps::structured::BCSide bcSide,
               const double bcValue,
               const SpatialOps::OperatorDatabase& opdb )
  {
    typedef typename BCOpT::BCEvalT BCEvaluator;
    Expr::ExpressionFactory& factory = *gh.exprFactory;
    const Expr::ExpressionID phiID = factory.get_registry().get_id(phiTag);
    Expr::Expression<FieldT>& phiExpr = dynamic_cast<Expr::Expression<FieldT>&>( factory.retrieve_expression( phiID, patch->getID(), true ) );
    BCOpT bcOp( bcPointIndex, bcSide, BCEvaluator(bcValue), opdb );
    phiExpr.process_after_evaluate( bcOp );
  }
  
  //-----------------------------------------------------------------------------

  template < typename FieldT, typename BCOpT >
  void set_bc_flux( const Uintah::Patch* const patch,
              const GraphHelper& gh,
              const Expr::Tag phiTag,
              const SpatialOps::structured::IntVec& bcPointIndex,
              const double bcValue,
              const SpatialOps::OperatorDatabase& opdb )
  {
    typedef typename BCOpT::BCEvalT BCEvaluator;
    Expr::ExpressionFactory& factory = *gh.exprFactory;
    const Expr::ExpressionID phiID = factory.get_registry().get_id(phiTag);
    Expr::Expression<FieldT>& phiExpr = dynamic_cast<Expr::Expression<FieldT>&>( factory.retrieve_expression( phiID, patch->getID(), true ) );
    BCOpT bcOp( bcPointIndex, BCEvaluator(bcValue), opdb );
    phiExpr.process_after_evaluate( bcOp );
  }
  
  
//  //-----------------------------------------------------------------------------
//  template < typename FieldT, typename BCOpT >
//  void set_bc( Expr::Expression<FieldT>& phiExpr,
//              const SpatialOps::structured::IntVec& bcPointIndex,
//              const double bcValue,
//              const SpatialOps::OperatorDatabase& opdb )
//  {
//    typedef typename BCOpT::BCEvalT BCEvaluator;
//    BCOpT bcOp( bcPointIndex, BCEvaluator(bcValue), opdb );
//    phiExpr.process_after_evaluate( bcOp );
//  }
//  
//  
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

    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::DivX,       BCEvalT >   NeumannTauX;
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::DivY,       BCEvalT >   NeumannTauY;
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::DivZ,       BCEvalT >   NeumannTauZ;
  };
  
  //-----------------------------------------------------------------------------
  
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
    
    // Did I find an iterator
    return( bc_kind.compare("NotSet") != 0 );
  }
  
  //-----------------------------------------------------------------------------

  void process_boundary_conditions( const Expr::Tag phiTag,
                     const Direction staggeredLocation,
                     const GraphHelper& graphHelper,
                     const Uintah::PatchSet* const localPatches,
                     const PatchInfoMap& patchInfoMap,
                     const Uintah::MaterialSubset* const materials,
                     bool isNormalStress)
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
            std::cout << "Inside Cell Dir: \n" << insideCellDir << std::endl;

            //get the number of children
            // jcs note that we need to do some error checking here.
            // If the BC has not been set then we get a cryptic error
            // from Uintah.
            const int numChildren = patch->getBCDataArray(face)->getNumberChildren(materialID);
            
            for( int child = 0; child<numChildren; ++child ){
              
              double bc_value = -9; 
              std::string bc_kind = "NotSet";
              SCIRun::Iterator bound_ptr;
              // NOTE TO SELF: The Uintah boundary iterator, in the absence of extra cells, as is the case in Wasatch,
              // will give the zero-based ijk indices of the SCALAR interior cells, adjacent to the boundary.
              // ALSO NOTE: that even with staggered scalar Wasatch fields, there is NO additional ghost cell on the x+ face. So
              // nx_staggered = nx_scalar
              bool foundIterator = get_iter_bcval_bckind( patch, face, child, phiName, materialID, bc_value, bound_ptr, bc_kind);
              
              if (foundIterator) {

                SS::IntVec faceOffset(0,0,0);
                switch( face ){
                  case Uintah::Patch::xminus:                                   break;
                  case Uintah::Patch::xplus : faceOffset = SS::IntVec(1,0,0);   break;
                  case Uintah::Patch::yminus:                                   break;
                  case Uintah::Patch::yplus : faceOffset = SS::IntVec(0,1,0);   break;
                  case Uintah::Patch::zminus:                                   break;
                  case Uintah::Patch::zplus : faceOffset = SS::IntVec(0,0,1);   break;
                  default:                                                      break;
                }                
                
                if( bc_kind.compare("Dirichlet")==0 ){
                  
                  switch( face ){
                    case Uintah::Patch::xminus:  SET_BC( BCEvaluator, DirichletX, SpatialOps::structured::MINUS_SIDE );  break;
                    case Uintah::Patch::xplus :  SET_BC( BCEvaluator, DirichletX, SpatialOps::structured::PLUS_SIDE  );  break;
                    case Uintah::Patch::yminus:  SET_BC( BCEvaluator, DirichletY, SpatialOps::structured::MINUS_SIDE );  break;
                    case Uintah::Patch::yplus :  SET_BC( BCEvaluator, DirichletY, SpatialOps::structured::PLUS_SIDE  );  break;
                    case Uintah::Patch::zminus:  SET_BC( BCEvaluator, DirichletZ, SpatialOps::structured::MINUS_SIDE );  break;
                    case Uintah::Patch::zplus :  SET_BC( BCEvaluator, DirichletZ, SpatialOps::structured::PLUS_SIDE  );  break;
                    case Uintah::Patch::numFaces:
                      throw Uintah::ProblemSetupException( "An invalid face Patch::numFaces was encountered while setting boundary conditions", __FILE__, __LINE__ );
                      break;
                    case Uintah::Patch::invalidFace:
                      throw Uintah::ProblemSetupException( "An invalid face Patch::invalidFace was encountered while setting boundary conditions", __FILE__, __LINE__ );
                      break;
                  }
                  
                } else if (bc_kind.compare("Neumann")==0 ){
                  if (isNormalStress) {
                    switch( face ){
                      case Uintah::Patch::xminus:  SET_BC( BCEvaluator, NeumannX, SpatialOps::structured::MINUS_SIDE );  break;
                      case Uintah::Patch::xplus :  SET_BC( BCEvaluator, NeumannX, SpatialOps::structured::PLUS_SIDE  );  break;
                      case Uintah::Patch::yminus:  SET_BC( BCEvaluator, NeumannY, SpatialOps::structured::MINUS_SIDE );  break;
                      case Uintah::Patch::yplus :  SET_BC( BCEvaluator, NeumannY, SpatialOps::structured::PLUS_SIDE  );  break;
                      case Uintah::Patch::zminus:  SET_BC( BCEvaluator, NeumannZ, SpatialOps::structured::MINUS_SIDE );  break;
                      case Uintah::Patch::zplus :  SET_BC( BCEvaluator, NeumannZ, SpatialOps::structured::PLUS_SIDE  );  break;
                      case Uintah::Patch::numFaces:
                        throw Uintah::ProblemSetupException( "An invalid face Patch::numFaces was encountered while setting boundary conditions", __FILE__, __LINE__ );
                        break;
                      case Uintah::Patch::invalidFace:
                        throw Uintah::ProblemSetupException( "An invalid face Patch::invalidFace was encountered while setting boundary conditions", __FILE__, __LINE__ );
                        break;
                    }    
                  } else {
                    switch( face ){
                      case Uintah::Patch::xminus:  SET_BC( BCEvaluator, NeumannX, SpatialOps::structured::MINUS_SIDE );  break;
                      case Uintah::Patch::xplus :  SET_BC( BCEvaluator, NeumannX, SpatialOps::structured::PLUS_SIDE  );  break;
                      case Uintah::Patch::yminus:  SET_BC( BCEvaluator, NeumannY, SpatialOps::structured::MINUS_SIDE );  break;
                      case Uintah::Patch::yplus :  SET_BC( BCEvaluator, NeumannY, SpatialOps::structured::PLUS_SIDE  );  break;
                      case Uintah::Patch::zminus:  SET_BC( BCEvaluator, NeumannZ, SpatialOps::structured::MINUS_SIDE );  break;
                      case Uintah::Patch::zplus :  SET_BC( BCEvaluator, NeumannZ, SpatialOps::structured::PLUS_SIDE  );  break;
                      case Uintah::Patch::numFaces:
                        throw Uintah::ProblemSetupException( "An invalid face Patch::numFaces was encountered while setting boundary conditions", __FILE__, __LINE__ );
                        break;
                      case Uintah::Patch::invalidFace:
                        throw Uintah::ProblemSetupException( "An invalid face Patch::invalidFace was encountered while setting boundary conditions", __FILE__, __LINE__ );
                        break;
                    }    
                  }              
                }
              }
            } // child loop
          } // face loop
        } // material loop
      } // patch subset loop
    } // local patch loop
  }

  //-----------------------------------------------------------------------------

  void set_pressure_bc( const Expr::Tag pressureTag, 
                       Uintah::CCVariable<Uintah::Stencil7>& pressureMatrix,
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
    const double dx2 = spacing[0]*spacing[0];
    const double dy2 = spacing[1]*spacing[1];
    const double dz2 = spacing[2]*spacing[2];
    const int materialID = 0;

    const std::string phiName = pressureTag.name();
    // loop over local patches
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
        bool foundIterator = get_iter_bcval_bckind( patch, face, child, phiName, materialID, bc_value, bound_ptr, bc_kind);
        
        if (foundIterator) {
          
          if ( bc_kind.compare("Dirichlet") == 0 ) {

            switch( face ){
                
              case Uintah::Patch::xminus:  
                for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
                  SCIRun::IntVector bc_point_indices(*bound_ptr);                     
                  pressureMatrix[bc_point_indices].w = 0.0;

                  // get flat index and manually set values for pressure and pressure rhs
                  const SS::IntVec   intCellIJK( bc_point_indices[0],   bc_point_indices[1], bc_point_indices[2] );
                  const SS::IntVec ghostCellIJK( bc_point_indices[0]-1, bc_point_indices[1], bc_point_indices[2] );
                  const int iInterior = pressureField.window_without_ghost().flat_index(   intCellIJK );
                  const int iGhost    = pressureField.window_without_ghost().flat_index( ghostCellIJK );
                  pressureField[iGhost] = 2*bc_value - pressureField[iInterior];
                  pressureRHS[iInterior] -= pressureField[iGhost]/dx2;
                }
              break;
              case Uintah::Patch::xplus:  
                for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
                  SCIRun::IntVector bc_point_indices(*bound_ptr);
                  pressureMatrix[bc_point_indices].e = 0.0;

                  // get flat index
                  const SS::IntVec   intCellIJK( bc_point_indices[0],   bc_point_indices[1], bc_point_indices[2] );
                  const SS::IntVec ghostCellIJK( bc_point_indices[0]+1, bc_point_indices[1], bc_point_indices[2] );
                  const int iInterior = pressureField.window_without_ghost().flat_index(   intCellIJK );
                  const int iGhost    = pressureField.window_without_ghost().flat_index( ghostCellIJK );
                  pressureField[iGhost] = 2*bc_value - pressureField[iInterior];
                  pressureRHS[iInterior] -= pressureField[iGhost]/dx2;                                                
                }
              break;
              case Uintah::Patch::yminus:
                for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
                  SCIRun::IntVector bc_point_indices(*bound_ptr);
                  pressureMatrix[bc_point_indices].s = 0.0;

                  // get flat index
                  const SS::IntVec   intCellIJK( bc_point_indices[0], bc_point_indices[1],   bc_point_indices[2] );
                  const SS::IntVec ghostCellIJK( bc_point_indices[0], bc_point_indices[1]-1, bc_point_indices[2] );
                  const int iInterior = pressureField.window_without_ghost().flat_index(   intCellIJK );
                  const int iGhost    = pressureField.window_without_ghost().flat_index( ghostCellIJK );
                  pressureField[iGhost] = 2*bc_value - pressureField[iInterior];
                  pressureRHS[iInterior] -= pressureField[iGhost]/dy2;                                                                        
                }
                break;
              case Uintah::Patch::yplus:  
                for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
                  SCIRun::IntVector bc_point_indices(*bound_ptr);                     
                  pressureMatrix[bc_point_indices].n = 0.0;
                  
                  // get flat index
                  const SS::IntVec   intCellIJK( bc_point_indices[0], bc_point_indices[1]  , bc_point_indices[2] );
                  const SS::IntVec ghostCellIJK( bc_point_indices[0], bc_point_indices[1]+1, bc_point_indices[2] );
                  const int iInterior = pressureField.window_without_ghost().flat_index(   intCellIJK );
                  const int iGhost    = pressureField.window_without_ghost().flat_index( ghostCellIJK );
                  pressureField[iGhost] = 2*bc_value - pressureField[iInterior];
                  pressureRHS[iInterior] -= pressureField[iGhost]/dy2;                                                                                                
                }
                break;
              case Uintah::Patch::zminus:  
                for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
                  SCIRun::IntVector bc_point_indices(*bound_ptr);                     
                  pressureMatrix[bc_point_indices].b = 0.0;
                  
                  // get flat index
                  const SS::IntVec   intCellIJK( bc_point_indices[0], bc_point_indices[1], bc_point_indices[2]   );
                  const SS::IntVec ghostCellIJK( bc_point_indices[0], bc_point_indices[1], bc_point_indices[2]-1 );
                  const int iInterior = pressureField.window_without_ghost().flat_index(   intCellIJK );
                  const int iGhost    = pressureField.window_without_ghost().flat_index( ghostCellIJK );
                  pressureField[iGhost] = 2*bc_value - pressureField[iInterior];
                  pressureRHS[iInterior] -= pressureField[iGhost]/dz2;                                                                                                
                }
                break;
              case Uintah::Patch::zplus:  
                for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
                  SCIRun::IntVector bc_point_indices(*bound_ptr);                     
                  pressureMatrix[bc_point_indices].t = 0.0;
                  
                  // get flat index
                  const SS::IntVec   intCellIJK( bc_point_indices[0], bc_point_indices[1], bc_point_indices[2]   );
                  const SS::IntVec ghostCellIJK( bc_point_indices[0], bc_point_indices[1], bc_point_indices[2]+1 );
                  const int iInterior = pressureField.window_without_ghost().flat_index(   intCellIJK );
                  const int iGhost    = pressureField.window_without_ghost().flat_index( ghostCellIJK );
                  pressureField[iGhost] = 2*bc_value - pressureField[iInterior];
                  pressureRHS[iInterior] -= pressureField[iGhost]/dz2;                                                                                                
                }
                break;
                
              case Uintah::Patch::numFaces:
                throw Uintah::ProblemSetupException( "An invalid face of type Patch::numFaces was encountered while setting boundary conditions", __FILE__, __LINE__ );
                break;
                
              case Uintah::Patch::invalidFace:
                throw Uintah::ProblemSetupException( "An invalid face of type Patch::invalidFace was encountered while setting boundary conditions", __FILE__, __LINE__ );
                break;
            }
            
          }
          else if( bc_kind.compare("Neumann") == 0 ){
            
            switch( face ){
              case Uintah::Patch::xminus:  
                
                for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
                  SCIRun::IntVector bc_point_indices(*bound_ptr);                     
                  pressureMatrix[bc_point_indices].w = 0.0;
                  
                  // get flat index
                  const SS::IntVec   intCellIJK( bc_point_indices[0],   bc_point_indices[1], bc_point_indices[2] );
                  const SS::IntVec ghostCellIJK( bc_point_indices[0]-1, bc_point_indices[1], bc_point_indices[2] );

                  const int iInterior = pressureField.window_without_ghost().flat_index(   intCellIJK );
                  const int iGhost    = pressureField.window_without_ghost().flat_index( ghostCellIJK );
                  
                  pressureField[iGhost] = pressureField[iInterior] - 2*spacing[0]*bc_value;
                  pressureRHS[iInterior] -= pressureField[iGhost]/dx2;
                }
                break;
              case Uintah::Patch::xplus:  
                for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
                  SCIRun::IntVector bc_point_indices(*bound_ptr);                     
                  pressureMatrix[bc_point_indices].e = 0.0;
                  
                  // get flat index
                  const SS::IntVec   intCellIJK( bc_point_indices[0],   bc_point_indices[1], bc_point_indices[2] );
                  const SS::IntVec ghostCellIJK( bc_point_indices[0]+1, bc_point_indices[1], bc_point_indices[2] );
                  const int iInterior = pressureField.window_without_ghost().flat_index(   intCellIJK );
                  const int iGhost    = pressureField.window_without_ghost().flat_index( ghostCellIJK );

                  pressureField[iGhost] = - pressureField[iInterior] + 2*spacing[0]*bc_value;
                  pressureRHS[iInterior] -= pressureField[iGhost]/dx2;                                                
                }
                break;
              case Uintah::Patch::yminus:  
                for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
                  SCIRun::IntVector bc_point_indices(*bound_ptr);                     
                  pressureMatrix[bc_point_indices].s = 0.0;
                  
                  // get flat index
                  const SS::IntVec   intCellIJK( bc_point_indices[0], bc_point_indices[1],   bc_point_indices[2] );
                  const SS::IntVec ghostCellIJK( bc_point_indices[0], bc_point_indices[1]-1, bc_point_indices[2] );
                  const int iInterior = pressureField.window_without_ghost().flat_index(   intCellIJK );
                  const int iGhost    = pressureField.window_without_ghost().flat_index( ghostCellIJK );

                  pressureField[iGhost] = pressureField[iInterior] - 2*spacing[1]*bc_value;
                  pressureRHS[iInterior] -= pressureField[iGhost]/dy2;                                                                        
                }
                break;
              case Uintah::Patch::yplus:  
                for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
                  SCIRun::IntVector bc_point_indices(*bound_ptr);                     
                  pressureMatrix[bc_point_indices].n = 0.0;
                  
                  // get flat index
                  const SS::IntVec   intCellIJK( bc_point_indices[0], bc_point_indices[1],   bc_point_indices[2] );
                  const SS::IntVec ghostCellIJK( bc_point_indices[0], bc_point_indices[1]+1, bc_point_indices[2] );
                  const int iInterior = pressureField.window_without_ghost().flat_index(   intCellIJK );
                  const int iGhost    = pressureField.window_without_ghost().flat_index( ghostCellIJK );

                  pressureField[iGhost] = - pressureField[iInterior] + 2*spacing[1]*bc_value;
                  pressureRHS[iInterior] -= pressureField[iGhost]/dy2;                                                                                                
                }
                break;
              case Uintah::Patch::zminus:  
                for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
                  SCIRun::IntVector bc_point_indices(*bound_ptr);                     
                  pressureMatrix[bc_point_indices].b = 0.0;
                  
                  // get flat index
                  const SS::IntVec   intCellIJK( bc_point_indices[0], bc_point_indices[1], bc_point_indices[2]   );
                  const SS::IntVec ghostCellIJK( bc_point_indices[0], bc_point_indices[1], bc_point_indices[2]-1 );
                  const int iInterior = pressureField.window_without_ghost().flat_index(   intCellIJK );
                  const int iGhost    = pressureField.window_without_ghost().flat_index( ghostCellIJK );

                  pressureField[iGhost] = pressureField[iInterior] - 2*spacing[2]*bc_value;
                  pressureRHS[iInterior] -= pressureField[iGhost]/dz2;                                                                                                
                }
                break;
              case Uintah::Patch::zplus:  
                for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {
                  SCIRun::IntVector bc_point_indices(*bound_ptr);                     
                  pressureMatrix[bc_point_indices].t = 0.0;
                  
                  // get flat index
                  const SS::IntVec   intCellIJK( bc_point_indices[0], bc_point_indices[1], bc_point_indices[2]   );
                  const SS::IntVec ghostCellIJK( bc_point_indices[0], bc_point_indices[1], bc_point_indices[2]+1 );
                  const int iInterior = pressureField.window_without_ghost().flat_index(   intCellIJK );
                  const int iGhost    = pressureField.window_without_ghost().flat_index( ghostCellIJK );

                  pressureField[iGhost] = -pressureField[iInterior] + 2*spacing[2]*bc_value;
                  pressureRHS[iInterior] -= pressureField[iGhost]/dz2;                                                                                                
                }
                break;
                
              case Uintah::Patch::numFaces:
                throw Uintah::ProblemSetupException( "An invalid face of type Patch::numFaces was encountered while setting boundary conditions", __FILE__, __LINE__ );
                break;
                
              case Uintah::Patch::invalidFace:
                throw Uintah::ProblemSetupException( "An invalid face of type Patch::invalidFace was encountered while setting boundary conditions", __FILE__, __LINE__ );
                break;
            }                
          }
        }
      } // child loop
    } // face loop
  }
                              
//  //-----------------------------------------------------------------------------
//  template< typename FieldT>
//  void build_bcs( Expr::Expression<FieldT>& fieldExpr,
//                 const Direction staggeredLocation,
//                 const GraphHelper& graphHelper,
//                 const Uintah::PatchSet* const localPatches,
//                 const PatchInfoMap& patchInfoMap,
//                 const Uintah::MaterialSubset* const materials )
//  {
//    /*
//     ALGORITHM:
//     1. loop over the patches
//     2. For each patch, loop over materials
//     3. For each material, loop over boundary faces
//     4. For each boundary face, loop over its children
//     5. For each child, get the cell faces and set appropriate
//     boundary conditions
//     */
//    // loop over all patches, and for each patch set boundary conditions  
//    std::string phiName = fieldExpr.name().name();
//    namespace SS = SpatialOps::structured;
//    typedef SS::ConstValEval BCEvaluator; // basic functor for constant functions.
//    // loop over local patches
//    for( int ip=0; ip<localPatches->size(); ++ip ){
//      
//      // get the patch subset
//      const Uintah::PatchSubset* const patches = localPatches->getSubset(ip);
//      
//      // loop over every patch in the patch subset
//      for( int ipss=0; ipss<patches->size(); ++ipss ){          
//        
//        // get a pointer to the current patch
//        const Uintah::Patch* const patch = patches->get(ipss);          
//        
//        // get the patch info from which we can get the operators database
//        const PatchInfoMap::const_iterator ipi = patchInfoMap.find( patch->getID() );
//        assert( ipi != patchInfoMap.end() );
//        const SpatialOps::OperatorDatabase& opdb = *(ipi->second.operators);
//        
//        // loop over materials
//        for( int im=0; im<materials->size(); ++im ){
//          
//          const int materialID = materials->get(im);
//          
//          std::vector<Uintah::Patch::FaceType> bndFaces;
//          patch->getBoundaryFaces(bndFaces);
//          std::vector<Uintah::Patch::FaceType>::const_iterator faceIterator = bndFaces.begin();
//          
//          // loop over the boundary faces
//          for( ; faceIterator!=bndFaces.end(); ++faceIterator ){
//            Uintah::Patch::FaceType face = *faceIterator;
//            
//            //get the number of children
//            const int numChildren = patch->getBCDataArray(face)->getNumberChildren(materialID);
//            
//            for( int child = 0; child<numChildren; ++child ){
//              
//              double bc_value = -9; 
//              std::string bc_kind = "NotSet";
//              SCIRun::Iterator bound_ptr;
//              bool foundIterator = get_iter_bcval_bckind( patch, face, child, phiName, materialID, bc_value, bound_ptr, bc_kind);
//              
//              if (foundIterator) {
//                
//                if( bc_kind.compare("Dirichlet") == 0 ){
//                  
//                  switch( face ){
//                    case Uintah::Patch::xminus:  SET_BC_EXPR( BCEvaluator, DirichletX, SpatialOps::structured::X_MINUS_SIDE );  break;
//                    case Uintah::Patch::xplus :  SET_BC_EXPR( BCEvaluator, DirichletX, SpatialOps::structured::X_PLUS_SIDE  );  break;
//                    case Uintah::Patch::yminus:  SET_BC_EXPR( BCEvaluator, DirichletY, SpatialOps::structured::Y_MINUS_SIDE );  break;
//                    case Uintah::Patch::yplus :  SET_BC_EXPR( BCEvaluator, DirichletY, SpatialOps::structured::Y_PLUS_SIDE  );  break;
//                    case Uintah::Patch::zminus:  SET_BC_EXPR( BCEvaluator, DirichletZ, SpatialOps::structured::Z_MINUS_SIDE );  break;
//                    case Uintah::Patch::zplus :  SET_BC_EXPR( BCEvaluator, DirichletZ, SpatialOps::structured::Z_PLUS_SIDE  );  break;
//                    case Uintah::Patch::numFaces:
//                      throw Uintah::ProblemSetupException( "numFaces is not a valid face", __FILE__, __LINE__ );
//                      break;
//                    case Uintah::Patch::invalidFace:
//                      throw Uintah::ProblemSetupException( "invalidFace is not a valid face", __FILE__, __LINE__ );
//                      break;
//                  }
//                  
//                } else if( bc_kind.compare("Neumann") == 0 ){
//                  
//                  switch( face ){
//                    case Uintah::Patch::xminus:  SET_BC_EXPR( BCEvaluator, NeumannX, SpatialOps::structured::X_MINUS_SIDE );  break;
//                    case Uintah::Patch::xplus :  SET_BC_EXPR( BCEvaluator, NeumannX, SpatialOps::structured::X_PLUS_SIDE  );  break;
//                    case Uintah::Patch::yminus:  SET_BC_EXPR( BCEvaluator, NeumannY, SpatialOps::structured::Y_MINUS_SIDE );  break;
//                    case Uintah::Patch::yplus :  SET_BC_EXPR( BCEvaluator, NeumannY, SpatialOps::structured::Y_PLUS_SIDE  );  break;
//                    case Uintah::Patch::zminus:  SET_BC_EXPR( BCEvaluator, NeumannZ, SpatialOps::structured::Z_MINUS_SIDE );  break;
//                    case Uintah::Patch::zplus :  SET_BC_EXPR( BCEvaluator, NeumannZ, SpatialOps::structured::Z_PLUS_SIDE  );  break;
//                    case Uintah::Patch::numFaces:
//                      throw Uintah::ProblemSetupException( "numFaces is not a valid face", __FILE__, __LINE__ );
//                      break;
//                    case Uintah::Patch::invalidFace:
//                      throw Uintah::ProblemSetupException( "invalidFace is not a valid face", __FILE__, __LINE__ );
//                      break;
//                  }                  
//                }
//              }
//            } // child loop
//          } // face loop
//        } // material loop
//      } // patch subset loop
//    } // local patch loop
//  }
  
} // namespace wasatch

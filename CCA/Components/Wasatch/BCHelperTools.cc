//-- Uintah framework includes --//
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>

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
  
  #define SET_BC( BCEvalT,      /* type of bc evaluator */                \
                  BCT,          /* type of BC */                          \
                  SIDE )                                                  \
    proc0cout << "SETTING BOUNDARY CONDITION ON "<< phiName << " ON SIDE " << SIDE <<std::endl;    \
    for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {            \
      SCIRun::IntVector bc_point_indices(*bound_ptr);                     \
      const SS::IntVec bcPointIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]); \
      switch (staggeredLocation) {                                        \
      case XDIR:                                                          \
        typedef SS::XVolField  XFieldT;                                    \
        if (bc_kind=="Dirichlet" && SIDE==SS::X_MINUS_SIDE) { 							\
  					set_bc_staggered<XFieldT> (patch,graphHelper,phiTag,bcPointIJK,bc_value);\
						const SS::IntVec ghostCellIJK(bc_point_indices[0] - 1,bc_point_indices[1],bc_point_indices[2]); \
  					set_bc_staggered<XFieldT> (patch,graphHelper,phiTag,ghostCellIJK,bc_value);     \
				}\
        else if (bc_kind=="Dirichlet" && SIDE==SS::X_PLUS_SIDE) {								\
            const SS::IntVec ghostCellIJK(bc_point_indices[0] + 1,bc_point_indices[1],bc_point_indices[2]); \
  					set_bc_staggered<XFieldT> (patch,graphHelper,phiTag,ghostCellIJK,bc_value);\
        }																																		\
        else set_bc< XFieldT, BCOpTypeSelector<XFieldT,BCEvalT>::BCT >( patch, graphHelper, phiTag, bcPointIJK, SIDE, bc_value, opdb ); \
        break;                                                            \
      case YDIR:                                                          \
        typedef SS::YVolField  YFieldT;                                    \
        if (bc_kind=="Dirichlet" && SIDE==SS::Y_MINUS_SIDE) { 							\
          set_bc_staggered<YFieldT> (patch,graphHelper,phiTag,bcPointIJK,bc_value);\
          const SS::IntVec ghostCellIJK(bc_point_indices[0],bc_point_indices[1] - 1,bc_point_indices[2]); \
          set_bc_staggered<YFieldT> (patch,graphHelper,phiTag,ghostCellIJK,bc_value);     \
        }\
        else if (bc_kind=="Dirichlet" && SIDE==SS::Y_PLUS_SIDE) {								\
          const SS::IntVec ghostCellIJK(bc_point_indices[0],bc_point_indices[1] + 1,bc_point_indices[2]); \
          set_bc_staggered<YFieldT> (patch,graphHelper,phiTag,ghostCellIJK,bc_value);\
        }																																		\
        else set_bc< YFieldT, BCOpTypeSelector<YFieldT,BCEvalT>::BCT >( patch, graphHelper, phiTag, bcPointIJK, SIDE, bc_value, opdb ); \
        break;																														\
      case ZDIR:																													\
        typedef SS::ZVolField  ZFieldT;                                    \
        if (bc_kind=="Dirichlet" && SIDE==SS::Z_MINUS_SIDE) { 							\
          set_bc_staggered<ZFieldT> (patch,graphHelper,phiTag,bcPointIJK,bc_value);\
          const SS::IntVec ghostCellIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2] - 1); \
          set_bc_staggered<ZFieldT> (patch,graphHelper,phiTag,ghostCellIJK,bc_value);     \
        }\
        else if (bc_kind=="Dirichlet" && SIDE==SS::Z_PLUS_SIDE) {								\
          const SS::IntVec ghostCellIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2] + 1); \
          set_bc_staggered<ZFieldT> (patch,graphHelper,phiTag,ghostCellIJK,bc_value);\
        }																																		\
        else set_bc< ZFieldT, BCOpTypeSelector<ZFieldT,BCEvalT>::BCT >( patch, graphHelper, phiTag, bcPointIJK, SIDE, bc_value, opdb ); \
        break;																														\
      case NODIR:																													\
        typedef SS::SVolField  SFieldT;                                   \
        set_bc< SFieldT, BCOpTypeSelector<SFieldT,BCEvalT>::BCT >( patch, graphHelper, phiTag, bcPointIJK, SIDE, bc_value, opdb ); \
        break;																														\
      default:																														\
        break;																														\
    }																																		  \
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
    //const Expr::Tag phiLabel( phiName, Expr::STATE_N );
    Expr::ExpressionFactory& factory = *gh.exprFactory;
    const Expr::ExpressionID phiID = factory.get_registry().get_id(phiTag);
    Expr::Expression<FieldT>& phiExpr = dynamic_cast<Expr::Expression<FieldT>&>( factory.retrieve_expression( phiID, patch->getID(), true ) );    
    BCOpT bcOp( bcPointIndex, bcSide, BCEvaluator(bcValue), opdb );
    phiExpr.process_after_evaluate( bcOp );
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
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::InterpC2FX, BCEvalT > DirichletX;
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::InterpC2FY, BCEvalT > DirichletY;
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::InterpC2FZ, BCEvalT > DirichletZ;
    
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::GradX, BCEvalT > NeumannX;
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::GradY, BCEvalT > NeumannY;
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::GradZ, BCEvalT > NeumannZ;
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
    const Uintah::BoundCondBase* bc = patch->getArrayBCValues( face, mat_id,
                                                              desc, bound_ptr,
                                                              nu, child );
    const Uintah::BoundCond<T>* new_bcs = dynamic_cast<const Uintah::BoundCond<T>*>(bc);
    
    bc_value=T(-9);
    bc_kind="NotSet";
    if (new_bcs != 0) {      // non-symmetric
      bc_value = new_bcs->getValue();
      bc_kind =  new_bcs->getBCType__NEW();
    }
    delete bc;
    
    // Did I find an iterator
    return( bc_kind != "NotSet" );
  }
  
  //-----------------------------------------------------------------------------

  void process_boundary_conditions( const Expr::Tag phiTag,
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

            //get the number of children
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
                
                if (bc_kind == "Dirichlet") {
                  
                  switch( face ){
                    case Uintah::Patch::xminus:  SET_BC( BCEvaluator, DirichletX, SpatialOps::structured::X_MINUS_SIDE );  break;
                    case Uintah::Patch::xplus :  SET_BC( BCEvaluator, DirichletX, SpatialOps::structured::X_PLUS_SIDE  );  break;
                    case Uintah::Patch::yminus:  SET_BC( BCEvaluator, DirichletY, SpatialOps::structured::Y_MINUS_SIDE );  break;
                    case Uintah::Patch::yplus :  SET_BC( BCEvaluator, DirichletY, SpatialOps::structured::Y_PLUS_SIDE  );  break;
                    case Uintah::Patch::zminus:  SET_BC( BCEvaluator, DirichletZ, SpatialOps::structured::Z_MINUS_SIDE );  break;
                    case Uintah::Patch::zplus :  SET_BC( BCEvaluator, DirichletZ, SpatialOps::structured::Z_PLUS_SIDE  );  break;
                    case Uintah::Patch::numFaces:
                      throw Uintah::ProblemSetupException( "An invalid face Patch::numFaces was encountered while setting boundary conditions", __FILE__, __LINE__ );
                      break;
                    case Uintah::Patch::invalidFace:
                      throw Uintah::ProblemSetupException( "An invalid face Patch::invalidFace was encountered while setting boundary conditions", __FILE__, __LINE__ );
                      break;
                  }
                  
                } else if (bc_kind == "Neumann") {
                  
                  switch( face ){
                    case Uintah::Patch::xminus:  SET_BC( BCEvaluator, NeumannX, SpatialOps::structured::X_MINUS_SIDE );  break;
                    case Uintah::Patch::xplus :  SET_BC( BCEvaluator, NeumannX, SpatialOps::structured::X_PLUS_SIDE  );  break;
                    case Uintah::Patch::yminus:  SET_BC( BCEvaluator, NeumannY, SpatialOps::structured::Y_MINUS_SIDE );  break;
                    case Uintah::Patch::yplus :  SET_BC( BCEvaluator, NeumannY, SpatialOps::structured::Y_PLUS_SIDE  );  break;
                    case Uintah::Patch::zminus:  SET_BC( BCEvaluator, NeumannZ, SpatialOps::structured::Z_MINUS_SIDE );  break;
                    case Uintah::Patch::zplus :  SET_BC( BCEvaluator, NeumannZ, SpatialOps::structured::Z_PLUS_SIDE  );  break;
                    case Uintah::Patch::numFaces:
                      throw Uintah::ProblemSetupException( "An invalid face Patch::numFaces was encountered while setting boundary conditions", __FILE__, __LINE__ );
                      break;
                    case Uintah::Patch::invalidFace:
                      throw Uintah::ProblemSetupException( "An invalid face Patch::invalidFace was encountered while setting boundary conditions", __FILE__, __LINE__ );
                      break;
                  }
                  
                }
              }
            } // child loop
          } // face loop
        } // material loop
      } // patch subset loop
    } // local patch loop
  }
  
} // namespace wasatch

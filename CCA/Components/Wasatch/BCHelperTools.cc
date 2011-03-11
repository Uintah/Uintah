//-- Uintah framework includes --//
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/FVStaggered.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- Wasatch includes --//
#include "Operators/OperatorTypes.h"
#include "FieldTypes.h"
#include "GraphHelperTools.h"
#include "BCHelperTools.h"

namespace Wasatch {

//
// this macro will be unwrapped inside the build_bcs method.  The
// variable names correspond to those defined within the appropriate
// scope of that method.
//
#define SET_BC( BCEvalT,      /* type of bc evaluator */                \
                BCT,          /* type of BC */                          \
                SIDE )                                                  \
  std::cout<<"SETTING BOUNDARY CONDITION ON "<< phiName <<std::endl;    \
  for( bound_ptr.reset(); !bound_ptr.done(); bound_ptr++ ) {            \
    SCIRun::IntVector bc_point_indices(*bound_ptr);                     \
    const SS::IntVec bcPointIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]); \
    if( staggeredDirection=="X" ){                                      \
      typedef SS::XVolField FieldT;                                     \
      set_bc< FieldT, BCOpTypeSelector<FieldT,BCEvalT>::BCT >( patch, graphHelper, phiName, bcPointIJK, SIDE, bc_value, opdb ); \
    }                                                                   \
    else if( staggeredDirection=="Y" ){                                 \
      typedef SS::YVolField  FieldT;                                    \
      set_bc< FieldT, BCOpTypeSelector<FieldT,BCEvalT>::BCT >( patch, graphHelper, phiName, bcPointIJK, SIDE, bc_value, opdb ); \
    }                                                                   \
    else if( staggeredDirection=="Z" ){                                 \
      typedef SS::ZVolField  FieldT;                                    \
      set_bc< FieldT, BCOpTypeSelector<FieldT,BCEvalT>::BCT >( patch, graphHelper, phiName, bcPointIJK, SIDE, bc_value, opdb ); \
    }                                                                   \
    else{                                                               \
      typedef SS::SVolField  FieldT;                                    \
      set_bc< FieldT, BCOpTypeSelector<FieldT,BCEvalT>::BCT >( patch, graphHelper, phiName, bcPointIJK, SIDE, bc_value, opdb ); \
    }                                                                   \
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

  template < typename FieldT, typename BCOpT >
  void set_bc( const Uintah::Patch* const patch,
               const GraphHelper& gh,
               const std::string& phiName,
               const SpatialOps::structured::IntVec& bcPointIndex,
               const SpatialOps::structured::BCSide bcSide,
               const double bcValue,
               const SpatialOps::OperatorDatabase& opdb )
  {
    typedef typename BCOpT::BCEvalT BCEvaluator;
    Expr::ExpressionFactory& factory = *gh.exprFactory;
    const Expr::Tag phiLabel( phiName, Expr::STATE_N );
    const Expr::ExpressionID phiID = factory.get_registry().get_id(phiLabel);
    Expr::Expression<FieldT>& phiExpr = dynamic_cast<Expr::Expression<FieldT>&>( factory.retrieve_expression( phiID, patch->getID(), true ) );
    BCOpT bcOp( bcPointIndex, bcSide, BCEvaluator(bcValue), opdb );
    phiExpr.process_after_evaluate( bcOp );
  }
  
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
  
  void build_bcs( const std::vector<EqnTimestepAdaptorBase*>& eqnAdaptors,
                  const GraphHelper& graphHelper,
                  const Uintah::PatchSet* const localPatches,
                  const PatchInfoMap& patchInfoMap,
                  const Uintah::MaterialSubset* const materials )
  {
     /*
     ALGORITHM:
     Note: Adaptors have been modified to save the parameters of the transport
     equation that they save.
     1. Loop over adaptors
       2. For each adaptor, get the solution variable name
       3. For each adaptor, get the staggering direction
       4. For each adaptor, loop over the patches
          5. For each patch, loop over materials
             6. For each material, loop over boundary faces
                7. For each boundary face, loop over its children
                   8. For each child, get the cell faces and set appropriate
                        boundary conditions
    */
    // loop over all patches, and for each patch set boundary conditions  
    
    namespace SS = SpatialOps::structured;
    typedef SS::ConstValEval BCEvaluator; // basic functor for constant functions.
    typedef std::vector<EqnTimestepAdaptorBase*> EquationAdaptors;
    
    for( EquationAdaptors::const_iterator ia=eqnAdaptors.begin(); ia!=eqnAdaptors.end(); ++ia ){
      EqnTimestepAdaptorBase* const adaptor = *ia;
      
      // get the input parameters corresponding to this transport equation
      Uintah::ProblemSpecP transEqnParams = adaptor->transEqnParams();
      
      // get the variable name
      std::string phiName;
      transEqnParams->get("SolutionVariable",phiName);
      
      // find out if this corresponds to a staggered or non-staggered field
      std::string staggeredDirection;
      Uintah::ProblemSpecP scalarStaggeredParams = transEqnParams->get( "StaggeredDirection", staggeredDirection );
      
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
              
              //get the number of children
              const int numChildren = patch->getBCDataArray(face)->getNumberChildren(materialID);
              
              for( int child = 0; child<numChildren; ++child ){

                double bc_value = -9; 
                std::string bc_kind = "NotSet";
                SCIRun::Iterator bound_ptr;
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
                      throw Uintah::ProblemSetupException( "numFaces is not a valid face", __FILE__, __LINE__ );
                      break;
                    case Uintah::Patch::invalidFace:
                      throw Uintah::ProblemSetupException( "invalidFace is not a valid face", __FILE__, __LINE__ );
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
                      throw Uintah::ProblemSetupException( "numFaces is not a valid face", __FILE__, __LINE__ );
                      break;
                    case Uintah::Patch::invalidFace:
                      throw Uintah::ProblemSetupException( "invalidFace is not a valid face", __FILE__, __LINE__ );
                      break;
                    }
                    
                  }
                }
              } // child loop
            } // face loop
          } // material loop
        } // patch subset loop
      } // local patch loop
    } // equation loop

  }


} // namespace wasatch

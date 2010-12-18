//-- Uintah framework includes --//
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>

//-- SpatialOps includes --//
#include "Operators/OperatorTypes.h"
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/FVStaggered.h>
#include <spatialops/structured/FVStaggeredBCTools.h>

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- Wasatch includes --//
#include "FieldTypes.h"
#include "GraphHelperTools.h"
#include "BCHelperTools.h"

namespace Wasatch {

  //-----------------------------------------------------------------------------
  
  /**
   *  @struct BCOpTypeSelector
   *
   *  @brief This templated struct is used to simplify boundary
   *         condition operator selection.
   */
  template< typename FieldT, typename BCEvaluator>
  struct BCOpTypeSelector
  {
  private:
    typedef OpTypes<FieldT> Ops;
    
  public:
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::InterpC2FX, BCEvaluator > DirichletX;
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::InterpC2FY, BCEvaluator > DirichletY;
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::InterpC2FZ, BCEvaluator > DirichletZ;
    
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::GradX, BCEvaluator > NeumannX;
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::GradY, BCEvaluator > NeumannY;
    typedef SpatialOps::structured::BoundaryConditionOp< typename Ops::GradZ, BCEvaluator > NeumannZ;
  };
  
  //-----------------------------------------------------------------------------

  template < typename FieldT, typename BCOpT >
  void setBoundaryCondition( const Uintah::Patch* const thePatch,
                             const GraphHelper& theGraph,
                             const std::string& phiName,
                             const SpatialOps::structured::IntVec& theBCPointIndex,
                             const SpatialOps::structured::IntVec& thePatchDimension, //send in the patch dimension to avoid calculating it for every point!
                             const bool bcx,
                             const bool bcy,
                             const bool bcz,
                             const SpatialOps::structured::BCSide theBCSide,
                             const double theBCValue,
                             const SpatialOps::OperatorDatabase& theOperatorsDb ) 
  {
    Expr::ExpressionFactory& theExprFactory = *theGraph.exprFactory;
    const Expr::Tag phiLabel( phiName, Expr::STATE_N );
    const Expr::ExpressionID phiID = theExprFactory.get_registry().get_id(phiLabel);  
    Expr::Expression<FieldT>& phiExpr = dynamic_cast<Expr::Expression<FieldT>&> ( theExprFactory.retrieve_expression( phiID, thePatch->getID() ) );
    //
    BCOpT theBCOperator(thePatchDimension, bcx, bcy, bcz, theBCPointIndex, theBCSide, SpatialOps::structured::ConstValEval(theBCValue), theOperatorsDb);
    phiExpr.process_after_evaluate(theBCOperator);
  }
  
  //-----------------------------------------------------------------------------
  
  template <typename T>
  bool getIteratorBCValueBCKind( const Uintah::Patch* patch, 
                                 const Uintah::Patch::FaceType face,
                                 const int child,
                                 const std::string& desc,
                                 const int mat_id,
                                 T& bc_value,
                                 SCIRun::Iterator& bound_ptr,
                                 std::string& bc_kind )
  {  
    SCIRun::Iterator nu;
    const Uintah::BoundCondBase* bc = patch->getArrayBCValues(face,mat_id,
                                                              desc, bound_ptr,
                                                              nu, child);
    const Uintah::BoundCond<T>* new_bcs;
    new_bcs =  dynamic_cast<const Uintah::BoundCond<T> *>(bc);
    
    bc_value=T(-9);
    bc_kind="NotSet";
    if (new_bcs != 0) {      // non-symmetric
      bc_value = new_bcs->getValue();
      bc_kind =  new_bcs->getBCType__NEW();
    }        
    delete bc;
    
    // Did I find an iterator
    if( bc_kind == "NotSet" ){
      return false;
    }else{
      return true;
    }
  }
  
  //-----------------------------------------------------------------------------
  
  void buildBoundaryConditions( const std::vector<EqnTimestepAdaptorBase*>& theEqnAdaptors, 
                                const GraphHelper& theGraphHelper,
                                const Uintah::PatchSet* const theLocalPatches,
                                const PatchInfoMap& thePatchInfoMap,
                                const Uintah::MaterialSubset* const theMaterials)
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
    
    for( EquationAdaptors::const_iterator ia=theEqnAdaptors.begin(); ia!=theEqnAdaptors.end(); ++ia ){
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
      for( int ip=0; ip<theLocalPatches->size(); ++ip ){
        
        // get the patch subset
        const Uintah::PatchSubset* const patches = theLocalPatches->getSubset(ip);
        
        // loop over every patch in the patch subset
        for( int ipss=0; ipss<patches->size(); ++ipss ){          
          
          // get a pointer to the current patch
          const Uintah::Patch* const patch = patches->get(ipss);          
          
          //
          const IntVector lo = patch->getCellLowIndex();
          const IntVector hi = patch->getCellHighIndex();
          const IntVector patchDim = hi - lo;
          const SS::IntVec thePatchDim(patchDim[0],patchDim[1],patchDim[2]);
          
          // loop over materials
          for( int im=0; im<theMaterials->size(); ++im ){
            
            const int materialID = theMaterials->get(im);
            
            // setup some info
            std::vector<Uintah::Patch::FaceType>::const_iterator faceIterator;
            std::vector<Uintah::Patch::FaceType> bndFaces;
            patch->getBoundaryFaces(bndFaces);
            
            // get plus face information
            const bool bcx = (*patch).getBCType(Uintah::Patch::xplus) != Uintah::Patch::Neighbor;
            const bool bcy = (*patch).getBCType(Uintah::Patch::yplus) != Uintah::Patch::Neighbor;
            const bool bcz = (*patch).getBCType(Uintah::Patch::zplus) != Uintah::Patch::Neighbor;
            
            // get the patch info from which we can get the operators database
            const PatchInfoMap::const_iterator ipi = thePatchInfoMap.find( patch->getID() );
            assert( ipi != thePatchInfoMap.end() );
            const SpatialOps::OperatorDatabase& theOperatorsDb= *(ipi->second.operators);
            
            // now loop over the boundary faces
            for (faceIterator = bndFaces.begin(); faceIterator !=bndFaces.end(); faceIterator++){
              Uintah::Patch::FaceType theFace = *faceIterator;
              
              //get the number of children
              int numChildren = patch->getBCDataArray(theFace)->getNumberChildren(materialID);
              
              for (int child = 0; child < numChildren; child++){
                //
                double bc_value = -9; 
                std::string bc_kind = "NotSet";
                SCIRun::Iterator bound_ptr;                
                bool foundIterator = getIteratorBCValueBCKind( patch, theFace, child, phiName, materialID, bc_value, bound_ptr, bc_kind);
                
                if (foundIterator) {
                  std::cout<<"SETTING BOUNDARY CONDITIONS\n";

                  if (bc_kind == "Dirichlet") {
                    switch (theFace) {
                        
                      case Uintah::Patch::xminus:
                        
                        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                          SCIRun::IntVector bc_point_indices(*bound_ptr); 
                          const SS::IntVec bcPointIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]);                          
                          //
                          if ( staggeredDirection=="X" ) { // X Volume Field
                            typedef SS::XVolField FieldT;
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletX >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::X_MINUS_SIDE, bc_value, theOperatorsDb);
                           
                          } else if (staggeredDirection=="Y") { // Y Volume Field
                            typedef SS::YVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletX >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::X_MINUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Z") { // Z Volume Field
                            typedef SS::ZVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletX >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::X_MINUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else { // Scalar Volume Field
                            typedef SS::SVolField  FieldT;            
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletX >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::X_MINUS_SIDE, bc_value, theOperatorsDb);
                          }                                                    
                          
                        }
                        break;
                        
                      case Uintah::Patch::xplus:
                        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                          SCIRun::IntVector bc_point_indices(*bound_ptr); 
                          const SS::IntVec bcPointIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]);
                          
                          if ( staggeredDirection=="X" ) { // X Volume Field
                            typedef SS::XVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletX >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::X_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Y") { // Y Volume Field
                            typedef SS::YVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletX >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::X_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Z") { // Z Volume Field
                            typedef SS::ZVolField  FieldT;     
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletX >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::X_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else { // Scalar Volume Field
                            typedef SS::SVolField  FieldT;
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletX >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::X_PLUS_SIDE, bc_value, theOperatorsDb);
                          }                                                    
                          
                          
                        }
                        break;
                        
                      case Uintah::Patch::yminus:
                        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                          SCIRun::IntVector bc_point_indices(*bound_ptr); 
                          const SS::IntVec bcPointIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]);
                          
                          if ( staggeredDirection=="X" ) { // X Volume Field
                            typedef SS::XVolField  FieldT;  
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletY >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Y_MINUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Y") { // Y Volume Field
                            typedef SS::YVolField  FieldT;                      
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletY >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Y_MINUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Z") { // Z Volume Field
                            typedef SS::ZVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletY >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Y_MINUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else { // Scalar Volume Field
                            typedef SS::SVolField  FieldT;       
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletY >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Y_MINUS_SIDE, bc_value, theOperatorsDb);
                          }                                                    
                        }
                        break;
                        
                      case Uintah::Patch::yplus:
                        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                          SCIRun::IntVector bc_point_indices(*bound_ptr); 
                          const SS::IntVec bcPointIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]);
                          
                          if ( staggeredDirection=="X" ) { // X Volume Field
                            typedef SS::XVolField  FieldT;  
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletY >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Y_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Y") { // Y Volume Field
                            typedef SS::YVolField  FieldT;                      
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletY >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Y_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Z") { // Z Volume Field
                            typedef SS::ZVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletY >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Y_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else { // Scalar Volume Field
                            typedef SS::SVolField  FieldT;       
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletY >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Y_PLUS_SIDE, bc_value, theOperatorsDb);
                          }                                                    
                          
                        }
                        break;
                        
                      case Uintah::Patch::zminus:
                        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                          SCIRun::IntVector bc_point_indices(*bound_ptr); 
                          const SS::IntVec bcPointIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]);
                          
                          if ( staggeredDirection=="X" ) { // X Volume Field
                            typedef SS::XVolField  FieldT;
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletZ >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Z_MINUS_SIDE, bc_value, theOperatorsDb);
                                                        
                          } else if (staggeredDirection=="Y") { // Y Volume Field
                            typedef SS::YVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletZ >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Z_MINUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Z") { // Z Volume Field
                            typedef SS::ZVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletZ >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Z_MINUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else { // Scalar Volume Field
                            typedef SS::SVolField  FieldT;                      
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletZ >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Z_MINUS_SIDE, bc_value, theOperatorsDb);
                          }                                                    
                          
                        }
                        break;
                        
                      case Uintah::Patch::zplus:
                        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                          SCIRun::IntVector bc_point_indices(*bound_ptr); 
                          const SS::IntVec bcPointIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]);
                          
                          if ( staggeredDirection=="X" ) { // X Volume Field
                            typedef SS::XVolField  FieldT;
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletZ >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Z_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Y") { // Y Volume Field
                            typedef SS::YVolField  FieldT;                      
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletZ >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Z_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Z") { // Z Volume Field
                            typedef SS::ZVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletZ >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Z_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else { // Scalar Volume Field
                            typedef SS::SVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::DirichletZ >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Z_PLUS_SIDE, bc_value, theOperatorsDb);
                          }                                                    
                          
                        }
                        break;
                        
                      case Uintah::Patch::numFaces:
                        throw Uintah::ProblemSetupException( "numFaces is not a valid face", __FILE__, __LINE__ );
                        break;
                        
                      case Uintah::Patch::invalidFace:
                        throw Uintah::ProblemSetupException( "invalidFace is not a valid face", __FILE__, __LINE__ );
                        break;
                    }
                    
                  } else if (bc_kind == "Neumann") {
                    
                    switch (theFace) {
                        
                      case Uintah::Patch::xminus:
                        
                        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                          SCIRun::IntVector bc_point_indices(*bound_ptr); 
                          const SS::IntVec bcPointIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]);
                          
                          //
                          if ( staggeredDirection=="X" ) { // X Volume Field
                            typedef SS::XVolField FieldT;
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannX >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::X_MINUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Y") { // Y Volume Field
                            typedef SS::YVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannX >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::X_MINUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Z") { // Z Volume Field
                            typedef SS::ZVolField  FieldT;                      
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannX >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::X_MINUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else { // Scalar Volume Field
                            typedef SS::SVolField  FieldT;            
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannX >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::X_MINUS_SIDE, bc_value, theOperatorsDb);
                          }                                                    
                          
                        }
                        break;
                        
                      case Uintah::Patch::xplus:
                        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                          SCIRun::IntVector bc_point_indices(*bound_ptr); 
                          const SS::IntVec bcPointIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]);
                          
                          if ( staggeredDirection=="X" ) { // X Volume Field
                            typedef SS::XVolField  FieldT; 
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannX >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::X_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Y") { // Y Volume Field
                            typedef SS::YVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannX >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::X_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Z") { // Z Volume Field
                            typedef SS::ZVolField  FieldT;     
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannX >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::X_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else { // Scalar Volume Field
                            typedef SS::SVolField  FieldT;
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannX >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::X_PLUS_SIDE, bc_value, theOperatorsDb);
                          }                                                    
                          
                          
                        }
                        break;
                        
                      case Uintah::Patch::yminus:
                        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                          SCIRun::IntVector bc_point_indices(*bound_ptr); 
                          const SS::IntVec bcPointIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]);
                          
                          if ( staggeredDirection=="X" ) { // X Volume Field
                            typedef SS::XVolField  FieldT;  
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannY >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Y_MINUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Y") { // Y Volume Field
                            typedef SS::YVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannY >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Y_MINUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Z") { // Z Volume Field
                            typedef SS::ZVolField  FieldT;                      
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannY >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Y_MINUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else { // Scalar Volume Field
                            typedef SS::SVolField  FieldT;       
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannY >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Y_MINUS_SIDE, bc_value, theOperatorsDb);
                          }                                                    
                        }
                        break;
                        
                      case Uintah::Patch::yplus:
                        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                          SCIRun::IntVector bc_point_indices(*bound_ptr); 
                          const SS::IntVec bcPointIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]);
                          
                          if ( staggeredDirection=="X" ) { // X Volume Field
                            typedef SS::XVolField  FieldT;  
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannY >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Y_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Y") { // Y Volume Field
                            typedef SS::YVolField  FieldT;    
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannY >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Y_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Z") { // Z Volume Field
                            typedef SS::ZVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannY >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Y_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else { // Scalar Volume Field
                            typedef SS::SVolField  FieldT;       
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannY >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Y_PLUS_SIDE, bc_value, theOperatorsDb);
                          }                                                    
                          
                        }
                        break;
                        
                      case Uintah::Patch::zminus:
                        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                          SCIRun::IntVector bc_point_indices(*bound_ptr); 
                          const SS::IntVec bcPointIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]);
                          
                          if ( staggeredDirection=="X" ) { // X Volume Field
                            typedef SS::XVolField  FieldT;
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannZ >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Z_MINUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Y") { // Y Volume Field
                            typedef SS::YVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannZ >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Z_MINUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Z") { // Z Volume Field
                            typedef SS::ZVolField  FieldT;                      
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannZ >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Z_MINUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else { // Scalar Volume Field
                            typedef SS::SVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannZ >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Z_MINUS_SIDE, bc_value, theOperatorsDb);
                          }                                                    
                          
                        }
                        break;
                        
                      case Uintah::Patch::zplus:
                        for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
                          SCIRun::IntVector bc_point_indices(*bound_ptr); 
                          const SS::IntVec bcPointIJK(bc_point_indices[0],bc_point_indices[1],bc_point_indices[2]);
                          
                          if ( staggeredDirection=="X" ) { // X Volume Field
                            typedef SS::XVolField  FieldT;
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannZ >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Z_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Y") { // Y Volume Field
                            typedef SS::YVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannZ >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Z_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else if (staggeredDirection=="Z") { // Z Volume Field
                            typedef SS::ZVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannZ >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Z_PLUS_SIDE, bc_value, theOperatorsDb);
                            
                          } else { // Scalar Volume Field
                            typedef SS::SVolField  FieldT;                          
                            setBoundaryCondition< FieldT, BCOpTypeSelector<FieldT,BCEvaluator>::NeumannZ >(patch, theGraphHelper, phiName, bcPointIJK, thePatchDim, bcx,bcy,bcz,SS::Z_PLUS_SIDE, bc_value, theOperatorsDb);
                          }                                                    
                          
                        }
                        break;
                        
                      case Uintah::Patch::numFaces:
                        throw Uintah::ProblemSetupException( "numFaces is not a valid face", __FILE__, __LINE__ );
                        break;
                        
                      case Uintah::Patch::invalidFace:
                        throw Uintah::ProblemSetupException( "invalidFace is not a valid face", __FILE__, __LINE__ );
                        break;
                    }                    
                  }
                }
              }
            }
          }
        }  
      }
    }
  }
  // ------------------------
} // namespace wasatch

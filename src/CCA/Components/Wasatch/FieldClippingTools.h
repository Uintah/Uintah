/*
 * The MIT License
 *
 * Copyright (c) 2011-2012 The University of Utah
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

#ifndef FieldClippingTools_h
#define FieldClippingTools_h

#include <spatialops/SpatialOpsConfigure.h>
#include <spatialops/structured/FVTools.h>
#include <expression/Expression.h>
#include "ParseTools.h"

namespace Wasatch{      
  
  /**
   *  @class  MinMaxClip
   *  @author Tony Saad
   *  @date   July, 2012
   *
   *  @brief Sets a Min-Max clip on a field.
   *
   *  @par Template Parameters
   *  <ul>
   *   <li> \b FieldT The type of field to set the clip on.
   
   *  </ul>
   *
   */
  template< typename FieldT >
  class MinMaxClip
  {
    const double min_;
    const double max_;
    
  public:
    
    /**
     *  @param min The minimum acceptable value for this field.
     *
     *  @param max The maximum acceptable value for this field.
     *     
     */
    MinMaxClip( const double min,
               const double max );
    
    ~MinMaxClip(){}
    
    /**
     *  Applies the clipping.
     *
     *  @param f The field that we want to apply the clip on.
     */
    inline void operator()( FieldT& f ) const;
  };
  
  
  
  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  //
  //                         Implementation
  //
  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
  template< typename FieldT >
  MinMaxClip<FieldT>::
  MinMaxClip( const double min, const double max )
  : min_( min ),
    max_( max )
  {}
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  void
  MinMaxClip<FieldT>::
  operator()( FieldT& f ) const
  {
    using namespace SpatialOps;
    f <<= cond( f < min_, min_ )
              ( f > max_, max_ )
              ( f );    
  }

  //------------------------------------------------------------------

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  //
  //                   Input Processing for Clipping
  //
  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  template<typename FieldT>
  void
  clip_expr(const Uintah::PatchSet* const localPatches, 
            const GraphHelper& graphHelper,
            const Expr::Tag& fieldTag,
            const double min,
            const double max) 
  {
    Expr::ExpressionFactory& factory = *graphHelper.exprFactory;
    
    for( int ip=0; ip<localPatches->size(); ++ip ){
      // get the patch subset
      const Uintah::PatchSubset* const patches = localPatches->getSubset(ip);
      
      // loop over every patch in the patch subset
      for( int ipss=0; ipss<patches->size(); ++ipss ){
        
        // get a pointer to the current patch
        const Uintah::Patch* const patch = patches->get(ipss);

        Expr::Expression<FieldT>& phiExpr = dynamic_cast<Expr::Expression<FieldT>&>( factory.retrieve_expression( fieldTag, patch->getID(), false ) );
        MinMaxClip<FieldT> clipper(min,max);
        phiExpr.process_after_evaluate(fieldTag.name(),clipper);    
      }
    }
  }
  
  
  void
  process_field_clipping( Uintah::ProblemSpecP parser,
                         GraphCategories& gc,
                         const Uintah::PatchSet* const localPatches)
  {
    //___________________________________
    // parse and clip expressions
    for( Uintah::ProblemSpecP clipParams = parser->findBlock("FieldClipping");
        clipParams != 0;
        clipParams = clipParams->findNextBlock("FieldClipping") ){
      
      std::string fieldType, taskListName;      
      clipParams->getAttribute("tasklist", taskListName);
      
      Category cat = INITIALIZATION;
      if     ( taskListName == "initialization"   )   cat = INITIALIZATION;
      else if( taskListName == "advance_solution" )   cat = ADVANCE_SOLUTION;
      else{
        std::ostringstream msg;
        msg << "ERROR: unsupported task list specified in FieldClipping block '" << taskListName << "'" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      
      GraphHelper* const graphHelper = gc[cat];
      
      for( Uintah::ProblemSpecP fieldParams = clipParams->findBlock("FieldExpression");
          fieldParams != 0;
          fieldParams = fieldParams->findNextBlock("FieldExpression") ){
        double min, max;
        
        fieldParams->getAttribute("type", fieldType);
        fieldParams->getAttribute("min", min);
        fieldParams->getAttribute("max",max);
        const Expr::Tag fieldTag = parse_nametag( fieldParams->findBlock("NameTag") );
        
        switch( get_field_type(fieldType) ){
          case SVOL : clip_expr< SVolField >( localPatches,*graphHelper, fieldTag, min, max );  break;
          case XVOL : clip_expr< XVolField >( localPatches,*graphHelper, fieldTag, min, max );  break;
          case YVOL : clip_expr< YVolField >( localPatches,*graphHelper, fieldTag, min, max );  break;
          case ZVOL : clip_expr< ZVolField >( localPatches,*graphHelper, fieldTag, min, max );  break;
          default:
            std::ostringstream msg;
            msg << "ERROR: unsupported field type '" << fieldType << "'" << std::endl;
            throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        
      }              
    }        
  }  
} // namespace Wasatch


#endif  // FieldClippingTools_h

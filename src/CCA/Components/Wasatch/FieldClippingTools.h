/*
 * The MIT License
 *
 * Copyright (c) 2011-2017 The University of Utah
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
#include <expression/Expression.h>
#include "ParseTools.h"

namespace WasatchCore{      
  
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
  : public Expr::Expression<FieldT>
  {
  private:
    const double min_, max_;
    const bool hasVolFrac_;
    DECLARE_FIELD(FieldT, volFrac_)
    
  public:
    MinMaxClip( const Expr::Tag& volFracTag,
                const double min,
                const double max )
    : Expr::Expression<FieldT>(),
      min_(min),
      max_(max),
      hasVolFrac_( volFracTag != Expr::Tag() )
    {
      if( hasVolFrac_ )  volFrac_ = this->template create_field_request<FieldT>(volFracTag);
    }
    
    class Builder : public Expr::ExpressionBuilder
    {
    public:
      /**
       * @param result Tag of the resulting expression.
       * @param bcValue   constant boundary condition value.
       * @param cghost ghost coefficient. This is usually provided by an operator.
       * @param flatGhostPoints  flat indices of the ghost points in which BC is being set.
       * @param cinterior interior coefficient. This is usually provided by an operator.
       * @param flatInteriorPoints  flat indices of the interior points that are used to set the ghost value.
       */
      Builder( const Expr::Tag& resultTag,
               const Expr::Tag& volFracTag,
               const double min,
               const double max )
        : ExpressionBuilder(resultTag),
          volFracTag_(volFracTag),
          min_(min),
          max_(max)
      {}
      Expr::ExpressionBase* build() const{ return new MinMaxClip(volFracTag_, min_, max_); }
    private:
      const Expr::Tag volFracTag_;
      const double min_, max_;
    };
    
    ~MinMaxClip(){}
    void evaluate();
  };
  
  
  
  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  //
  //                         Implementation
  //
  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  void MinMaxClip<FieldT>::evaluate()
  {
    using namespace SpatialOps;
    FieldT& f = this->value();
    if( hasVolFrac_ ){
      const FieldT& volFrac = volFrac_->field_ref();
      f <<= volFrac * max( min( f, max_ ), min_ );
    }
    else{
      f <<= max( min( f, max_ ), min_ );
    }
  }


  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  //
  //                   Input Processing for Clipping
  //
  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  template<typename FieldT>
  void
  clip_expr( const Uintah::PatchSet* const localPatches,
             GraphHelper& graphHelper,
             const Category& cat,
             const Expr::Tag& fieldTag,
             const double min,
             const double max,
             const Expr::Tag& volFracTag )
  {
    Expr::ExpressionFactory& factory = *graphHelper.exprFactory;
    
    if( cat == POSTPROCESSING ){
      const Expr::ExpressionID expID = factory.get_id(fieldTag);
      graphHelper.rootIDs.insert(expID);
    }
    
    for( int ip=0; ip<localPatches->size(); ++ip ){
      // get the patch subset
      const Uintah::PatchSubset* const patches = localPatches->getSubset(ip);
      // loop over every patch in the patch subset
      for( int ipss=0; ipss<patches->size(); ++ipss ){
        // get a pointer to the current patch
        const Uintah::Patch* const patch = patches->get(ipss);
        const std::string strPatchID = number_to_string(patch->getID());
        const Expr::Tag modTag( fieldTag.name() + "_clipper_patch_" + strPatchID , Expr::STATE_NONE );
        Expr::ExpressionBuilder* builder = new typename MinMaxClip<FieldT>::Builder( modTag, volFracTag, min, max );
        factory.register_expression( builder, true );
        factory.attach_modifier_expression(modTag, fieldTag, patch->getID(), true);
      }
    }
  }
  
  
  void
  process_field_clipping( Uintah::ProblemSpecP parser,
                          GraphCategories& gc,
                          const Uintah::PatchSet* const localPatches )
  {
    Expr::Tag svolFracTag;
    Expr::Tag xvolFracTag;
    Expr::Tag yvolFracTag;
    Expr::Tag zvolFracTag;
    const bool hasVolFrac = parser->findBlock("EmbeddedGeometry");
    const EmbeddedGeometryHelper& vNames = EmbeddedGeometryHelper::self();
    if( hasVolFrac ){
      svolFracTag = vNames.vol_frac_tag<SVolField>();
      xvolFracTag = vNames.vol_frac_tag<XVolField>();
      yvolFracTag = vNames.vol_frac_tag<YVolField>();
      zvolFracTag = vNames.vol_frac_tag<ZVolField>();
    }

    //___________________________________
    // parse and clip expressions
    for( Uintah::ProblemSpecP clipParams = parser->findBlock("FieldClipping");
        clipParams != nullptr;
        clipParams = clipParams->findNextBlock("FieldClipping") ){
      
      std::string fieldType, taskListName;      
      clipParams->getAttribute("tasklist", taskListName);
      
      Category cat = INITIALIZATION;
      if      ( taskListName == "initialization"   )   cat = INITIALIZATION;
      else if ( taskListName == "advance_solution" )   cat = ADVANCE_SOLUTION;
      else if ( taskListName == "post_processing"  )   cat = POSTPROCESSING;
      else{
        std::ostringstream msg;
        msg << "ERROR: unsupported task list specified in FieldClipping block '" << taskListName << "'" << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
      
      GraphHelper* const graphHelper = gc[cat];
      
      for( Uintah::ProblemSpecP fieldParams = clipParams->findBlock("FieldExpression");
          fieldParams != nullptr;
          fieldParams = fieldParams->findNextBlock("FieldExpression") ){
        double min, max;
        
        fieldParams->getAttribute("type", fieldType);
        fieldParams->getAttribute("min", min);
        fieldParams->getAttribute("max",max);
        const Expr::Tag fieldTag = parse_nametag( fieldParams->findBlock("NameTag") );
        
        switch( get_field_type(fieldType) ){
          case SVOL : clip_expr< SVolField >( localPatches, *graphHelper, cat, fieldTag, min, max, svolFracTag );  break;
          case XVOL : clip_expr< XVolField >( localPatches, *graphHelper, cat, fieldTag, min, max, xvolFracTag );  break;
          case YVOL : clip_expr< YVolField >( localPatches, *graphHelper, cat, fieldTag, min, max, yvolFracTag );  break;
          case ZVOL : clip_expr< ZVolField >( localPatches, *graphHelper, cat, fieldTag, min, max, zvolFracTag );  break;
          default:
            std::ostringstream msg;
            msg << "ERROR: unsupported field type '" << fieldType << "'" << std::endl;
            throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
        }
        
      }              
    }        
  }  
} // namespace WasatchCore


#endif  // FieldClippingTools_h

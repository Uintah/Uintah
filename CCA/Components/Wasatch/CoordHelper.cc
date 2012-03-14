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

//-- Wasatch includes --//
#include "CoordHelper.h"
#include "Expressions/Coordinate.h"
#include "StringNames.h"

//-- Uintah includes --//
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Task.h>
#include <Core/Parallel/ProcessorGroup.h>

#include "StringNames.h"

//-- ExprLib includes --//
#include <expression/ExpressionFactory.h>


namespace Wasatch{

  CoordHelper::CoordHelper( Expr::ExpressionFactory& exprFactory )
    : context_( Expr::STATE_NONE ),
      sName_( StringNames::self() ),
      xsvt_( sName_.xsvolcoord, context_ ),  ysvt_( sName_.ysvolcoord, context_ ),  zsvt_( sName_.zsvolcoord, context_ ),
      xxvt_( sName_.xxvolcoord, context_ ),  yxvt_( sName_.yxvolcoord, context_ ),  zxvt_( sName_.zxvolcoord, context_ ),
      xyvt_( sName_.xyvolcoord, context_ ),  yyvt_( sName_.yyvolcoord, context_ ),  zyvt_( sName_.zyvolcoord, context_ ),
      xzvt_( sName_.xzvolcoord, context_ ),  yzvt_( sName_.yzvolcoord, context_ ),  zzvt_( sName_.zzvolcoord, context_ )
  {
    needCoords_ = false;

    xSVolCoord_ = ySVolCoord_ = zSVolCoord_ = false;
    xXVolCoord_ = yXVolCoord_ = zXVolCoord_ = false;
    xYVolCoord_ = yYVolCoord_ = zYVolCoord_ = false;
    xZVolCoord_ = yZVolCoord_ = zZVolCoord_ = false;

    hasSetVarlabels_ = false;

    //_____________________________________________________________
    // build expressions to set coordinates.  If any initialization
    // expressions require the coordinates, then this will trigger
    // their construction and incorporation into a graph.
    exprFactory.register_expression( scinew Coordinate<SVolField>::Builder(xsvt_,*this,XDIR) );
    exprFactory.register_expression( scinew Coordinate<SVolField>::Builder(ysvt_,*this,YDIR) );
    exprFactory.register_expression( scinew Coordinate<SVolField>::Builder(zsvt_,*this,ZDIR) );

    exprFactory.register_expression( scinew Coordinate<XVolField>::Builder(xxvt_,*this,XDIR) );
    exprFactory.register_expression( scinew Coordinate<XVolField>::Builder(yxvt_,*this,YDIR) );
    exprFactory.register_expression( scinew Coordinate<XVolField>::Builder(zxvt_,*this,ZDIR) );

    exprFactory.register_expression( scinew Coordinate<YVolField>::Builder(xyvt_,*this,XDIR) );
    exprFactory.register_expression( scinew Coordinate<YVolField>::Builder(yyvt_,*this,YDIR) );
    exprFactory.register_expression( scinew Coordinate<YVolField>::Builder(zyvt_,*this,ZDIR) );

    exprFactory.register_expression( scinew Coordinate<ZVolField>::Builder(xzvt_,*this,XDIR) );
    exprFactory.register_expression( scinew Coordinate<ZVolField>::Builder(yzvt_,*this,YDIR) );
    exprFactory.register_expression( scinew Coordinate<ZVolField>::Builder(zzvt_,*this,ZDIR) );
  }

  //------------------------------------------------------------------

  CoordHelper::~CoordHelper()
  {
    // wipe out VarLabels
    if( xSVolCoord_ ) Uintah::VarLabel::destroy(xSVol_);
    if( ySVolCoord_ ) Uintah::VarLabel::destroy(ySVol_);
    if( zSVolCoord_ ) Uintah::VarLabel::destroy(zSVol_);

    if( xXVolCoord_ ) Uintah::VarLabel::destroy(xXVol_);
    if( yXVolCoord_ ) Uintah::VarLabel::destroy(yXVol_);
    if( zXVolCoord_ ) Uintah::VarLabel::destroy(zXVol_);

    if( xYVolCoord_ ) Uintah::VarLabel::destroy(xYVol_);
    if( yYVolCoord_ ) Uintah::VarLabel::destroy(yYVol_);
    if( zYVolCoord_ ) Uintah::VarLabel::destroy(zYVol_);

    if( xZVolCoord_ ) Uintah::VarLabel::destroy(xZVol_);
    if( yZVolCoord_ ) Uintah::VarLabel::destroy(yZVol_);
    if( zZVolCoord_ ) Uintah::VarLabel::destroy(zZVol_);
  }

  //------------------------------------------------------------------

  void
  CoordHelper::create_task( Uintah::SchedulerP& sched,
                            const Uintah::PatchSet* patches,
                            const Uintah::MaterialSet* materials )
  {
    // at this point, if we needed coordinate information we will
    // have called back to set that fact. Schedule the coordinate
    // calculation prior to the initialization task.
    if( needCoords_ ){
      Uintah::Task* task = scinew Uintah::Task( "set coordinates", this, &CoordHelper::set_grid_variables );
      register_coord_fields( *task, *patches, *materials );
      sched->addTask( task, patches, materials );
    }
  }

  //------------------------------------------------------------------

  void
  CoordHelper::register_coord_fields( Uintah::Task& task,
                                      const Uintah::PatchSet& patches,
                                      const Uintah::MaterialSet& materials )
  {
    const Uintah::MaterialSubset* const mss = materials.getUnion();
    const Uintah::PatchSubset* const pss = patches.getUnion();

    if( xSVolCoord_ ) reg_field<SVolField>( xSVol_, xsvt_, task, pss, mss );
    if( ySVolCoord_ ) reg_field<SVolField>( ySVol_, ysvt_, task, pss, mss );
    if( zSVolCoord_ ) reg_field<SVolField>( zSVol_, zsvt_, task, pss, mss );

    if( xXVolCoord_ ) reg_field<XVolField>( xXVol_, xxvt_, task, pss, mss );
    if( yXVolCoord_ ) reg_field<XVolField>( yXVol_, yxvt_, task, pss, mss );
    if( zXVolCoord_ ) reg_field<XVolField>( zXVol_, zxvt_, task, pss, mss );

    if( xYVolCoord_ ) reg_field<YVolField>( xYVol_, xyvt_, task, pss, mss );
    if( yYVolCoord_ ) reg_field<YVolField>( yYVol_, yyvt_, task, pss, mss );
    if( zYVolCoord_ ) reg_field<YVolField>( zYVol_, zyvt_, task, pss, mss );

    if( xZVolCoord_ ) reg_field<ZVolField>( xZVol_, xzvt_, task, pss, mss );
    if( yZVolCoord_ ) reg_field<ZVolField>( yZVol_, yzvt_, task, pss, mss );
    if( zZVolCoord_ ) reg_field<ZVolField>( zZVol_, zzvt_, task, pss, mss );

    hasSetVarlabels_ = true;
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  void set_coord( const Uintah::VarLabel* varLabel,
                  Uintah::DataWarehouse* const dw,
                  const Uintah::Patch* const patch,
                  const int material,
                  const SCIRun::Vector& shift,
                  const int idir )
  {
    typename SelectUintahFieldType<FieldT>::type field;
    dw->allocateAndPut( field, varLabel, material, patch, get_uintah_ghost_type<FieldT>(), get_n_ghost<FieldT>() );
    const IntVector lo = field.getLow();
    const IntVector hi = field.getHigh();
    for( int k=lo[2]; k<hi[2]; ++k ){
      for( int j=lo[1]; j<hi[1]; ++j ){
        for( int i=lo[0]; i<hi[0]; ++i ){
          const IntVector index(i,j,k);
          const SCIRun::Vector xyz = patch->getCellPosition(index).vector();
          field[index] = xyz[idir] + shift[idir];  // jcs note that this is inefficient.
        }
      }
    }
  }

  //------------------------------------------------------------------

  void
  CoordHelper::set_grid_variables( const Uintah::ProcessorGroup* const pg,
                                   const Uintah::PatchSubset* const patches,
                                   const Uintah::MaterialSubset* const materials,
                                   Uintah::DataWarehouse* const oldDW,
                                   Uintah::DataWarehouse* const newDW )
  {
    for( int ip=0; ip<patches->size(); ++ip ){
      const Uintah::Patch* const patch = patches->get(ip);

      for( int im=0; im<materials->size(); ++im ){
        const int material = materials->get(im);

        const SCIRun::Vector spacing = patch->dCell();
        SCIRun::Vector shift( 0.0, 0.0, 0.0 );

        if( xSVolCoord_ ) set_coord<SVolField>( xSVol_, newDW, patch, material, shift, 0 );
        if( ySVolCoord_ ) set_coord<SVolField>( ySVol_, newDW, patch, material, shift, 1 );
        if( zSVolCoord_ ) set_coord<SVolField>( zSVol_, newDW, patch, material, shift, 2 );

        shift[0] = -spacing[0]*0.5;  // shift x by -dx/2
        if( xXVolCoord_ ) set_coord<XVolField>( xXVol_, newDW, patch, material, shift, 0 );
        if( yXVolCoord_ ) set_coord<XVolField>( yXVol_, newDW, patch, material, shift, 1 );
        if( zXVolCoord_ ) set_coord<XVolField>( zXVol_, newDW, patch, material, shift, 2 );

        shift[0] = 0;
        shift[1] = -spacing[1]*0.5;
        if( xYVolCoord_ ) set_coord<YVolField>( xYVol_, newDW, patch, material, shift, 0 );
        if( yYVolCoord_ ) set_coord<YVolField>( yYVol_, newDW, patch, material, shift, 1 );
        if( zYVolCoord_ ) set_coord<YVolField>( zYVol_, newDW, patch, material, shift, 2 );

        shift[1] = 0;
        shift[2] = -spacing[2]*0.5;
        if( xZVolCoord_ ) set_coord<ZVolField>( xZVol_, newDW, patch, material, shift, 0 );
        if( yZVolCoord_ ) set_coord<ZVolField>( yZVol_, newDW, patch, material, shift, 1 );
        if( zZVolCoord_ ) set_coord<ZVolField>( zZVol_, newDW, patch, material, shift, 2 );

      }  // material loop
    } // patch loop
  }

} // namespace Wasatch

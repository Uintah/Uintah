//-- Wasatch includes --//
#include "CoordHelper.h"
#include "Expressions/Coordinate.h"

//-- Uintah includes --//
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Task.h>
#include <Core/Parallel/ProcessorGroup.h>

//-- ExprLib includes --//
#include <expression/ExpressionFactory.h>


namespace Wasatch{

  CoordHelper::CoordHelper( Expr::ExpressionFactory& exprFactory )
  {
    //_____________________________________________________________
    // build expressions to set coordinates.  If any initialization
    // expressions require the coordinates, then this will trigger
    // their construction and incorporation into a graph.
    exprFactory.register_expression( Expr::Tag("XSVOL",Expr::STATE_NONE), scinew Coordinate<SVolField>::Builder(*this,XDIR) );
    exprFactory.register_expression( Expr::Tag("YSVOL",Expr::STATE_NONE), scinew Coordinate<SVolField>::Builder(*this,YDIR) );
    exprFactory.register_expression( Expr::Tag("ZSVOL",Expr::STATE_NONE), scinew Coordinate<SVolField>::Builder(*this,ZDIR) );

    exprFactory.register_expression( Expr::Tag("XXVOL",Expr::STATE_NONE), scinew Coordinate<XVolField>::Builder(*this,XDIR) );
    exprFactory.register_expression( Expr::Tag("YXVOL",Expr::STATE_NONE), scinew Coordinate<XVolField>::Builder(*this,YDIR) );
    exprFactory.register_expression( Expr::Tag("ZXVOL",Expr::STATE_NONE), scinew Coordinate<XVolField>::Builder(*this,ZDIR) );

    exprFactory.register_expression( Expr::Tag("XYVOL",Expr::STATE_NONE), scinew Coordinate<YVolField>::Builder(*this,XDIR) );
    exprFactory.register_expression( Expr::Tag("YYVOL",Expr::STATE_NONE), scinew Coordinate<YVolField>::Builder(*this,YDIR) );
    exprFactory.register_expression( Expr::Tag("ZYVOL",Expr::STATE_NONE), scinew Coordinate<YVolField>::Builder(*this,ZDIR) );

    exprFactory.register_expression( Expr::Tag("XZVOL",Expr::STATE_NONE), scinew Coordinate<ZVolField>::Builder(*this,XDIR) );
    exprFactory.register_expression( Expr::Tag("YZVOL",Expr::STATE_NONE), scinew Coordinate<ZVolField>::Builder(*this,YDIR) );
    exprFactory.register_expression( Expr::Tag("ZZVOL",Expr::STATE_NONE), scinew Coordinate<ZVolField>::Builder(*this,ZDIR) );
  }

  //------------------------------------------------------------------

  CoordHelper::~CoordHelper()
  {
    // wipe out VarLabels
    Uintah::VarLabel::destroy(xSVol_);  Uintah::VarLabel::destroy(ySVol_);  Uintah::VarLabel::destroy(zSVol_);
    Uintah::VarLabel::destroy(xXVol_);  Uintah::VarLabel::destroy(yXVol_);  Uintah::VarLabel::destroy(zXVol_);
    Uintah::VarLabel::destroy(xYVol_);  Uintah::VarLabel::destroy(yYVol_);  Uintah::VarLabel::destroy(zYVol_);
    Uintah::VarLabel::destroy(xZVol_);  Uintah::VarLabel::destroy(yZVol_);  Uintah::VarLabel::destroy(zZVol_);
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
      Uintah::Task* task = scinew Uintah::Task( "coordinates", this, &CoordHelper::set_grid_variables );
      register_coord_fields( task, patches, materials );
      sched->addTask( task, patches, materials );
    }
  }

  //------------------------------------------------------------------

  void
  CoordHelper::register_coord_fields( Uintah::Task* const task,
                                      const Uintah::PatchSet* const ps,
                                      const Uintah::MaterialSet* const ms )
  {
    const Uintah::PatchSubset*    const pss = ps->getUnion();
    const Uintah::MaterialSubset* const mss = ms->getUnion();

    const Uintah::Task::DomainSpec domain = Uintah::Task::NormalDomain;

    if( xSVolCoord_ ){
      xSVol_=Uintah::VarLabel::create("XSVOL", getUintahFieldTypeDescriptor<SVolField>(), getUintahGhostDescriptor<SVolField>() );
      task->computes( xSVol_, pss, domain, mss, domain );
    }
    if( ySVolCoord_ ){
      ySVol_=Uintah::VarLabel::create("YSVOL", getUintahFieldTypeDescriptor<SVolField>(), getUintahGhostDescriptor<SVolField>() );
      task->computes( ySVol_, pss, domain, mss, domain );
    }
    if( zSVolCoord_ ){
      zSVol_=Uintah::VarLabel::create("ZSVOL", getUintahFieldTypeDescriptor<SVolField>(), getUintahGhostDescriptor<SVolField>() );
      task->computes( zSVol_, pss, domain, mss, domain );
    }

    if( xXVolCoord_ ){
      xXVol_=Uintah::VarLabel::create("XXVOL", getUintahFieldTypeDescriptor<XVolField>(), getUintahGhostDescriptor<XVolField>() );
      task->computes( xXVol_, pss, domain, mss, domain );
    }
    if( yXVolCoord_ ){
      yXVol_=Uintah::VarLabel::create("YXVOL", getUintahFieldTypeDescriptor<XVolField>(), getUintahGhostDescriptor<XVolField>() );
      task->computes( yXVol_, pss, domain, mss, domain );
    }
    if( zXVolCoord_ ){
      zSVol_=Uintah::VarLabel::create("ZXVOL", getUintahFieldTypeDescriptor<XVolField>(), getUintahGhostDescriptor<XVolField>() );
      task->computes( zXVol_, pss, domain, mss, domain );
    }

    if( xYVolCoord_ ){
      xYVol_=Uintah::VarLabel::create("XYVOL", getUintahFieldTypeDescriptor<YVolField>(), getUintahGhostDescriptor<YVolField>() );
      task->computes( xYVol_, pss, domain, mss, domain );
    }
    if( yYVolCoord_ ){
      yYVol_=Uintah::VarLabel::create("YYVOL", getUintahFieldTypeDescriptor<YVolField>(), getUintahGhostDescriptor<YVolField>() );
      task->computes( yYVol_, pss, domain, mss, domain );
    }
    if( zYVolCoord_ ){
      zYVol_=Uintah::VarLabel::create("ZYVOL", getUintahFieldTypeDescriptor<YVolField>(), getUintahGhostDescriptor<YVolField>() );
      task->computes( zYVol_, pss, domain, mss, domain );
    }

    if( xZVolCoord_ ){
      xZVol_=Uintah::VarLabel::create("XZVOL", getUintahFieldTypeDescriptor<ZVolField>(), getUintahGhostDescriptor<ZVolField>() );
      task->computes( xZVol_, pss, domain, mss, domain );
    }
    if( yZVolCoord_ ){
      yZVol_=Uintah::VarLabel::create("YZVOL", getUintahFieldTypeDescriptor<ZVolField>(), getUintahGhostDescriptor<ZVolField>() );
      task->computes( yZVol_, pss, domain, mss, domain );
    }
    if( zZVolCoord_ ){
      zZVol_=Uintah::VarLabel::create("ZZVOL", getUintahFieldTypeDescriptor<ZVolField>(), getUintahGhostDescriptor<ZVolField>() );
      task->computes( zZVol_, pss, domain, mss, domain );
    }
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  void set_coord( FieldT& field, const Uintah::Patch* p, const double shift, const int idir )
  {
    const IntVector lo = field.getLow();
    const IntVector hi = field.getHigh();
    for( int k=lo[2]; k<hi[2]; ++k ){
      for( int j=lo[1]; j<hi[1]; ++j ){
        for( int i=lo[0]; i<hi[0]; ++i ){
          const IntVector index(i,j,k);
          const SCIRun::Vector xyz = p->getCellPosition(index).vector();
          field[index] = xyz[idir] + shift;
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

        double shift = 0.0;
        if( xSVolCoord_ ){
          SelectUintahFieldType<SVolField>::type field;
          newDW->allocateAndPut( field, xSVol_, material, patch, getUintahGhostType<SVolField>(), getNGhost<SVolField>() );
          set_coord( field, patch, shift, 0 );
        }
        if( ySVolCoord_ ){
          SelectUintahFieldType<SVolField>::type field;
          newDW->allocateAndPut( field, ySVol_, material, patch, getUintahGhostType<SVolField>(), getNGhost<SVolField>() );
          set_coord( field, patch, shift, 0 );
        }
        if( zSVolCoord_ ){
          SelectUintahFieldType<SVolField>::type field;
          newDW->allocateAndPut( field, zSVol_, material, patch, getUintahGhostType<SVolField>(), getNGhost<SVolField>() );
          set_coord( field, patch, shift, 0 );
        }

        shift = -spacing[0]*0.5;  // shift x by -dx/2
        if( xXVolCoord_ ){
          SelectUintahFieldType<XVolField>::type field;
          newDW->allocateAndPut( field, xXVol_, material, patch, getUintahGhostType<XVolField>(), getNGhost<XVolField>() );
          set_coord( field, patch, shift, 0 );
        }
        if( yXVolCoord_ ){
          SelectUintahFieldType<XVolField>::type field;
          newDW->allocateAndPut( field, yXVol_, material, patch, getUintahGhostType<XVolField>(), getNGhost<XVolField>() );
          set_coord( field, patch, shift, 0 );
        }
        if( zXVolCoord_ ){
          SelectUintahFieldType<XVolField>::type field;
          newDW->allocateAndPut( field, zXVol_, material, patch, getUintahGhostType<XVolField>(), getNGhost<XVolField>() );
          set_coord( field, patch, shift, 0 );
        }

        shift = -spacing[1]*0.5;
        if( xYVolCoord_ ){
          SelectUintahFieldType<YVolField>::type field;
          newDW->allocateAndPut( field, xYVol_, material, patch, getUintahGhostType<YVolField>(), getNGhost<YVolField>() );
          set_coord( field, patch, shift, 1 );
        }
        if( yYVolCoord_ ){
          SelectUintahFieldType<YVolField>::type field;
          newDW->allocateAndPut( field, yYVol_, material, patch, getUintahGhostType<YVolField>(), getNGhost<YVolField>() );
          set_coord( field, patch, shift, 1 );
        }
        if( zYVolCoord_ ){
          SelectUintahFieldType<YVolField>::type field;
          newDW->allocateAndPut( field, zYVol_, material, patch, getUintahGhostType<YVolField>(), getNGhost<YVolField>() );
          set_coord( field, patch, shift, 1 );
        }

        shift = -spacing[1]*0.5;
        if( xYVolCoord_ ){
          SelectUintahFieldType<ZVolField>::type field;
          newDW->allocateAndPut( field, xZVol_, material, patch, getUintahGhostType<ZVolField>(), getNGhost<ZVolField>() );
          set_coord( field, patch, shift, 2 );
        }
        if( yZVolCoord_ ){
          SelectUintahFieldType<ZVolField>::type field;
          newDW->allocateAndPut( field, yZVol_, material, patch, getUintahGhostType<ZVolField>(), getNGhost<ZVolField>() );
          set_coord( field, patch, shift, 2 );
        }
        if( zZVolCoord_ ){
          SelectUintahFieldType<ZVolField>::type field;
          newDW->allocateAndPut( field, zZVol_, material, patch, getUintahGhostType<ZVolField>(), getNGhost<ZVolField>() );
          set_coord( field, patch, shift, 2 );
        }

      }  // material loop
    } // patch loop
  }

} // namespace Wasatch

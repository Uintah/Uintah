#ifndef Wasatch_CoordHelper_h
#define Wasatch_CoordHelper_h

#include "FieldTypes.h"

//-- Uintah Framework Includes --//
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SchedulerP.h>

// forward declarations
namespace Uintah{
  class DataWarehouse;
  class ProcessorGroup;
  class Task;
  class VarLabel;
}

namespace Expr{ class ExpressionFactory; }

namespace Wasatch{
  
  class CoordHelper
  {
    bool needCoords_,
      xSVolCoord_, ySVolCoord_,zSVolCoord_,
      xXVolCoord_, yXVolCoord_,zXVolCoord_,
      xYVolCoord_, yYVolCoord_,zYVolCoord_,
      xZVolCoord_, yZVolCoord_,zZVolCoord_;

    Uintah::VarLabel *xSVol_, *ySVol_, *zSVol_;
    Uintah::VarLabel *xXVol_, *yXVol_, *zXVol_;
    Uintah::VarLabel *xYVol_, *yYVol_, *zYVol_;
    Uintah::VarLabel *xZVol_, *yZVol_, *zZVol_;

    /** \brief sets the requested grid variables - callback for an initialization task */
    void set_grid_variables( const Uintah::ProcessorGroup* const pg,
                             const Uintah::PatchSubset* const patches,
                             const Uintah::MaterialSubset* const materials,
                             Uintah::DataWarehouse* const oldDW,
                             Uintah::DataWarehouse* const newDW );

    /** \brief registers requested coordinate fields */
    void register_coord_fields( Uintah::Task* const task,
                                const Uintah::PatchSet* const ps,
                                const Uintah::MaterialSet* const mss );

  public:

    CoordHelper( Expr::ExpressionFactory& );
    ~CoordHelper();

    /**
     *  \brief specify that we want coordinate information
     *  \param dir the coordinate (XDIR -> x coord)
     *
     *  This method can be called to request the coordinate for the
     *  given field type. For example,
     *
     *  \code requires_coordinate<SVolField>(XDIR) \endcode
     *
     *  results in the X-coordinate of the scalar volumes being
     *  populated as a field.  These can subsequently be used in a
     *  graph.
     *
     *  Note that coordinates have very specific names that should be
     *  used in the input file.  See Wasatch.cc and specifically the
     *  implementation of register_coord_fields for more information.
     */
    template<typename FieldT> void requires_coordinate( const Direction dir );

    void create_task( Uintah::SchedulerP& sched,
                      const Uintah::PatchSet* patches,
                      const Uintah::MaterialSet* materials );

  };



  template<> inline void CoordHelper::requires_coordinate<SVolField>( const Direction dir )
  {
    needCoords_ = true;
    switch (dir) {
    case XDIR : xSVolCoord_=true; break;
    case YDIR : ySVolCoord_=true; break;
    case ZDIR : zSVolCoord_=true; break;
    default: assert(0);
    }
  }
  template<> inline void CoordHelper::requires_coordinate<XVolField>( const Direction dir )
  {
    needCoords_ = true;
    switch (dir) {
    case XDIR : xXVolCoord_=true; break;
    case YDIR : yXVolCoord_=true; break;
    case ZDIR : zXVolCoord_=true; break;
    default: assert(0);
    }
  }
  template<> inline void CoordHelper::requires_coordinate<YVolField>( const Direction dir )
  {
    needCoords_ = true;
    switch (dir) {
    case XDIR : xYVolCoord_=true; break;
    case YDIR : yYVolCoord_=true; break;
    case ZDIR : zYVolCoord_=true; break;
    default: assert(0);
    }
  }
  template<> inline void CoordHelper::requires_coordinate<ZVolField>( const Direction dir )
  {
    needCoords_ = true;
    switch (dir) {
    case XDIR : xZVolCoord_=true; break;
    case YDIR : yZVolCoord_=true; break;
    case ZDIR : zZVolCoord_=true; break;
    default: assert(0);
    }
  }

} // namespace Wasatch

#endif // Wasatch_CoordHelper_h

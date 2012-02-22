#ifndef Wasatch_CoordHelper_h
#define Wasatch_CoordHelper_h

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/FieldAdaptor.h>

//-- Uintah Framework Includes --//
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SchedulerP.h>

//-- ExprLib includes --//
#include <expression/Tag.h>


// forward declarations
namespace Uintah{
  class DataWarehouse;
  class ProcessorGroup;
  class Task;
  class VarLabel;
}

namespace Expr{ class ExpressionFactory; }

namespace Wasatch{

  class StringNames;
  
  /**
   *  \class CoordHelper
   *  \author James C. Sutherland
   *  \ingroup WasatchCore
   *
   *  \brief Allows easy creation of coordinate fields.
   */
  class CoordHelper
  {
    const Expr::Context context_;
    const StringNames& sName_;
    const Expr::Tag xsvt_, ysvt_, zsvt_;
    const Expr::Tag xxvt_, yxvt_, zxvt_;
    const Expr::Tag xyvt_, yyvt_, zyvt_;
    const Expr::Tag xzvt_, yzvt_, zzvt_;

    bool needCoords_,
      xSVolCoord_, ySVolCoord_, zSVolCoord_,
      xXVolCoord_, yXVolCoord_, zXVolCoord_,
      xYVolCoord_, yYVolCoord_, zYVolCoord_,
      xZVolCoord_, yZVolCoord_, zZVolCoord_;

    Uintah::VarLabel *xSVol_, *ySVol_, *zSVol_;
    Uintah::VarLabel *xXVol_, *yXVol_, *zXVol_;
    Uintah::VarLabel *xYVol_, *yYVol_, *zYVol_;
    Uintah::VarLabel *xZVol_, *yZVol_, *zZVol_;

    Expr::TagSet fieldTags_;

    /** \brief sets the requested grid variables - callback for an initialization task */
    void set_grid_variables( const Uintah::ProcessorGroup* const pg,
                             const Uintah::PatchSubset* const patches,
                             const Uintah::MaterialSubset* const materials,
                             Uintah::DataWarehouse* const oldDW,
                             Uintah::DataWarehouse* const newDW );

    /** \brief registers requested coordinate fields */
    void register_coord_fields( Uintah::Task&,
                                const Uintah::PatchSet&,
                                const Uintah::MaterialSet& );

    template<typename FieldT> void reg_field( Uintah::VarLabel*& vl,
                                              const Expr::Tag tag,
                                              Uintah::Task& task,
                                              const Uintah::PatchSubset* const,
                                              const Uintah::MaterialSubset* const );

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

    /**
     *  \brief create a task to calculate the requested coordinates
     *  \param sched - The scheduler for this task
     *  \param patches - the set of patches that this task is associated with
     *  \param materials - the set of materials that this task is associated with
     */
    void create_task( Uintah::SchedulerP& sched,
                      const Uintah::PatchSet* patches,
                      const Uintah::MaterialSet* materials );

    /**
     *  \brief Obtain the TagSet describing all of the fields that
     *         have been requested of this CoordHelper object.
     */
    const Expr::TagSet& field_tags() const{ return fieldTags_; }

  };


  template<typename FieldT>
  inline void
  CoordHelper::reg_field( Uintah::VarLabel*& vl,
                          const Expr::Tag tag,
                          Uintah::Task& task,
                          const Uintah::PatchSubset* const pss,
                          const Uintah::MaterialSubset* const mss )
  {
    const Uintah::Task::MaterialDomainSpec domain = Uintah::Task::NormalDomain;
    vl = Uintah::VarLabel::create( tag.field_name(),
                                   get_uintah_field_type_descriptor<FieldT>(),
                                   get_uintah_ghost_descriptor<FieldT>() );
    fieldTags_.insert( tag );
    task.computes( vl, pss, domain, mss, domain );
  }

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

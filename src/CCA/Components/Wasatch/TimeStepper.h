#ifndef Wasatch_TimeStepper_h
#define Wasatch_TimeStepper_h

#include <set>

#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/ComputeSet.h>

#include <expression/ExpressionID.h>
#include <expression/FieldManager.h> // field type conversion tools
#include <expression/ExpressionFactory.h>
#include <expression/PlaceHolderExpr.h>
#include <expression/ExprLib.h>

#include "GraphHelperTools.h"

#include "PatchInfo.h"
#include "FieldAdaptor.h"
#include "FieldTypes.h"

#include <list>

namespace Uintah{
  class ProcessorGroup;
  class DataWarehouse;
}

namespace Wasatch{

  class CoordHelper;
  class TaskInterface;

  /**
   *  \ingroup WasatchCore
   *  \class  TimeStepper
   *  \author James C. Sutherland
   *  \date   June 2010
   *
   *  \brief Support for integrating a set of transport equations
   *         (explicit time integration methods for now).
   */
  class TimeStepper
  {
  public:
    /**
     *  \ingroup WasatchCore
     *  \struct FieldInfo
     *  \author James C. Sutherland
     *
     *  \brief provides strongly typed information about a field.
     *         These are used to provide information about what fields
     *         we are advancing in the time integrator, and their
     *         associated RHS expressions.
     */
    template<typename FieldT>
    struct FieldInfo
    {
      std::string varname;
      Uintah::VarLabel* varLabel;
      Uintah::VarLabel* rhsLabel;
      FieldInfo( const std::string& name,
                 Uintah::VarLabel* const vl,
                 Uintah::VarLabel* const rhsl )
        : varname( name ), varLabel( vl ), rhsLabel( rhsl )
      {}
    };

  private:

    typedef std::vector< FieldInfo<SpatialOps::structured::SVolField> > ScalarFields;
    typedef std::vector< FieldInfo<SpatialOps::structured::XVolField> > XVolFields;
    typedef std::vector< FieldInfo<SpatialOps::structured::YVolField> > YVolFields;
    typedef std::vector< FieldInfo<SpatialOps::structured::ZVolField> > ZVolFields;

    ScalarFields scalarFields_;  ///< A vector of the scalar fields being solved by this time integrator.
    XVolFields   xVolFields_;    ///< A vector of the x-volume fields being solved by this time integrator.
    YVolFields   yVolFields_;    ///< A vector of the y-volume fields being solved by this time integrator.
    ZVolFields   zVolFields_;    ///< A vector of the z-volume fields being solved by this time integrator.

    GraphHelper* const solnGraphHelper_;
    const Uintah::VarLabel* const deltaTLabel_;  ///< label for the time step variable.

    CoordHelper* coordHelper_;   ///< provides ability to obtain coordinate values on any field type.

    std::list  < TaskInterface*    > taskInterfaceList_;  ///< all of the TaskInterface objects managed here
    std::vector< Uintah::VarLabel* > createdVarLabels_;   ///< a list of all VarLabel objects created (so we can delete them later)

    /**
     *  \brief used internally to obtain the appropriate vector
     *         (e.g. scalarFields_) given the type of field we are
     *         considering.
     */
    template<typename FieldT>
    std::vector< FieldInfo<FieldT> >& field_info_selctor();

    /**
     *  \brief the call-back for Uintah to execute this.
     */
    void update_variables( const Uintah::ProcessorGroup* const,
                           const Uintah::PatchSubset* const,
                           const Uintah::MaterialSubset* const,
                           Uintah::DataWarehouse* const,
                           Uintah::DataWarehouse* const,
                           const int rkStage );

    void
    update_current_time( const Uintah::ProcessorGroup* const pg,
                         const Uintah::PatchSubset* const patches,
                         const Uintah::MaterialSubset* const materials,
                         Uintah::DataWarehouse* const oldDW,
                         Uintah::DataWarehouse* const newDW,
                         Expr::ExpressionTree::TreePtr timeTree,
                         const int rkStage );


  public:

    /**
     *  \brief Construct a TimeStepper object to advance equations forward in time
     *
     *  \param deltaTLabel - the VarLabel associated with the time step value
     *
     *  \param factory - the ExpressionFactory that will be used to
     *                   construct the trees for any transport
     *                   equations added to this library.  The same
     *                   factory should be used when constructing the
     *                   expressions in each transport equation.
     */
    TimeStepper( const Uintah::VarLabel* deltaTLabel,
                 GraphHelper& solnGraphHelper );

    ~TimeStepper();

    /**
     *  \brief Add a transport equation to this TimeStepper
     *
     *  \param solnVarName the name of the solution variable for this transport equation.
     *
     *  \param rhsID the Expr::ExpressionID for the right-hand-side of this transport equation.
     *
     *  This method is strongly typed to ensure that the solution
     *  variables are advanced properly and to guarantee compatibility
     *  with the Expression library.
     */
    template<typename FieldT>
    inline void add_equation( const std::string& solnVarName,
                              Expr::ExpressionID rhsID );

    /**
     *  \brief schedule the tasks associated with this TimeStepper
     *
     *  \param timeID the ExpressionID for the Expression that calculates the time.
     *  \param infoMap information about each patch including operators, etc.
     *  \param localPatches the patches that this task will be executed on
     *  \param materials the materials that this task will be executed on
     *  \param sched the scheduler
     */
    void create_tasks( const Expr::ExpressionID timeID,
                       const PatchInfoMap& infoMap,
                       const Uintah::PatchSet* const localPatches,
                       const Uintah::MaterialSet* const materials,
                       const Uintah::LevelP& level,
                       Uintah::SchedulerP& sched,
                       const int rkStage,
                       const std::set<std::string>& ioFieldSet);
  };

  //------------------------------------------------------------------

  template<>
  inline std::vector< TimeStepper::FieldInfo<SpatialOps::structured::SVolField> >&
  TimeStepper::field_info_selctor<SpatialOps::structured::SVolField>()
  {
    return scalarFields_;
  }
  template<>
  inline std::vector<TimeStepper::FieldInfo<SpatialOps::structured::XVolField> >&
  TimeStepper::field_info_selctor<SpatialOps::structured::XVolField>()
  {
    return xVolFields_;
  }
  template<>
  inline std::vector<TimeStepper::FieldInfo<SpatialOps::structured::YVolField> >&
  TimeStepper::field_info_selctor<SpatialOps::structured::YVolField>()
  {
    return yVolFields_;
  }
  template<>
  inline std::vector<TimeStepper::FieldInfo<SpatialOps::structured::ZVolField> >&
  TimeStepper::field_info_selctor<SpatialOps::structured::ZVolField>()
  {
    return zVolFields_;
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  void
  TimeStepper::add_equation( const std::string& solnVarName,
                             Expr::ExpressionID rhsID )
  {
    const std::string& rhsName = solnGraphHelper_->exprFactory->get_label(rhsID).name();
    const Uintah::TypeDescription* typeDesc = get_uintah_field_type_descriptor<FieldT>();
    const Uintah::IntVector ghostDesc       = get_uintah_ghost_descriptor<FieldT>();
    Uintah::VarLabel* solnVarLabel = Uintah::VarLabel::create( solnVarName, typeDesc, ghostDesc );
    Uintah::VarLabel* rhsVarLabel  = Uintah::VarLabel::create( rhsName,     typeDesc, ghostDesc );
    std::vector< FieldInfo<FieldT> >& fields = field_info_selctor<FieldT>();
    fields.push_back( FieldInfo<FieldT>( solnVarName, solnVarLabel, rhsVarLabel ) );
    //rhsIDs_.insert( rhsID );
    createdVarLabels_.push_back( solnVarLabel );
    createdVarLabels_.push_back( rhsVarLabel );

    typedef Expr::PlaceHolder<FieldT>  FieldExpr;
    solnGraphHelper_->exprFactory->register_expression( new typename FieldExpr::Builder(Expr::Tag(solnVarName,Expr::STATE_N  )) );
    solnGraphHelper_->exprFactory->register_expression( new typename FieldExpr::Builder(Expr::Tag(solnVarName,Expr::STATE_NP1)) );
  }

  //==================================================================

} // namespace Wasatch

#endif // Wasatch_TimeStepper_h

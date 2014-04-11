/*
 * The MIT License
 *
 * Copyright (c) 2012-2014-2014 The University of Utah
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

#ifndef Wasatch_TimeStepper_h
#define Wasatch_TimeStepper_h

#include <set>
#include <list>
#include <vector>

#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Task.h>

#include <expression/ExprFwd.h>

#include "GraphHelperTools.h"
#include "PatchInfo.h"
#include "FieldAdaptor.h"
#include "FieldTypes.h"

namespace Uintah{
  class ProcessorGroup;
  class DataWarehouse;
  class VarLabel;
}

namespace Wasatch{

  /**
   *  \ingroup WasatchCore
   *  \enum  TimeIntegratorEnum
   *  \author Tony Saad
   *  \date   July 2013
   *
   *  \brief Enum that defines the currently supported time integrators in Wasatch.
   */
  enum TimeIntegratorEnum {
    FE,      // Forward-Euler
    RK2SSP,  // Runge-Kutta 2nd order strong stability preserving
    RK3SSP   // Runge-Kutta 3rd order strong stability preserving
  };

  /**
   *  \ingroup WasatchCore
   *  \struct  TimeIntegrator
   *  \author Tony Saad
   *  \date   July 2013
   *
   *  \brief Defines coefficients for Runge-Kutta type integrators only two level
   storage requirements (i.e. old, and new).
   */
  struct TimeIntegrator {
    TimeIntegratorEnum timeIntEnum;
    int nStages;
    double alpha[3];
    double beta[3];

    TimeIntegrator(TimeIntegratorEnum theTimeIntEnum)
    : timeIntEnum(theTimeIntEnum)    
    {
      initialize();
    }

    TimeIntegrator(const std::string& timeIntName)
    : timeIntEnum( (timeIntName == "RK2SSP") ? RK2SSP : ( (timeIntName == "RK3SSP") ? RK3SSP : FE ) )
    {
      initialize();
    }
    
    void initialize()
    {
      switch (timeIntEnum) {
        default:
        case FE:
          nStages = 1;
          alpha[0] = 1.0; beta[0]  = 1.0;
          alpha[1] = 0.0; beta[1]  = 0.0;
          alpha[2] = 0.0; beta[2]  = 0.0;
          break;
        case RK2SSP:
          nStages = 2;
          alpha[0] = 1.0; beta[0]  = 1.0;
          alpha[1] = 0.5; beta[1]  = 0.5;
          alpha[2] = 0.0; beta[2]  = 0.0;
          break;
        case RK3SSP:
          nStages = 3;
          alpha[0] = 1.0;     beta[0]  = 1.0;
          alpha[1] = 0.75;    beta[1]  = 0.25;
          alpha[2] = 1.0/3.0; beta[2]  = 2.0/3.0;
          break;
      }
    }
  };

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
      bool operator==( const FieldInfo& fi ) const{ return varname.compare(fi.varname); }
      bool operator<( const FieldInfo& fi ) const{ return varname < fi.varname; }
      bool operator>( const FieldInfo& fi ) const{ return varname > fi.varname; }
    };

  private:

    Uintah::SimulationStateP sharedState_;

    typedef std::set< FieldInfo<SpatialOps::structured::SVolField> > ScalarFields;
    typedef std::set< FieldInfo<SpatialOps::structured::XVolField> > XVolFields;
    typedef std::set< FieldInfo<SpatialOps::structured::YVolField> > YVolFields;
    typedef std::set< FieldInfo<SpatialOps::structured::ZVolField> > ZVolFields;

    ScalarFields scalarFields_;  ///< A vector of the scalar fields being solved by this time integrator.
    XVolFields   xVolFields_;    ///< A vector of the x-volume fields being solved by this time integrator.
    YVolFields   yVolFields_;    ///< A vector of the y-volume fields being solved by this time integrator.
    ZVolFields   zVolFields_;    ///< A vector of the z-volume fields being solved by this time integrator.

    GraphHelper* const solnGraphHelper_;
    GraphHelper* const postProcGraphHelper_;      
    
    const TimeIntegrator timeInt_; ///< Multistage time integrator coefs

    std::vector< Uintah::VarLabel* > createdVarLabels_;   ///< a list of all VarLabel objects created (so we can delete them later)
    std::list< TaskInterface* > taskInterfaceList_;    ///< all of the TaskInterface objects managed here

    /**
     *  \brief used internally to obtain the appropriate vector
     *         (e.g. scalarFields_) given the type of field we are
     *         considering.
     */
    template<typename FieldT>
    std::set< FieldInfo<FieldT> >& field_info_selctor();

    /**
     *  \brief the call-back for Uintah to execute this.
     */
    void update_variables( Uintah::Task::CallBackEvent event, 
                           const Uintah::ProcessorGroup* const,
                           const Uintah::PatchSubset* const,
                           const Uintah::MaterialSubset* const,
                           Uintah::DataWarehouse* const,
                           Uintah::DataWarehouse* const,
                           void* stream,
                           const int rkStage );

    void
    update_current_time( const Uintah::ProcessorGroup* const pg,
                         const Uintah::PatchSubset* const patches,
                         const Uintah::MaterialSubset* const materials,
                         Uintah::DataWarehouse* const oldDW,
                         Uintah::DataWarehouse* const newDW,
                         Expr::ExpressionFactory* const factory,
                         const int rkStage );


  public:

    /**
     *  \brief Construct a TimeStepper object to advance equations forward in time
     *
     *  \param sharedState
     *
     *  \param factory - the ExpressionFactory that will be used to
     *                   construct the trees for any transport
     *                   equations added to this library.  The same
     *                   factory should be used when constructing the
     *                   expressions in each transport equation.
     */
    TimeStepper( Uintah::SimulationStateP sharedState,
                 GraphCategories& grafCat,
                 const TimeIntegrator timeInt);

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
    void add_equation( const std::string& solnVarName,
                       const Expr::ExpressionID& rhsID );

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

    const std::list< TaskInterface* >&
    get_task_interfaces() const{ return taskInterfaceList_; }

  };

  //==================================================================

} // namespace Wasatch

#endif // Wasatch_TimeStepper_h

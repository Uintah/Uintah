/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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
#include <expression/Tag.h>
#include <expression/ExprFwd.h>

#include "GraphHelperTools.h"
#include "PatchInfo.h"
#include "FieldAdaptor.h"
#include "FieldTypes.h"
#include <CCA/Components/Wasatch/TimeIntegratorTools.h>
#include <expression/dualtime/BDFDualTimeIntegrator.h>

namespace WasatchCore{

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
      
      Expr::Tag solnVarTag;
      Expr::Tag rhsTag;
      
      FieldInfo( const std::string& name,
                 Expr::Tag varTag,
                 Expr::Tag rhsVarTag )
        : varname( name ), solnVarTag(varTag), rhsTag(rhsVarTag)
      {}
      bool operator==( const FieldInfo& fi ) const{ return varname.compare(fi.varname); }
      bool operator<( const FieldInfo& fi ) const{ return varname < fi.varname; }
      bool operator>( const FieldInfo& fi ) const{ return varname > fi.varname; }
    };

  private:

    typedef std::map<int, Expr::DualTime::BDFDualTimeIntegrator*> DTIntegratorMapT;
    Uintah::SimulationStateP sharedState_;

    typedef std::set< FieldInfo<SpatialOps::SVolField              > > ScalarFields;
    typedef std::set< FieldInfo<SpatialOps::XVolField              > > XVolFields;
    typedef std::set< FieldInfo<SpatialOps::YVolField              > > YVolFields;
    typedef std::set< FieldInfo<SpatialOps::ZVolField              > > ZVolFields;
    typedef std::set< FieldInfo<SpatialOps::Particle::ParticleField> > ParticleFields;

    ScalarFields   scalarFields_;   ///< the scalar   fields being solved by this time integrator.
    XVolFields     xVolFields_;     ///< the x-volume fields being solved by this time integrator.
    YVolFields     yVolFields_;     ///< the y-volume fields being solved by this time integrator.
    ZVolFields     zVolFields_;     ///< the z-volume fields being solved by this time integrator.
    ParticleFields particleFields_; ///< the particle fields being solved by this time integrator.
    
    GraphHelper* const solnGraphHelper_;
    GraphHelper* const postProcGraphHelper_;      
    
    const TimeIntegrator timeInt_; ///< Multistage time integrator coefs

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

  public:

    /**
     *  \brief Construct a TimeStepper object to advance equations forward in time
     *
     *  \param sharedState
     *  \param grafCat
     *  \param timeInt the time integrator to use
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
     *  \param rhsTag the Expr::Tag for the right-hand-side of this transport equation.
     *
     *  This method is strongly typed to ensure that the solution
     *  variables are advanced properly and to guarantee compatibility
     *  with the Expression library.
     */
    template<typename FieldT>
    void add_equation( const std::string& solnVarName,
                       const Expr::Tag& rhsTag );

    /**
     *  \brief Add a collection of transport equations to this TimeStepper
     *
     *  \param solnVarTags the Expr::TagList corresponding to the solution variables of these transport equations.
     *
     *  \param rhsTags the Expr::TagList corresponding to the right-hand-sides of these transport equations.
     *
     *  This method is strongly typed to ensure that the solution
     *  variables are advanced properly and to guarantee compatibility
     *  with the Expression library.
     */
    template<typename FieldT>
    void add_equations( const Expr::TagList& solnVarTags,
                        const Expr::TagList& rhsTags );

    /**
     *  \brief schedule the tasks associated with this TimeStepper
     *
     *  \param infoMap information about each patch including operators, etc.
     *  \param localPatches the patches that this task will be executed on
     *  \param materials the materials that this task will be executed on
     *  \param level the level of interest
     *  \param sched the scheduler
     *  \param rkStage the RK stage (1 for forward euler)
     *  \param ioFieldSet the set of fields that should be locked to maintain persistence
     */
    void create_tasks( const PatchInfoMap& infoMap,
                       const Uintah::PatchSet* const localPatches,
                       const Uintah::MaterialSet* const materials,
                       const Uintah::LevelP& level,
                       Uintah::SchedulerP& sched,
                       const int rkStage,
                       const std::set<std::string>& ioFieldSet );
    
    /**
     *  \brief schedule the tasks associated with this TimeStepper
     *
     *  \param infoMap information about each patch including operators, etc.
     *  \param localPatches the patches that this task will be executed on
     *  \param materials the materials that this task will be executed on
     *  \param level the level of interest
     *  \param sched the scheduler
     *  \param dualTimeIntegrators
     *  \param ioFieldSet the set of fields that should be locked to maintain persistence
     */
    void create_dualtime_tasks( const PatchInfoMap& infoMap,
                                const Uintah::PatchSet* const localPatches,
                                const Uintah::MaterialSet* const materials,
                                const Uintah::LevelP& level,
                                Uintah::SchedulerP& sched,
                                DTIntegratorMapT& dualTimeIntegrators,
                                const std::set<std::string>& ioFieldSet );


    const std::list< TaskInterface* >&
    get_task_interfaces() const{ return taskInterfaceList_; }

  };

  //==================================================================

} // namespace WasatchCore

#endif // Wasatch_TimeStepper_h

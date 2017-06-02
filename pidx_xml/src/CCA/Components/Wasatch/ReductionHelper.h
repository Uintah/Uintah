/**
 *  \file   ReductionHelper.h
 *  \date   June, 2013
 *  \author "Tony Saad"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013-2017 The University of Utah
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

#ifndef Reduction_Helper_h
#define Reduction_Helper_h

//-- stl includes --//
#include <string>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Expressions/ReductionBase.h>
#include "GraphHelperTools.h"

//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Material.h>

//-- Expression includes --//
#include <expression/ExprFwd.h>
#include <expression/ExpressionTree.h>

// forward declarations
namespace Uintah{
  class ProcessorGroup;
  class DataWarehouse;
}

namespace WasatchCore {

  class Wasatch;  // forward declaration

  /**
   *  \class  ReductionHelper
   *  \author Tony Saad
   *  \date   June, 2013
   *  \brief  This class provides support for creating reduction variables.
   *
   *  Example usage:
   *  \code
   *   ReductionHelper& redHelp = ReductionHelper::self();
   *   redHelp.add_variable<SVolField, ReductionMinOpT  >( graph_category, velocity_min, varTag1 );
   *   redHelp.add_variable<SSurfXField, ReductionMaxOpT>( graph_category, varTag2 );
   *   redHelp.add_variable<XVolField, ReductionSumOpT >( graph_category, varTag3 );
   *   // ...
   *   redHelp.schedule_tasks(Uintah::getLevelP(pss), scheduler_, materials_, tree, patchID, rkStage);
   *  \endcode
   *
   *  \warning Note that add_variable will ONLY register the appropriate reduction
   expressions to the graph and cleave from parents and children. You still need
   to schedule the reduction expressions at the proper locations. You can use
   either ReductionHelper::self().schedule_tasks(...) or schedule each of the 
   reduction expressions independently.
   *
   *  \todo describe what happens behind the scenes
   */
  class ReductionHelper
  {
  public:
    /**
     * @return the singleton instance of ReductionHelper
     */
    static ReductionHelper& self();

    ~ReductionHelper();

    void sync_with_wasatch( Wasatch* const wasatch );

    ReductionEnum select_reduction_enum( const std::string& strRedOp );
    
    void parse_reduction_spec( Uintah::ProblemSpecP reductionParams );
    
    /**
     * @brief add a new variable to the list of reduction variables.
     * @param category indicates which task category this should be active on.
     * @param resultTag indicates tag of the resulting reduction expression, i.e.
              ("pressure_min", Expr::NONE).
     * @param srcTag indicates tag of the source expression on which the reduction
              is to be applied, i.e. ("pressure", Expr::NONE).
     * @param printVar indicates whether you want to output the reduction result
              to the processor_0 std::cout stream.
     * @param reduceOnAllRKStages Indicates whether you want to perform the reduction
              at every intermediate runge-kutta stage. See warning below for advice
              on use this parameter.
     *
     \warning It is recommended that you do NOT reduce at every single runge-kutta stage.
     Reduction at every stage not only reduces scalability but also induces the creation
     and managment of additional Uintah variables.
     */
    template<typename SrcT, typename ReductionOpT>
    void add_variable( const Category category,
                       const Expr::Tag& resultTag,
                       const Expr::Tag& srcTag,
                       const bool printVar=false,
                       const bool reduceOnAllRKStages=false );
    
    void schedule_tasks( const Uintah::LevelP& level,
                         Uintah::SchedulerP sched,
                         const Uintah::MaterialSet* const materials,
                         const Expr::ExpressionTree::TreePtr tree,
                         const int patchID,
                         const int rkStage );// do we really need the rkstage here? maybe in the future...
  private:

    ReductionHelper();

    Wasatch* wasatch_;
    bool wasatchSync_;
    bool hasDoneSetup_;
    bool reduceOnAllRKStages_;
  };

} /* namespace WasatchCore */
#endif /* Reduction_Helper_h */

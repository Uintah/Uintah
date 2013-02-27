/*
 * The MIT License
 *
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

#ifndef Wasatch_TaskInterface_h
#define Wasatch_TaskInterface_h

#include <set>

//-- Uintah includes --//
#include <CCA/Ports/SchedulerP.h>
#include <Core/Grid/Variables/ComputeSet.h>

//-- Wasatch includes --//
#include "PatchInfo.h"
#include "GraphHelperTools.h"

// forward declarations
namespace Uintah{
  class DataWarehouse;
  class Task;
  class ProcessorGroup;
  class Material;
  class Patch;
}

namespace Wasatch{

  class TreeTaskExecute;

  /**
   *  \ingroup WasatchCore
   *  \ingroup WasatchGraph
   *  \class  TaskInterface
   *  \author James C. Sutherland
   *  \date   June, 2010
   *
   *  \brief Provides an interface to Uintah
   *
   *  The TaskInterface class provides an interface to Uintah for the
   *  ExpressionTree and Expression objects.  Each time this task is
   *  executed, the fields and operators are re-bound on all of the
   *  expressions.  This is because one ExpressionTree (and hence
   *  Uintah::Task) may be executed on multiple patches (data).
   *  Therefore, we must re-bind these to the data specific to the
   *  patch that they are being executed on.  Alternatively, a unique
   *  tree and set of expressions could be constructed for each
   *  patch. This would require more memory, however.
   *
   *  There are two main "modes" that a TaskInterface can be created
   *  to support:
   *
   *   - Create expressions and the associated ExpressionTree on each
   *      patch. This is required when setting boundary conditions,
   *      since this must modify expressions on boundary patches.
   *
   *   - Create expressions and the associated ExpressionTree only
   *     once. This will be executed on each patch. This will only be
   *     valid if all patches are homogeneous, i.e. no boundary
   *     conditions are being applied.
   *
   *  These modes are selected when the TaskInterface is constructed.
   *
   *  \todo for tree cleaving, we need to put MODIFIES flags on fields
   *        rather than COMPUTES or REQUIRES.  This might be a bit sticky.
   */
  class TaskInterface
  {
  public:

    /**
     *  \brief Create a TaskInterface from a list of root expressions
     *
     *  \param roots The root nodes of the tree to create
     *  \param taskName the name of this task
     *  \param factory the Expr::ExpressionFactory that will be used to build the tree.
     *  \param sched The Scheduler that this task will be loaded on.
     *  \param patches the patches to associate this task with.
     *  \param materials the MaterialSet for the materials associated with this task
     *  \param info The PatchInfoMap object.
     *  \param RKStage the stage of the RK integrator (use 1 otherwise)
     *  \param ioFieldSet the fields that are required for output and should not
     *         be managed "externally" so that their memory is not reclaimed.
     *
     *  This registers fields on the FieldManagerList (which is created if necessary).
     */
    TaskInterface( const IDSet& roots,
                   const std::string taskName,
                   Expr::ExpressionFactory& factory,
                   const Uintah::LevelP& level,
                   Uintah::SchedulerP& sched,
                   const Uintah::PatchSet* const patches,
                   const Uintah::MaterialSet* const materials,
                   const PatchInfoMap& info,
                   const int RKStage,
                   const std::set<std::string>& ioFieldSet );

    ~TaskInterface();

    /**
     *  \brief Schedule for execution with Uintah
     *  \param newDWFields - a vector of Expr::Tag indicating fields
     *         should be pulled from the new DW.  This is particularly
     *         useful for situations where another task will be
     *         computing a given field and the ExpressionTree has
     *         wrapped that field as a PlaceHolderExpr.  This helps us
     *         determine where the field will exist in Uintah's
     *         DataWarehouse
     *
     *  This sets all field requirements for the Uintah task and
     *  scheduled it for execution.
     */
    void schedule( const Expr::TagSet& newDWFields, const int RKStage = 1 );


    /**
     *  \brief Schedule for execution with Uintah
     *
     *  This sets all field requirements for the Uintah task and
     *  scheduled it for execution.
     */
    void schedule( const int RKStage = 1);

    /**
     * Obtain a TagList containing all tags computed by the graph(s)
     * associated with this TaskInterface.
     */
    Expr::TagList collect_tags_in_task() const;

  private:

    /**
     *  \brief A vector of TreeTaskExecute pointers that hold each
     *  Uintah::Task to be executed as part of this TaskInterface.
     */
    typedef std::vector< TreeTaskExecute* > ExecList;

    /**
     *  The ordered list of trees to be executed as tasks.  This is
     *  obtained by cleaving the original tree obtained by the root
     *  expressions supplied to the TaskInterface.
     */
    ExecList execList_;
  };

} // namespace Wasatch

#endif // Wasatch_TaskInterface_h

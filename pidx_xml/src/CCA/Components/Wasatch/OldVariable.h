/**
 *  \file   OldVariable.h
 *  \date   Feb 7, 2013
 *  \author "James C. Sutherland"
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

#ifndef OLDVARIABLE_H_
#define OLDVARIABLE_H_

#include <string>
#include <list>

#include <Core/Grid/Variables/ComputeSet.h>
#include <CCA/Ports/SchedulerP.h>

#include <expression/ExprFwd.h>

#include "GraphHelperTools.h"

// forward declarations
namespace Uintah{
  class ProcessorGroup;
  class DataWarehouse;
}

namespace WasatchCore {

  class Wasatch;  // forward declaration

  /**
   *  \class OldVariable
   *  \author James C. Sutherland, Tony Saad
   *  \brief  This class provides support for carrying old variables around. It
   can also be used to copy variables from the old datawarehouse.
   *
   *  Example usage:
   *  \code
   *   OldVariable& oldVar = OldVariable::self();
   *   oldVar.add_variable<SVolField  >( graph_category, varTag1 );
   *   oldVar.add_variable<SSurfXField>( graph_category, varTag2 );
   *   oldVar.add_variable<XVolField  >( graph_category, varTag3 );
   *   // ...
   *   oldVar.setup_tasks( patches, materials, sched );
   *  \endcode
   *
   *  Behind the scenes, the following operations are performed:
   *   -# Create an "old" version of the variable
   *   -# Create a PlaceHolder expression for the "old" variable on
   *      the given ExpressionFactory.
   *   -# Create a task that will move all of the variables forward.
   *
   *  Note that when you want to only copy variables, you should use:
   *  \code
   *   OldVariable& oldVar = OldVariable::self();
   *   oldVar.add_variable<SVolField  >( graph_category, varTag1, true );
   *  \endcode
   *  This will basically retain the same varlabel name in the newDW.
   *
   *  \todo Need to prevent the Expression graph from marking these fields as temporary.
   */
  class OldVariable
  {
  public:
    /**
     * @return the singleton instance of OldVariable
     */
    static OldVariable& self();

    ~OldVariable();

    void sync_with_wasatch( Wasatch* const wasatch );

    /**
     * @brief creates and schedules tasks for all of the variables requested.
     * @param patches
     * @param materials
     * @param sched
     * @param rkStage the Runge-Kutta stage
     */
    void setup_tasks( const Uintah::PatchSet* const patches,
                      const Uintah::MaterialSet* const materials,
                      Uintah::SchedulerP& sched,
                      const int rkStage );

    /**
     * @brief add a new variable to the list of variables that should have
     *  "old" copies retained.
     * @param category indicates which task category this should be active on
     * @param var the variable that we want an "old" copy for
     * @param retainName indicates whether to retain the same varlabel for the
     *        the old variable. Set this to true when you want to copy a variable
     *        from the old DW to keep it floating around and avoid recalculating it.
     *
     * When a variable is added, an "old" counterpart is created and a
     * PlaceHolder expression is registered with the factory associated
     * with the specified category.  After the variable's value has been
     * computed, it will automatically be copied to the "old" value via
     * a Task that is created when "setup_tasks" is called.
     */
    template< typename T >
    void add_variable( const Category category,
                       const Expr::Tag& var,
                       const bool retainName=false);

    class VarHelperBase;  // forward

  private:

    OldVariable();

    void populate_old_variable( const Uintah::ProcessorGroup* const pg,
                                const Uintah::PatchSubset* const patches,
                                const Uintah::MaterialSubset* const materials,
                                Uintah::DataWarehouse* const oldDW,
                                Uintah::DataWarehouse* const newDW,
                                const int rkStage);

    Wasatch* wasatch_;
    std::list<VarHelperBase*> varHelpers_;
    bool wasatchSync_;
    bool hasDoneSetup_;
  };

} /* namespace WasatchCore */
#endif /* OLDVARIABLE_H_ */

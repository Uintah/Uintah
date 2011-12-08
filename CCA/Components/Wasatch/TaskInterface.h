#ifndef Wasatch_TaskInterface_h
#define Wasatch_TaskInterface_h

#include <set>

//-- Uintah includes --//
#include <CCA/Ports/SchedulerP.h>
#include <Core/Grid/Variables/ComputeSet.h>

//-- Wasatch includes --//
#include "PatchInfo.h"
#include "GraphHelperTools.h"

//-- ExprLib includes --//
#include <expression/ExpressionTree.h>

// forward declarations
namespace Uintah{
  class DataWarehouse;
  class Task;
  class ProcessorGroup;
  class Material;
  class Patch;
}

namespace Expr{
  class FieldManagerList;
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
     *
     *  \param taskName the name of this task
     *
     *  \param factory the Expr::ExpressionFactory that will be used to build the tree.
     *
     *  \param sched The Scheduler that this task will be loaded on.
     *
     *  \param patches the patches to associate this task with.
     *
     *  \param materials the MaterialSet for the materials associated with this task
     *
     *  \param info The PatchInfoMap object.
     *
     *  \param createUniqueTreePerPatch if true, then a tree will be
     *         constructed for each patch.  If false, one tree will be
     *         used across all patches.
     *
     *  \param fml [OPTIONAL] the FieldManagerList to associate with
     *         this expression.  If not supplied, one will be created.
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
                   const bool createUniqueTreePerPatch,
                   const int RKStage,
                   const std::set<std::string>& ioFieldSet,
                   Expr::FieldManagerList* fml = NULL );

    /**
     *  \brief Create a TaskInterface from a list of root expressions
     *
     *  \param root The root node of the tree to create
     *
     *  \param taskName the name of this task
     *
     *  \param factory the Expr::ExpressionFactory that will be used to build the tree.
     *
     *  \param sched The Scheduler that this task will be loaded on.
     *
     *  \param patches the patches to associate this task with.
     *
     *  \param materials the MaterialSet for the materials associated with this task
     *
     *  \param info The PatchInfoMap object.
     *
     *  \param createUniqueTreePerPatch if true, then a tree will be
     *         constructed for each patch.  If false, one tree will be
     *         used across all patches.
     *
     *  \param fml [OPTIONAL] the FieldManagerList to associate with
     *         this expression.  If not supplied, one will be created.
     *
     *  This registers fields on the FieldManagerList (which is created if necessary).
     */
    TaskInterface( const Expr::ExpressionID& root,
                   const std::string taskName,
                   Expr::ExpressionFactory& factory,
                   const Uintah::LevelP& level,
                   Uintah::SchedulerP& sched,
                   const Uintah::PatchSet* const patches,
                   const Uintah::MaterialSet* const materials,
                   const PatchInfoMap& info,
                   const bool createUniqueTreePerPatch,
                   const int RKStage,
                   const std::set<std::string>& ioFieldSet,
                   Expr::FieldManagerList* fml = NULL );

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
    void schedule( const Expr::TagSet& newDWFields, const int RKStage );


    /**
     *  \brief Schedule for execution with Uintah
     *
     *  This sets all field requirements for the Uintah task and
     *  scheduled it for execution.
     */
    void schedule( const int RKStage);

    Expr::ExpressionTree::TreePtr get_time_tree();

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

    const bool builtFML_;               ///< true if we constructed a FieldManagerList internally.
    Expr::FieldManagerList* const fml_; ///< the FieldManagerList for this TaskInterface

  };

} // namespace Wasatch

#endif // Wasatch_TaskInterface_h

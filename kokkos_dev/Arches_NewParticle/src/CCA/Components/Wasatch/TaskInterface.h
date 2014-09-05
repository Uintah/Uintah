#ifndef Wasatch_TaskInterface_h
#define Wasatch_TaskInterface_h

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

  /**
   *  \ingroup WasatchCore
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
                   Uintah::SchedulerP& sched,
                   const Uintah::PatchSet* const patches,
                   const Uintah::MaterialSet* const materials,
                   const PatchInfoMap& info,
                   const bool createUniqueTreePerPatch,
                   Expr::FieldManagerList* fml = NULL );

    TaskInterface( const Expr::ExpressionID& root,
                   const std::string taskName,
                   Expr::ExpressionFactory& factory,
                   Uintah::SchedulerP& sched,
                   const Uintah::PatchSet* const patches,
                   const Uintah::MaterialSet* const materials,
                   const PatchInfoMap& info,
                   const bool createUniqueTreePerPatch,
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
    void schedule( const Expr::TagSet& newDWFields );


    /**
     *  \brief Schedule for execution with Uintah
     *
     *  This sets all field requirements for the Uintah task and
     *  scheduled it for execution.
     */
    void schedule();

  private:

    typedef std::pair< Expr::ExpressionTree*, Uintah::Task* > TreeTaskPair;
    typedef std::map< int, TreeTaskPair > PatchTreeMap;

    Uintah::SchedulerP& scheduler_;
    const Uintah::PatchSet* const patches_;
    const Uintah::MaterialSet* const materials_;

    const bool createUniqueTreePerPatch_;
    PatchTreeMap patchTreeMap_;

    const std::string taskName_;        ///< the name of the task

    const PatchInfoMap& patchInfoMap_;  ///< information for each individual patch.

    const bool builtFML_;               ///< true if we constructed a FieldManagerList internally.
    Expr::FieldManagerList* const fml_; ///< the FieldManagerList for this TaskInterface

    bool hasBeenScheduled_;             ///< true after the call to schedule().  Must be true prior to add_fields_to_task().

    void setup_tree( const IDSet& roots,
                     Expr::ExpressionFactory& factory );

    /** advertises field requirements to Uintah. */
    static void add_fields_to_task( Uintah::Task& task,
                                    const Expr::ExpressionTree& tree,
                                    Expr::FieldManagerList& fml,
                                    const Uintah::PatchSubset* const patches,
                                    const Uintah::MaterialSubset* const materials,
                                    const std::vector<Expr::Tag>& );

    /** main execution driver - the callback function exposed to Uintah. */
    void execute( const Uintah::ProcessorGroup* const,
                  const Uintah::PatchSubset* const,
                  const Uintah::MaterialSubset* const,
                  Uintah::DataWarehouse* const,
                  Uintah::DataWarehouse* const );

  };

} // namespace Wasatch

#endif // Wasatch_TaskInterface_h

#ifndef Wasatch_TaskInterface_h
#define Wasatch_TaskInterface_h

//-- Uintah includes --//
#include <CCA/Ports/SchedulerP.h>
#include <Core/Grid/Variables/ComputeSet.h>

//-- Wasatch includes --//
#include "PatchInfo.h"

// forward declarations
namespace Uintah{
  class DataWarehouse;
  class Task;
  class ProcessorGroup;
  class Material;
  class Patch;
}

namespace Expr{
  class ExpressionTree;
  class FieldDeps;
  class FieldManagerList;
}

namespace SpatialOps{
  class OperatorDatabase;
}


namespace Wasatch{


  /**
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
   *  \todo for tree cleaving, we need to put MODIFIES flags on fields
   *        rather than COMPUTES or REQUIRES.  This might be a bit sticky.
   */
  class TaskInterface
  {
  public:

    /**
     *  \brief Create a TaskInterface from an ExpressionTree
     *
     *  \param tree the ExpressionTree that we want to expose as a Uintah::Task
     *
     *  \param fml [OPTIONAL] the FieldManagerList to associate with
     *         this expression.  If not supplied, one will be created.
     *
     *  This registers fields on the FieldManagerList (which is created if necessary).
     */
    TaskInterface( Expr::ExpressionTree* tree,
                   const PatchInfoMap& info,
                   Expr::FieldManagerList* fml = NULL );

    ~TaskInterface();

    /**
     *  \brief Schedule for execution with Uintah
     *  \param scheduler the Uintah::Scheduler that we will put this task on
     *  \param patches the Uintah::PatchSet associated with this task
     *  \param material the Uintah::MaterialSet associated with this task
     *
     *  This sets all field requirements for the Uintah task and
     *  scheduled it for execution.
     */
    void schedule( Uintah::SchedulerP& scheduler,
                   const Uintah::PatchSet* const patches,
                   const Uintah::MaterialSet* const materials );
   
  private:

    Expr::ExpressionTree* const tree_;  ///< the underlying ExpressionTree associated with this task.

    const std::string taskName_;        ///< the name of the task

    const PatchInfoMap& patchInfoMap_;  ///< information for each individual patch.

    const bool builtFML_;               ///< true if we constructed a FieldManagerList internally.
    Expr::FieldManagerList* const fml_; ///< the FieldManagerList for this TaskInterface

    Uintah::Task* const uintahTask_;    ///< the Uintah::Task that is created from this TaskInterface.
    bool hasBeenScheduled_;             ///< true after the call to schedule().  Must be true prior to add_fields_to_task().

    /** advertises field requirements to Uintah. */
    void add_fields_to_task( const Uintah::PatchSet* const patches,
                             const Uintah::MaterialSet* const materials );

    /** iterates all Expressions in the ExpressionTree and binds fields/operators. */
    void bind_fields_operators( const Expr::AllocInfo&,
                                const SpatialOps::OperatorDatabase& opDB );

    /** main execution driver - the callback function exposed to Uintah. */
    void execute( const Uintah::ProcessorGroup* const,
                  const Uintah::PatchSubset* const,
                  const Uintah::MaterialSubset* const,
                  Uintah::DataWarehouse* const,
                  Uintah::DataWarehouse* const );

  };

} // namespace Wasatch

#endif // Wasatch_TaskInterface_h

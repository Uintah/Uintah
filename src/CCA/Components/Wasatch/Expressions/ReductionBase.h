#ifndef ReductionBase_Expr_h
#define ReductionBase_Expr_h

//-- ExprLib Includes --//
#include <expression/Expression.h>

//-- Uintah Includes --//
#include <Core/Grid/Variables/Reductions.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Material.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>

// reduction diagnostics
static SCIRun::DebugStream dbgr("Wasatch_Reduction", false);
#define dbg_reduction_on  dbgr.active()
#define dbg_red  if( dbg_reduction_on  ) dbgr

// useful reduction enum
enum ReductionEnum {
  ReduceMin,
  ReduceMax,
  ReduceSum
};

// some useful typedefs
typedef Uintah::Reductions::Min<double> ReductionMinOpT;
typedef Uintah::Reductions::Max<double> ReductionMaxOpT;
typedef Uintah::Reductions::Sum<double> ReductionSumOpT;

/**
 *  \class  ReductionBase
 *  \author Tony Saad
 *  \date   June, 2013
 *  \brief  A base class that provides a unified interface to insert Uintah reduction
 tasks between Wasatch tasks.
 *
 *  \warning This class should not be used as is unless needed. It merely provides
 a simple way to schedule Uintah reduction tasks without a templated source field
 or reduction type. See TaskInterface.cc for an example of how this base class
 is used.
 */
class ReductionBase
 : public Expr::Expression<double>
{

protected:
  const Expr::Tag srcTag_;               // Expr::Tag for the source field on which a reduction is to be applied
  Uintah::VarLabel *reductionVarLabel_;  // this is the reduction varlabel
  Uintah::VarLabel *thisVarLabel_;       // varlabel for the current perpatch expression
  ReductionEnum reductionName_;          // enum that provides simple switch option
  bool printVar_;                        // specify whether you want the reduction var printed or not
  
  void
  populate_reduction_variable( const Uintah::ProcessorGroup* const pg,
                              const Uintah::PatchSubset* const patches,
                              const Uintah::MaterialSubset* const materials,
                              Uintah::DataWarehouse* const oldDW,
                              Uintah::DataWarehouse* const newDW );
  
  void
  get_reduction_variable( const Uintah::ProcessorGroup* const pg,
                         const Uintah::PatchSubset* const patches,
                         const Uintah::MaterialSubset* const materials,
                         Uintah::DataWarehouse* const oldDW,
                         Uintah::DataWarehouse* const newDW );

  ReductionBase( const Expr::Tag& resultTag,
                 const Expr::Tag& srcTag,
                 Uintah::VarLabel* reductionVarLabel,
                 ReductionEnum reductionName,
                 bool printVar=false);
  
public:
  
  //--------------------------------------------------------------------
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a Reduction expression
     *  @param resultTag The tag for the value that this expression computes.
     *  @param srcTag The tag of the source field on which we want to apply reduction.
     *  @param reductionVarLabel The Uintah::VarLabel of the Uintah::ReductionVariable<>.
     *  @param reductionName An enum designating the type of reduction operation. put here for ease of use.
     *  @param printVar A boolean that specifies whether you want the reduced variable output to the cout stream.
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& srcTag,
             Uintah::VarLabel* reductionVarLabel,
             ReductionEnum reductionName,
             bool printVar=false);

    Expr::ExpressionBase* build() const;

  private:
    const Expr::Tag resultTag_, srcTag_;
    Uintah::VarLabel* reductionVarLabel_;
    ReductionEnum reductionName_;
    bool printVar_;
  };
  //--------------------------------------------------------------------

  ~ReductionBase();
  
  /**
   *  \brief A static list of tags that keeps track of all reduction variables.
   *
   */
  static Expr::TagList reductionTagList;

  /**
   *  \brief Schedules a Uintah task that populates a reduction variable and
   also fills in a PerPatch (i.e. Expression<double>) with the reduced values.
   *
   */
  void schedule_set_reduction_vars( const Uintah::LevelP& level,
                              Uintah::SchedulerP sched,
                              const Uintah::MaterialSet* const materials,
                              const int RKStage );
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
};

#endif // ReductionBase_Expr_h

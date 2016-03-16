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
static Uintah::DebugStream dbgr("WASATCH_REDUCTIONS", false);
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
 : public Expr::Expression<SpatialOps::SingleValueField>
{
  typedef Uintah::ReductionVariable<double, ReductionMinOpT > UintahReduceMin;
  typedef Uintah::ReductionVariable<double, ReductionMaxOpT > UintahReduceMax;
  typedef Uintah::ReductionVariable<double, ReductionSumOpT > UintahReduceSum;

protected:
  const Expr::Tag srcTag_;               // Expr::Tag for the source field on which a reduction is to be applied
  std::vector<Uintah::VarLabel*> rkRedVarLbls_; // these are the reduction varlabels, for each RK stage
  Uintah::VarLabel *thisVarLabel_;       // varlabel for the current perpatch expression
  const ReductionEnum reductionName_;          // enum that provides simple switch option
  const bool printVar_;                        // specify whether you want the reduction var printed or not
  
  void
  populate_reduction_variable( const Uintah::ProcessorGroup* const pg,
                               const Uintah::PatchSubset* const patches,
                               const Uintah::MaterialSubset* const materials,
                               Uintah::DataWarehouse* const oldDW,
                               Uintah::DataWarehouse* const newDW,
                               const int RKStage );
  
  void
  get_reduction_variable( const Uintah::ProcessorGroup* const pg,
                          const Uintah::PatchSubset* const patches,
                          const Uintah::MaterialSubset* const materials,
                          Uintah::DataWarehouse* const oldDW,
                          Uintah::DataWarehouse* const newDW,
                          const int RKStage );

  ReductionBase( const Expr::Tag& resultTag,
                 const Expr::Tag& srcTag,
                 const ReductionEnum reductionName,
                 const bool printVar=false);
  
public:
  
  ~ReductionBase();
  
  /**
   *  \brief A static map of tags that keeps track of all reduction variables along with a boolean that specifies whether reduction on
       that variable must be done at every Runge-Kutta stage
   *
   */
  static std::map<Expr::Tag, bool > reductionTagList;  // jcs should not be public.

  /**
   *  \brief Schedules a Uintah task that populates a reduction variable and
   also fills in a PerPatch (i.e. Expression<double>) with the reduced values.
   *
   */
  void schedule_set_reduction_vars( const Uintah::LevelP& level,
                              Uintah::SchedulerP sched,
                              const Uintah::MaterialSet* const materials,
                              const int RKStage );
  
  void evaluate();
};

#endif // ReductionBase_Expr_h

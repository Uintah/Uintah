/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, subl‰icense, and/or
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

#ifndef RadiationSource_Expr_h
#define RadiationSource_Expr_h

//-- ExprLib Includes --//
#include <expression/Expression.h>

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Operators/Operators.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/BCHelper.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

//-- Uintah Includes --//
#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/Stencil4.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>

namespace Uintah{
  class Ray;
}

namespace Wasatch{

/**
 *  \class 	  RadiationSource
 *  \ingroup 	Expressions
 *  \ingroup	WasatchCore
 *  \authors  Tony Saad, James C. Sutherland
 *  \date 	  January, 2014
 *
 *  \brief Expression to form and solve the poisson system for RadiationSource.
 *
 *  \note: this expression BREAKS WITH CONVENTION!  Notably, it has
 *  uintah tenticles that reach into it, and mixes SpatialOps and
 *  Uintah constructs.  This is because we don't (currently) have a
 *  robust interface to deal with parallel linear solves through the
 *  expression library, but Uintah has a reasonably robust interface.
 *
 *  This expression does play well with expression graphs, however.
 *  There are only a few places where Uintah reaches in.
 *
 *  Because of the hackery going on here, this expression is placed in
 *  the Wasatch namespace.  This should reinforce the concept that it
 *  is not intended for external use.
 */
class RadiationSource
 : public Expr::Expression<SVolField>
{

  const Expr::Tag temperatureTag_, absorptionTag_, celltypeTag_;

  const Uintah::VarLabel *temperatureLabel_, *absorptionLabel_, *celltypeLabel_, *divqLabel_, *VRFluxLabel_,
  *boundFluxLabel_, *radiationVolqLabel_;
  
  const SVolField* divQ_;
  Uintah::Ray* rmcrt_;
  void schedule_setup_bndflux( const Uintah::LevelP& level,
                              Uintah::SchedulerP sched,
                              const Uintah::MaterialSet* const materials );

  RadiationSource( const std::string& RadiationSourceName,
                   const Expr::Tag& temperatureTag,
                   const Expr::Tag& absorptionTag,
                   const Expr::Tag& celltypeTag,
                   const Uintah::ProblemSpecP& radiationSpec,
                   Uintah::SimulationStateP sharedState,
                   Uintah::GridP grid);

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag temperatureTag_, absorptionTag_, celltypeTag_;
    Uintah::ProblemSpecP     radiationSpec_;
    Uintah::SimulationStateP sharedState_;
    Uintah::GridP            grid_;
    
  public:
    Builder( const Expr::TagList& results,
             const Expr::Tag& temperatureTag,
             const Expr::Tag& absorptionTag,
             const Expr::Tag& celltypeTag,
             Uintah::ProblemSpecP& radiationSpec,
             Uintah::SimulationStateP& sharedState,
             Uintah::GridP& grid);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  };

  ~RadiationSource();

  /**
   *  \brief allows Wasatch::TaskInterface to reach in and give this
   *         expression the information requried to schedule the
   *         linear solver.
   */
  void schedule_ray_tracing( const Uintah::LevelP& level,
                             Uintah::SchedulerP sched,
                             const Uintah::MaterialSet* const materials,
                             const int RKStage );

 
  /**
   *  \brief allows Wasatch::TaskInterface to reach in and provide
   *         this expression with a way to set the variables that it
   *         needs to.
   */
  void declare_uintah_vars( Uintah::Task& task,
                            const Uintah::PatchSubset* const patches,
                            const Uintah::MaterialSubset* const materials,
                            const int RKStage );

  
  /**
   *  \brief allows Wasatch::TaskInterface to reach in and provide
   *         this expression with a way to retrieve Uintah-specific
   *         variables from the data warehouse.
   *
   *  This should be done very carefully.  Any "external" dependencies
   *  should not be introduced here.  This is only for variables that
   *  are very uintah-specific and only used internally to this
   *  expression.  Specifically, the RadiationSource-rhs field and the LHS
   *  matrix.  All other variables should be expressed as dependencies
   *  through the advertise_dependents method.
   */
  void bind_uintah_vars( Uintah::DataWarehouse* const dw,
                         const Uintah::Patch* const patch,
                         const int material,
                         const int RKStage );

  void setup_bndflux( const Uintah::ProcessorGroup* const pg,
                      const Uintah::PatchSubset* const patches,
                      const Uintah::MaterialSubset* const materials,
                      Uintah::DataWarehouse* const oldDW,
                      Uintah::DataWarehouse* const newDW );

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};

} // namespace Wasatch

#endif // RadiationSource_Expr_h

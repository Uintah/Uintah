/*
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

#ifndef Poisson_Eq_h
#define Poisson_Eq_h

//-- ExprLib Includes --//
#include <expression/Expression.h>

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Operators/Operators.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

//-- Uintah Includes --//
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/Stencil4.h>
#include <Core/Grid/Variables/CCVariable.h>

namespace Uintah{
  class SolverInterface;
  class SolverParameters;
}

namespace Wasatch{
  
  /**
   *  \class 	 PoissonExpression
   *  \ingroup Expressions
   *  \ingroup	WasatchCore
   *  \author 	Tony Saad
   *  \date 	  June, 2011
   *
   *  \brief Expression to solve a poisson equation \f$ \nabla^2 \phi = S \f$.
   *
   *  NOTE: this expression BREAKS WITH CONVENTION!  Notably, it has
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
  class PoissonExpression
  : public Expr::Expression<SVolField>
  {
    const Expr::Tag phit_;
    const Expr::Tag phirhslocalt_;
    const Expr::Tag phirhst_;
       
    const bool doX_, doY_, doZ_;
    bool didAllocateMatrix_;
    int  materialID_;
    int  rkStage_;
    const bool useRefPhi_;
    const double refPhiValue_;
    const SCIRun::IntVector refPhiLocation_;
    const bool use3DLaplacian_;
    
    const Uintah::SolverParameters& solverParams_;
    Uintah::SolverInterface& solver_;
    const Uintah::VarLabel* matrixLabel_;
    const Uintah::VarLabel* phiLabel_;
    const Uintah::VarLabel* phirhsLabel_;
    
    const double* timestep_;
    
    const SVolField* phirhs_;
        
    typedef Uintah::CCVariable<Uintah::Stencil4> MatType;
    MatType matrix_;
    const Uintah::Patch* patch_;
    // NOTE that this expression computes a rhs locally. We will need to modify 
    // the RHS of this expression due to boundary conditions hence we need a 
    // locally computed field.
    PoissonExpression( const Expr::Tag& phitag,
                       const Expr::Tag& phirhslocaltag, 
                       const Expr::Tag& phirhstag,
                       const bool       useRefPhi,
                       const double     refPhiValue,
                       const SCIRun::IntVector refPhiLocation,
                       const bool       use3dlaplacian,
                       const Uintah::SolverParameters& solverParams,
                       Uintah::SolverInterface& solver );
    
  public:  

    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::Tag phirhst_;
      const bool userefphi_;
      const double refphivalue_;
      const SCIRun::IntVector refphilocation_;
      const bool use3dlaplacian_;
      const Uintah::SolverParameters& sparams_;
      Uintah::SolverInterface& solver_;
    public:
      Builder( const Expr::TagList& results,
              const Expr::Tag& phirhst,
              const bool       useRefPhi,
              const double     refPhiValue,
              const SCIRun::IntVector refPhiLocation,
              const bool       use3DLaplacian,            
              const Uintah::SolverParameters& sparams,
              Uintah::SolverInterface& solver );
      ~Builder(){}
      Expr::ExpressionBase* build() const;
    };
    
    ~PoissonExpression();
    static Expr::TagList poissonTagList;
    
    /**
     *  \brief Allows Wasatch::TaskInterface to reach in and give this
     *         expression the information requried to schedule the
     *         linear solver.
     */
    void schedule_solver( const Uintah::LevelP& level,
                         Uintah::SchedulerP sched,
                         const Uintah::MaterialSet* const materials,
                         const int RKStage, const bool isDoingInitialization );
    
    /**
     *  \brief Allows Wasatch::TaskInterface to reach in set the boundary conditions
     on PoissonExpression at the appropriate time - namely after the linear solve. 
     */  
    // NOTE: Maybe we should not expose this to the outside?
    void schedule_set_poisson_bcs( const Uintah::LevelP& level,
                                   Uintah::SchedulerP sched,
                                   const Uintah::MaterialSet* const materials,
                                   const int RKStage );
    
    /**
     *  \brief Allows Wasatch::TaskInterface to reach in and provide
     *         this expression with a way to set the variables that it
     *         needs to.
     */
    void declare_uintah_vars( Uintah::Task& task,
                             const Uintah::PatchSubset* const patches,
                             const Uintah::MaterialSubset* const materials,
                             const int RKStage );
    
    /**
     *  \brief Save pointer to the patch associated with this expression. This
    *          is needed to set boundary conditions and extract other mesh info.
     */    
    void set_patch( const Uintah::Patch* const patch ){ patch_ = const_cast<Uintah::Patch*> (patch); }
    
    /**
     *  \brief set the RKStage for the current PoissonExpression evaluation. We need this to
     reduce the number of PoissonExpression-solve iterations in the 2nd and 3rd
     stages of the RK3SSP integrator. Since these subsequent RK stages
     use the guess PoissonExpression from the newDW, then we should NOT initialize
     the PoissonExpression in the new DW to zero for those stages.
     */  
    void set_RKStage( const int RKStage ){ rkStage_ = RKStage; }
    
    /**
     *  \brief allows Wasatch::TaskInterface to reach in and provide
     *         this expression with a way to retrieve Uintah-specific
     *         variables from the data warehouse.
     *
     *  This should be done very carefully.  Any "external" dependencies
     *  should not be introduced here.  This is only for variables that
     *  are very uintah-specific and only used internally to this
     *  expression.  Specifically, the PoissonExpression-rhs field and the LHS
     *  matrix.  All other variables should be expressed as dependencies
     *  through the advertise_dependents method.
     */
    void bind_uintah_vars( Uintah::DataWarehouse* const dw,
                          const Uintah::Patch* const patch,
                          const int material,
                          const int RKStage );
    /**
     * \brief Calculates PoissonExpression coefficient matrix.
     */
    void setup_matrix();
    
    /**
     * \brief Special function to apply PoissonExpression boundary conditions after the PoissonExpression solve.
     *        This is needed because process_after_evaluate is executed before the PoissonExpression solve.
     *        We may need to split the PoissonExpression expression into a PoissonExpression_rhs and a PoissonExpression...
     */  
    void process_bcs ( const Uintah::ProcessorGroup* const pg,
                      const Uintah::PatchSubset* const patches,
                      const Uintah::MaterialSubset* const materials,
                      Uintah::DataWarehouse* const oldDW,
                      Uintah::DataWarehouse* const newDW);
    
    //Uintah::CCVariable<Uintah::Stencil7> PoissonExpression_matrix(){ return matrix_ ;}
    void advertise_dependents( Expr::ExprDeps& exprDeps );
    void bind_fields( const Expr::FieldManagerList& fml );
    void evaluate();
    
  };
} // namespace Wasatch

#endif // Poisson_Eq_h

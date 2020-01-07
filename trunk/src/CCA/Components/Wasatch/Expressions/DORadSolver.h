/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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

#ifndef Wasatch_Discrete_Ordinates_h
#define Wasatch_Discrete_Ordinates_h

//-- ExprLib Includes --//
#include <expression/Expression.h>

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

//-- Uintah Includes --//
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/CCVariable.h>

namespace Uintah{
  class SolverInterface;
}

//==============================================================================

/**
 *  \class  OrdinateDirections
 *  \date   May 15, 2014
 *  \author "James C. Sutherland"
 *
 *  This class holds information on the quadrature for discrete ordinates.
 *  For a given approximation level (Sn), this provides all of the ordinate
 *  directions as well as quadrature weights.
 */
class OrdinateDirections
{
public:

  /**
   * \brief Holds information about a single ordinate direction.
   */
  struct SVec{
    SVec( const double x, const double y, const double z, const double w );

    double x,y,z;  ///< Cartesian components of the ordinate direction vector
    double w;      ///< quadrature weight
  };

  /**
   * Construct an OrdinateDirections object.
   * @param n The order (Sn).  Supports 2, 4, 6, 8.
   */
  OrdinateDirections( const int n );

  ~OrdinateDirections();

  /**
   * Obtain the information for the ith ordinate direction.
   */
  inline const SVec& get_ordinate_information( const int i ) const{ return ordinates_[i]; }

  /**
   * Obtain all of the ordinate information.
   */
  inline const std::vector<SVec>& get_ordinate_information() const{ return ordinates_; }

  inline size_t number_of_directions() const{ return ordinates_.size(); }

private:
  std::vector<SVec> ordinates_;
};

//==============================================================================

namespace WasatchCore{
  
  /**
   *  \class   DORadSolver
   *  \ingroup Expressions
   *  \author 	James C. Sutherland
   *  \date 	October, 2014
   *  \brief Solves a system for the Discrete Ordinates (DO) radiation model.
   *
   *  The DO model requires solution of a number of linear equations to obtain
   *  the radiative intensity along each ordinate direction.  This expression
   *  solves for the intensity in ONE of these directions. The DORadSrc
   *  expression combines these to obtain the divergence of the heat flux.
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
  class DORadSolver
  : public Expr::Expression<SVolField>
  {
    const OrdinateDirections::SVec svec_;
    const bool hasAbsCoef_, hasScatCoef_;

    const Expr::Tag temperatureTag_;
    
    const bool doX_, doY_, doZ_;
    bool didAllocateMatrix_;
    int  materialID_;
    int  rkStage_;
    

    Uintah::SolverInterface& solver_;
    const Uintah::VarLabel *matrixLabel_, *intensityLabel_, *rhsLabel_;
    
    DECLARE_FIELDS(SVolField, temperature_, absCoef_, scatCoef_)
    
    typedef Uintah::CCVariable<Uintah::Stencil7> MatType;
    MatType matrix_;
    const Uintah::Patch* patch_;

    // NOTE that this expression computes a rhs locally. We will need to modify 
    // the RHS of this expression due to boundary conditions hence we need a 
    // locally computed field.
    DORadSolver( const std::string intensityName,
                 const std::string intensityRHSName,
                 const OrdinateDirections::SVec& svec,
                 const Expr::Tag& absCoefTag,
                 const Expr::Tag& scatCoefTag,
                 const Expr::Tag& temperatureTag,
                 Uintah::SolverInterface& solver );
    
  public:

    static Expr::TagList intensityTags;

    class Builder : public Expr::ExpressionBuilder
    {
      const Expr::Tag absCoefTag_, scatCoefTag_, temperatureTag_;
      Uintah::SolverInterface& solver_;
      const OrdinateDirections::SVec svec_;
    public:
      Builder( const std::string intensityName,
               const OrdinateDirections::SVec& svec,
               const Expr::Tag absCoefTag,
               const Expr::Tag scatCoefTag,
               const Expr::Tag temperatureTag,
               Uintah::SolverInterface& solver );
      ~Builder(){}
      Expr::ExpressionBase* build() const;
    };
    
    ~DORadSolver();
    
    /**
     *  \brief Allows WasatchCore::TaskInterface to reach in and give this
     *         expression the information required to schedule the
     *         linear solver.
     */
    void schedule_solver( const Uintah::LevelP& level,
                          Uintah::SchedulerP sched,
                          const Uintah::MaterialSet* const materials,
                          const int rkStage,
                          const bool isDoingInitialization );
    
    /**
     *  \brief Allows WasatchCore::TaskInterface to reach in and provide
     *         this expression with a way to set the variables that it
     *         needs to.
     */
    void declare_uintah_vars( Uintah::Task& task,
                              const Uintah::PatchSubset* const patches,
                              const Uintah::MaterialSubset* const materials,
                              const int rkStage );
    
    /**
     *  \brief allows WasatchCore::TaskInterface to reach in and provide
     *         this expression with a way to retrieve Uintah-specific
     *         variables from the data warehouse.
     *
     *  This should be done very carefully.  Any "external" dependencies
     *  should not be introduced here.  This is only for variables that
     *  are very uintah-specific and only used internally to this
     *  expression.  Specifically, the rhs field and the LHS matrix.
     *  All other variables should be expressed as dependencies
     *  through the declare_field macro.
     */
    void bind_uintah_vars( Uintah::DataWarehouse* const dw,
                           const Uintah::Patch* const patch,
                           const int material,
                           const int rkStage );
    /**
     * \brief Calculates DORadSolver coefficient matrix.
     */
    void setup_matrix( SVolField& rhs, const SVolField& temperature );
    void evaluate();
  };


  /**
   *  \class DORadSrc
   *  \author James C. Sutherland
   *  \date   October, 2014
   *
   *  \brief Calculates the radiative source term, \f$\nabla\cdot\mathbf{q}^\mathrm{rad}\f$,
   *   from the discrete ordinates model.
   */
  class DORadSrc
   : public Expr::Expression<SVolField>
  {
    const Expr::Tag temperatureTag_, absCoefTag_;
    const OrdinateDirections& ord_;
    const bool hasAbsCoef_;

    DECLARE_FIELDS(SVolField, temperature_, absCoef_)
    DECLARE_VECTOR_OF_FIELDS(SVolField, intensity_)

    DORadSrc( const Expr::Tag& temperatureTag,
              const Expr::Tag& absCoefTag,
              const OrdinateDirections& ord );

  public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
      /**
       *  @brief Build a DORadSrc expression
       *  @param divQTag the tag for the divergence of the heat flux
       *  @param temperatureTag the temperature
       *  @param absCoefTag the absorption coefficient (empty for unity)
       *  @param ord the OrdinateDirections information
       */
      Builder( const Expr::Tag divQTag,
               const Expr::Tag temperatureTag,
               const Expr::Tag absCoefTag,
               const OrdinateDirections& ord );

      Expr::ExpressionBase* build() const;

    private:
      const Expr::Tag temperatureTag_, absCoefTag_;
      const OrdinateDirections ord_;
    };

    ~DORadSrc();
    void evaluate();
  };

} // namespace WasatchCore

#endif // Wasatch_Discrete_Ordinates_h

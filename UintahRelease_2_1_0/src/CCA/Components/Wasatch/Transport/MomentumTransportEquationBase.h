/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

#ifndef Wasatch_MomentumTransportEquationBase_h
#define Wasatch_MomentumTransportEquationBase_h

//-- ExprLib includes --//
#include <expression/ExpressionFactory.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>
#include <CCA/Components/Wasatch/Transport/TransportEquation.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>

//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SolverInterface.h>

namespace WasatchCore{
  
  
  template< typename FieldT >
  Expr::ExpressionID
  register_strain_tensor(const Direction momComponent,
                         const bool* const doMom,
                         const bool isViscous,
                         const Expr::TagList& velTags,
                         Expr::TagList& strainTags,
                         const Expr::Tag& dilTag,
                         Expr::ExpressionFactory& factory );


  template< typename FaceFieldT >
  Expr::ExpressionID
  setup_strain( const Expr::Tag& strainTag,
                const Expr::Tag& vel1Tag,
                const Expr::Tag& vel2Tag,
                const Expr::Tag& dilTag,
                Expr::ExpressionFactory& factory );

  void register_turbulence_expressions( const TurbulenceParameters& turbParams,
                                        Expr::ExpressionFactory& factory,
                                        const Expr::TagList& velTags,
                                        const Expr::Tag densTag,
                                        const bool isConstDensity );
  
  template< typename FluxT, typename AdvelT >
  Expr::ExpressionID
  setup_momentum_convective_flux( const Expr::Tag& fluxTag,
                                  const Expr::Tag& momTag,
                                  const Expr::Tag& advelTag,
                                  ConvInterpMethods convInterpMethod,
                                  const Expr::Tag& volFracTag,
                                  Expr::ExpressionFactory& factory );

  template< typename FieldT >
  Expr::ExpressionID
  register_momentum_convective_fluxes( const Direction momComponent,
                                       const bool* const doMom,
                                       const Expr::TagList& velTags,
                                       Expr::TagList& cfTags,
                                       ConvInterpMethods convInterpMethod,
                                       const Expr::Tag& momTag,
                                       const Expr::Tag& volFracTag,
                                       Expr::ExpressionFactory& factory );
  
  Expr::Tag mom_tag( const std::string& momName, const bool old=false );

  Expr::Tag rhs_part_tag( const Expr::Tag& momTag );
  
  Expr::Tag rhs_part_tag( const std::string& momName );
  
  void set_vel_tags( Uintah::ProblemSpecP params,
                     Expr::TagList& velTags );
  
  void set_mom_tags( Uintah::ProblemSpecP params,
                     Expr::TagList& momTags,
                     const bool old=false);

  bool is_normal_to_boundary( const Direction stagLoc,
                              const Uintah::Patch::FaceType face);
  /**
   *  \ingroup WasatchCore
   *  \class MomentumTransportEquationBase
   *  \authors James C. Sutherland, Tony Saad
   *  \date January, 2011
   *
   *  \brief Creates a momentum transport equation base class
   *
   *  \todo Allow more flexibility in specifying initial and boundary conditions for momentum.
   */
  template< typename FieldT >
  class MomentumTransportEquationBase : public WasatchCore::TransportEquation
  {
  public:

    typedef typename FaceTypes<FieldT>::XFace  XFace; ///< The type of field on the x-faces of the volume.
    typedef typename FaceTypes<FieldT>::YFace  YFace; ///< The type of field on the y-faces of the volume.
    typedef typename FaceTypes<FieldT>::ZFace  ZFace; ///< The type of field on the z-faces of the volume.

    /**
     *  \brief Construct a MomentumTransportEquationBase
     *  \param momComponent the direction of this component of momentum
     *  \param velName the name of the velocity component solved by this MomentumTransportEquationBase
     *  \param momName the name of the momentum component solved by this MomentumTransportEquationBase
     *  \param densTag the tag for the mixture mass density
     *  \param isConstDensity
     *  \param bodyForceTag tag for body force
     *  \param srcTermTag if not empty, this specifies a tag for an expression to add as a momentum source.
     *  \param gc
     *  \param params Parser information for this momentum equation
     *  \param turbulenceParams
     */
    MomentumTransportEquationBase( const Direction momComponent,
                                   const std::string velName,
                                   const std::string momName,
                                   const Expr::Tag densTag,
                                   const bool isConstDensity,
                                   const Expr::Tag bodyForceTag,
                                   const Expr::Tag srcTermTag,
                                   GraphCategories& gc,
                                   Uintah::ProblemSpecP params,
                                   TurbulenceParameters turbulenceParams );

    ~MomentumTransportEquationBase();

    void setup_boundary_conditions( WasatchBCHelper& bcHelper,
                                    GraphCategories& graphCat)
    {}
    
    /**
     *  \brief apply the boundary conditions on the initial condition
     *         associated with this transport equation
     */
    void apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                            WasatchBCHelper& bcHelper )
    {}

    /**
     *  \brief setup the boundary conditions associated with this momentum equation
     */
    void apply_boundary_conditions( const GraphHelper& graphHelper,
                                    WasatchBCHelper& bcHelper )
    {}
    
    /**
     *  \brief setup the initial conditions for this momentum equation.
     */
    virtual Expr::ExpressionID initial_condition( Expr::ExpressionFactory& exprFactory ) = 0;
    
    /**
     *  \brief Parse the input file to get the name of this MomentumTransportEquationBase
     *
     *  \param params the Uintah::ProblemSpec XML description for this
     *         equation. Scope should be within the TransportEquation tag.
     */
    static std::string get_phi_name( Uintah::ProblemSpecP params );

  protected:

    void setup_diffusive_flux( FieldTagInfo& ){}
    void setup_convective_flux( FieldTagInfo& ){}
    void setup_source_terms( FieldTagInfo&, Expr::TagList& ){}                        
    
    virtual Expr::ExpressionID setup_rhs( FieldTagInfo& info,
                                          const Expr::TagList& srcTags ) = 0;
    
    const Direction momComponent_;
    Uintah::ProblemSpecP params_;
    const bool isViscous_, isTurbulent_;
    const Expr::Tag thisVelTag_, densityTag_;
    const Expr::Tag pressureTag_;
    
    Expr::ExpressionID normalStrainID_, normalConvFluxID_, pressureID_;
    Expr::TagList velTags_;  ///< TagList for the velocity expressions
    Expr::TagList momTags_, oldMomTags_;  ///< TagList for the momentum expressions
    Expr::Tag     thisVolFracTag_;
    Expr::Tag     normalStrainTag_, normalConvFluxTag_;

  private:

  };

} // namespace WasatchCore

#endif // Wasatch_MomentumTransportEquationBase_h

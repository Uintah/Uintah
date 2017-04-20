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

#ifndef Wasatch_PseudospeciesTransport_h
#define Wasatch_PseudospeciesTransport_h

#include <sci_defs/wasatch_defs.h>

#ifndef HAVE_POKITT
// kill compilation if we don't have pokitt.
#  error Species transport requires PoKiTT.
#endif

//-- ExprLib includes --//
#include <CCA/Components/Wasatch/Transport/TransportEquation.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulentDiffusivity.h>

namespace WasatchCore{

  /**
   *  \ingroup WasatchCore
   *  \class PseudospeciesTransportEquation
   *  \authors Josh McConnell
   *  \date November, 2016
   *
   *  \brief Support for a generic transport equation for a pseudospecies (tar, soot, etc.)
   */
  template<typename FieldT>
  class PseudospeciesTransportEquation : public WasatchCore::TransportEquation
  {
  public:

    // these typedefs are provided for convenience.
    typedef typename FaceTypes<FieldT>::XFace  XFaceT; ///< The type of field on the x-faces of the volume.
    typedef typename FaceTypes<FieldT>::YFace  YFaceT; ///< The type of field on the y-faces of the volume.
    typedef typename FaceTypes<FieldT>::ZFace  ZFaceT; ///< The type of field on the z-faces of the volume.

    /**
     *  \brief Construct a PseudospeciesTransportEquation
     *  \param pseudospeciesName the name of the pseudospecies for this PseudospeciesTransportEquation
     *  \param params the tag from the input file for the entire "wasatch" block of the input file
     *  \param gc
     *  \param densityTag a tag containing density for necessary cases. it will be empty where
     *         it is not needed.
     *  \param turbulenceParams information on turbulence models
     *  \param callSetup for objects that derive from PseudospeciesTransportEquation,
     *         this flag should be set to false, and those objects should call
     *         setup() at the end of their constructor.
     *
     *  Note that the static member methods get_rhs_expr_id,
     *  get_primvar_name and get_solnvar_name can be useful
     *  to obtain the appropriate input arguments here.
     */
    PseudospeciesTransportEquation( const std::string pseudospeciesName,
                                    Uintah::ProblemSpecP params,
                                    GraphCategories& gc,
                                    const Expr::Tag densityTag,
                                    const TurbulenceParameters& turbulenceParams,
                                    const bool callSetup=true );

    virtual ~PseudospeciesTransportEquation();

    /**
     *  \brief Used to check the validity of the boundary conditions specified
     *   by the user at a given boundary and also to infer/add new BCs on the
     *   type of boundary.  Example: at a stationary impermeable wall, we can
     *   immediately infer zero-velocity boundary conditions and check whether
     *   the user has specified any velocity BCs at that boundary. See examples
     *   in the momentum transport equation.
     */
    void setup_boundary_conditions( WasatchBCHelper& bcHelper,
                                    GraphCategories& graphCat );

    /**
     *  \brief setup the boundary conditions associated with this transport equation
     */
    void apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                            WasatchBCHelper& bcHelper );

    /**
     *  \brief setup the boundary conditions associated with this transport equation
     */
    void apply_boundary_conditions( const GraphHelper& graphHelper,
                                    WasatchBCHelper& bcHelper );

    /**
     *  \brief setup the initial conditions for this transport equation.
     */
    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& icFactory );

    std::string pseudospecies_name() { return pseudoSpecName_; };

    bool is_weak_form() const{ return !isStrong_; }
    bool is_strong_form() const{ return isStrong_; }

  protected:
    virtual void setup_diffusive_flux( FieldTagInfo& ) = 0;
    virtual void setup_convective_flux( FieldTagInfo& );
    virtual void setup_source_terms( FieldTagInfo&, Expr::TagList& ) = 0;
    virtual Expr::ExpressionID setup_rhs( FieldTagInfo&,
                                          const Expr::TagList& srcTags  );
    Expr::Tag get_species_rhs_tag( std::string name );

    Uintah::ProblemSpecP params_, psParams_;
    const std::string solnVarName_, pseudoSpecName_;
    const Expr::Tag densityTag_, primVarTag_;
    const bool enableTurbulence_;
    bool isStrong_, isConstDensity_;
    Expr::Tag turbDiffTag_;
    FieldTagInfo infoStar_;  // needed to form predicted scalar quantities
  };

} // namespace WasatchCore

#endif // Wasatch_PseudospeciesTransport_h

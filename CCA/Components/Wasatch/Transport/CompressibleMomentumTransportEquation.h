/*
 * The MIT License
 *
 * Copyright (c) 2015 The University of Utah
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

/**
 *  \file   CompressibleMomentumTransportEquation.h
 *  \date   Nov 20, 2015
 *  \author James C. Sutherland
 */

#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Transport/TransportEquation.h>
#include <CCA/Components/Wasatch/Transport/MomentumTransportEquationBase.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>

namespace WasatchCore{

  /**
   * \class CompressibleMomentumTransportEquation
   * \date November, 2015
   *
   * \brief Construct a compressible momentum transport equation.
   *
   * \notes:
   *   - there are many tools in the low-mach momentum equation that should be shared between them.
   */
  class CompressibleMomentumTransportEquation : public WasatchCore::MomentumTransportEquationBase<SVolField>
  {
    typedef SpatialOps::SVolField FieldT;

  public:
    CompressibleMomentumTransportEquation( const Direction momComponent,
                                           const std::string velName,
                                           const std::string momName,
                                           const Expr::Tag densityTag,
                                           const Expr::Tag temperatureTag,
                                           const Expr::Tag mixMWTag,
                                           const double gasConstant,
                                           const Expr::Tag bodyForceTag,
                                           const Expr::Tag srcTermTag,
                                           GraphCategories& gc,
                                           Uintah::ProblemSpecP params,
                                           TurbulenceParameters turbParams );

    ~CompressibleMomentumTransportEquation();

    void setup_boundary_conditions( WasatchBCHelper& bcHelper,
                                    GraphCategories& graphCat )
    {
      assert(false);  // not ready
    }

    void apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                            WasatchBCHelper& bcHelper )
    {
      assert(false); // not ready
    }


    void apply_boundary_conditions( const GraphHelper& graphHelper,
                                    WasatchBCHelper& bcHelper )
    {
      assert( false );  // not ready
    }

    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& icFactory )
    {
      // jcs can share the initial condition expression with the low-mach momentum equation
      assert(false);  // not ready
    }

  protected:

    void setup_diffusive_flux( FieldTagInfo& ){ assert(false); }
    void setup_convective_flux( FieldTagInfo& ){ assert(false); }
    void setup_source_terms( FieldTagInfo&, Expr::TagList& ){ assert(false); }
    Expr::ExpressionID setup_rhs( FieldTagInfo& info, const Expr::TagList& srcTags ){ assert(false); }

  };

}




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

#ifndef Wasatch_TemperatureTransportEquation_h
#define Wasatch_TemperatureTransportEquation_h


//-- ExprLib includes --//
#include <CCA/Components/Wasatch/transport/TransportEquation.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/BCHelperTools.h>


namespace Expr{
  class ExpressionFactory;
  class ExpressionID;
}


namespace Wasatch{

  /**
   *  \class TemperatureTransportEquation
   *  \author James C. Sutherland
   *  \date June, 2010
   *
   *  \brief A basic transport equation for temperature.
   *
   */
  class TemperatureTransportEquation : public Wasatch::TransportEquation
  {
  public:

    // define the types of fields that this TransportEquation deals with.
    typedef SVolField            FieldT;
    typedef FaceTypes<FieldT>::XFace  XFaceT;
    typedef FaceTypes<FieldT>::YFace  YFaceT;
    typedef FaceTypes<FieldT>::ZFace  ZFaceT;

    TemperatureTransportEquation( Expr::ExpressionFactory& solnExprFactory );

    ~TemperatureTransportEquation();

    /**
     *  Set up the boundary condition on initial conditions evaluators for this
     *  TransportEquation. Each derived class must implement this
     *  method.  Boundary conditions are imposed by adding additional
     *  tasks to modify values in an Expression after it is evaluated.
     *  This is done by attaching a functor to the applicable expression
     *  via the <code>Expression::process_after_evaluate</code> method.
     */
    void setup_initial_boundary_conditions( const GraphHelper& graphHelper,
                                                   const Uintah::PatchSet* const localPatches,
                                                   const PatchInfoMap& patchInfoMap,
                                                   const Uintah::MaterialSubset* const materials);


    void setup_boundary_conditions( const GraphHelper& graphHelper,
                                   const Uintah::PatchSet* const localPatches,
                                   const PatchInfoMap& patchInfoMap,
                                   const Uintah::MaterialSubset* const materials);

    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& icFactory );

  private:

    static Expr::ExpressionID get_rhs_id( Expr::ExpressionFactory& );

  };

} // namespace Wasatch


#endif // Wasatch_TemperatureTransportEquation_h

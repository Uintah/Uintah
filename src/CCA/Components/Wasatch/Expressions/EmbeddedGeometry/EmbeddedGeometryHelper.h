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

#ifndef Wasatch_EmbeddedGeometryHelper_h
#define Wasatch_EmbeddedGeometryHelper_h

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <expression/Expression.h>

#include <CCA/Components/Wasatch/GraphHelperTools.h>
#include <CCA/Components/Wasatch/BCHelper.h>
#include <CCA/Components/Wasatch/WasatchBCHelper.h>
#include <Core/Exceptions/InvalidState.h>

/**
 *  \file BasicExprBuilder.h
 *  \brief parser support for creating embedded geometry.
 */
namespace Expr{
  class ExpressionBuilder;
}

namespace WasatchCore{
  
  class EmbeddedGeometryHelper
  {
  public:
    /**
     *  Access the volume fraction names and tags.
     */
    static EmbeddedGeometryHelper& self();
    
    template <typename FieldT>
    Expr::Tag vol_frac_tag() const;
    
    void set_vol_frac_names(const std::string& svolfrac, const std::string& xvolfrac, const std::string& yvolfrac, const std::string& zvolfrac)
    {
      svolfrac_ = svolfrac;
      xvolfrac_ = xvolfrac;
      yvolfrac_ = yvolfrac;
      zvolfrac_ = zvolfrac;
    }
    
    void set_has_moving_geometry(const bool hasMovingGeometry)   { hasMovingGeometry_ = hasMovingGeometry; }
    bool has_moving_geometry() const { return hasMovingGeometry_; }
    
    void set_has_embedded_geometry( const bool hasEmbeddedGeometry) { hasEmbeddedGeometry_ = hasEmbeddedGeometry; }
    bool has_embedded_geometry() const { return hasEmbeddedGeometry_;}
    
    void set_state( const bool done ) { doneSetup_ = done; }
    
  private:
    std::string svolfrac_, xvolfrac_,yvolfrac_,zvolfrac_;
    bool hasEmbeddedGeometry_, hasMovingGeometry_, doneSetup_;
    EmbeddedGeometryHelper();
    
    void check_state() const
    {
      if (!doneSetup_) {
        std::ostringstream msg;
        msg << "ERROR: Trying to access embedded geometry information before it was parsed." << std::endl;
        throw Uintah::InvalidState( msg.str(), __FILE__, __LINE__ );
      }
    }

  };

  /**
   *  \addtogroup WasatchParser
   *  \addtogroup Expressions
   *
   *  \brief Creates expressions for embedded geometry
   *
   *  \param parser - the Uintah::ProblemSpec block that contains \verbatim <EmbeddedGeometry> \endverbatim specification.
   *  \param gc - the GraphCategories object that the embedded geometry should be associated with.
   */
  void
  parse_embedded_geometry( Uintah::ProblemSpecP parser,
                           GraphCategories& gc );
  
  void
  apply_intrusion_boundary_conditions(WasatchBCHelper& bcHelper);

} // namespace WasatchCore


#endif // Wasatch_EmbeddedGeometryHelper_h

/*
 * The MIT License
 *
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

#ifndef Wasatch_EmbeddedGeometryHelper_h
#define Wasatch_EmbeddedGeometryHelper_h

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <expression/Expression.h>

#include <CCA/Components/Wasatch/GraphHelperTools.h>

/**
 *  \file BasicExprBuilder.h
 *  \brief parser support for creating embedded geometry.
 */
namespace Expr{
  class ExpressionBuilder;
}

namespace Wasatch{
  
  class VolFractionNames
  {
  public:
    
    /**
     *  Access the VolumeFraction names and tags.
     */
    static VolFractionNames& self();
    
    Expr::Tag svol_frac_tag() {return Expr::Tag(svolfrac_,Expr::STATE_NONE);}
    Expr::Tag xvol_frac_tag() {return Expr::Tag(xvolfrac_,Expr::STATE_NONE);}
    Expr::Tag yvol_frac_tag() {return Expr::Tag(yvolfrac_,Expr::STATE_NONE);}
    Expr::Tag zvol_frac_tag() {return Expr::Tag(zvolfrac_,Expr::STATE_NONE);}
    
    void set_svol_frac_name(const std::string svolfrac){svolfrac_ = svolfrac;}
    void set_xvol_frac_name(const std::string xvolfrac){xvolfrac_ = xvolfrac;}
    void set_yvol_frac_name(const std::string yvolfrac){yvolfrac_ = yvolfrac;}
    void set_zvol_frac_name(const std::string zvolfrac){zvolfrac_ = zvolfrac;}
    
  private:
    std::string svolfrac_, xvolfrac_,yvolfrac_,zvolfrac_;
    VolFractionNames();
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
  
} // namespace Wasatch


#endif // Wasatch_EmbeddedGeometryHelper_h

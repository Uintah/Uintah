/*
 * The MIT License
 *
 * Copyright (c) 2012-2014 The University of Utah
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

#include <CCA/Components/Wasatch/Transport/ParticleEquationBase.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/EmbeddedGeometryHelper.h>

namespace Wasatch{

  //---------------------------------------------------------------------------

  ParticleEquationBase::
  ParticleEquationBase( const std::string& solnVarName,
                       const ParticleDirection pDir,
                       const Expr::TagList& particlePositionTags,
                       const Expr::Tag& particleSizeTag,
                       Uintah::ProblemSpecP particleEqsSpec,
                       GraphCategories& gc )
  : EquationBase::EquationBase(gc, solnVarName, particleEqsSpec),
    pDir_(pDir),
    pDirName_(particle_direction_to_string(pDir_)),
    pPosTags_(particlePositionTags),
    pSizeTag_(particleSizeTag)
  {}

  //---------------------------------------------------------------------------

  ParticleEquationBase::ParticleDirection string_to_particle_direction(const std::string& dirname)
  {
    if      (dirname=="X"  ) return ParticleEquationBase::XDIR;
    else if (dirname == "Y") return ParticleEquationBase::YDIR;
    else if (dirname == "Z") return ParticleEquationBase::ZDIR;
    else                     return ParticleEquationBase::NODIR;
  }

  //---------------------------------------------------------------------------
  
  std::string particle_direction_to_string(const ParticleEquationBase::ParticleDirection dir)
  {
    std::string strDir ="";
    switch (dir) {
      case XDIR:
        strDir = "X";
        break;
      case YDIR:
        strDir = "Y";
        break;
      case ZDIR:
        strDir = "Z";
        break;
      default:
        break;
    }
    return strDir;
  }

} // namespace Wasatch

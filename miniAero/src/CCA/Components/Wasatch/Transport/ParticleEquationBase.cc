/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
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
#include <CCA/Components/Wasatch/ParticlesHelper.h>
#include <CCA/Components/Wasatch/Transport/ParticleEquationBase.h>
#include <CCA/Components/Wasatch/Expressions/EmbeddedGeometry/EmbeddedGeometryHelper.h>

namespace Wasatch{

  //---------------------------------------------------------------------------

  ParticleEquationBase::
  ParticleEquationBase( const std::string& solnVarName,
                       const Direction pDir,
                       const Expr::TagList& particlePositionTags,
                       const Expr::Tag& particleSizeTag,
                       Uintah::ProblemSpecP particleEqsSpec,
                       GraphCategories& gc )
  : EquationBase::EquationBase(gc, solnVarName, pDir, particleEqsSpec),
    pPosTags_(particlePositionTags),
    pSizeTag_(particleSizeTag)
  {
    Uintah::ParticlesHelper::needs_boundary_condition(solnVarName);
  }

} // namespace Wasatch

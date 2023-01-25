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

#ifndef Wasatch_ParticleEquationBase_h
#define Wasatch_ParticleEquationBase_h

#include <string>

//-- ExprLib includes --//
#include <expression/Tag.h>
#include <expression/ExpressionFactory.h>
#include <expression/ExpressionID.h>
#include <expression/Context.h>

//-- Uintah framework includes --//
#include <Core/Grid/Patch.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/PatchInfo.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/Expressions/RHSTerms.h>
#include <CCA/Components/Wasatch/Transport/EquationBase.h>

namespace WasatchCore{

  class  ExprDeps;  // forward declaration.
  class  BCHelper;
  struct GraphHelper;
  /**
   *  \ingroup WasatchCore
   *  \ingroup WasatchParticles
   *  \class  ParticleEquationBase
   *  \author Tony Saad
   *  \date   June, 2014
   *  \brief  Base class for defining a particle transport equation.
   */
  class ParticleEquationBase : public EquationBase {

  public:

    /**
     * @brief Construct a ParticleEquationBase
     * \param solnVarName The name of the solution variable for this equation
     *
     * \param pdir Specifies which position or momentum component this equation solves.
     *
     * \param particlePositionTags A taglist containing the tags of x, y, and z
     *        particle coordinates. Those may be needed by some particle
     *        expressions that require particle operators
     *
     * \param particleSizeTag Particle size tag. May be needed by some expressions.
     *
     * \param gc The GraphCategories object from Wasatch
     */
    ParticleEquationBase( const std::string& solnVarName,
                          const Direction pdir,
                          const Expr::TagList& particlePositionTags,
                          const Expr::Tag& particleSizeTag,
                          GraphCategories& gc );

    virtual ~ParticleEquationBase(){}
    
  protected:
    const Expr::TagList pPosTags_;
    const Expr::Tag     pSizeTag_;
  };
  
//  // helper functions
//  ParticleEquationBase::ParticleDirection string_to_particle_direction(const std::string& dirname);
//  std::string particle_direction_to_string(const ParticleEquationBase::ParticleDirection dir);
} // namespace WasatchCore

#endif // Wasatch_ParticleEquationBase_h

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

#ifndef SetupCoalModels_h
#define SetupCoalModels_h

#include <sci_defs/wasatch_defs.h>

#ifndef HAVE_POKITT
// kill compilation if we don't have pokitt.
#error Coal models require PoKiTT.
#endif

#include <expression/ExpressionFactory.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/SolverInterface.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Transport/EquationAdaptors.h>
#include <CCA/Components/Wasatch/GraphHelperTools.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/TagNames.h>

#include <CCA/Components/Wasatch/Coal/CoalInterface.h>


/**
 *  \file SetupCoalModels.h
 *  \brief Performs tasks needed to run coal models
 */

namespace WasatchCore{

typedef std::vector<EqnTimestepAdaptorBase*> EquationAdaptors;

  /**
   *  \class SetupCoalModels
   *  \author Josh McConnell
   *  \date January 2016
   *
   *  This class is used so turn on coal models and perform the necessary
   *  cell-to-particle and particle-to-cell interpolations.
   */
  class SetupCoalModels
  {

  public:
    SetupCoalModels( Uintah::ProblemSpecP& particleSpec,
                     Uintah::ProblemSpecP& wasatchSpec,
                     Uintah::ProblemSpecP& coalSpec,
                     GraphCategories& gc,
                     std::set<std::string> persistentFields );

    EquationAdaptors get_adaptors();

    ~SetupCoalModels();

  private:
    void setup_cantera();
    void set_initial_conditions();
    void setup_coal_src_terms();
    const Expr::Tag get_rhs_tag( std::string specName );
    const Expr::Tag get_c2p_tag( std::string specName );
    const Expr::Tag get_p2c_src_tag( std::string specName );

    GraphCategories& gc_;
    Uintah::ProblemSpecP& wasatchSpec_, particleSpec_, coalSpec_;
    Expr::ExpressionFactory& factory_;
    const TagNames& tagNames_;
    const Expr::Tag pMassTag_, pSizeTag_, pDensTag_, pTempTag_, gDensTag_;
    bool equationsSet_;
    EquationAdaptors adaptors_;
    Expr::TagList pPosTags_, yiTags_;

    Coal::CoalInterface<ParticleField>* coalInterface_;
  };
} // namespace WasatchCore

#endif // SetupCoalModels_h

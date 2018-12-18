/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef CCA_COMPONENTS_ELECTROCHEM_FLUXMODELS_FLUXMODELFACTORY_H
#define CCA_COMPONENTS_ELECTROCHEM_FLUXMODELS_FLUXMODELFACTORY_H

#include <CCA/Components/ElectroChem/FluxModels/FluxModel.h>
#include <CCA/Components/ElectroChem/FluxModels/BasicFlux.h>
#include <CCA/Components/ElectroChem/FluxModels/NullFlux.h>

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <string>

namespace ElectroChem {
  namespace FluxModelFactory {

    FluxModel* create(ProblemSpecP& ps) {
      ProblemSpecP flux_ps = ps->findBlock("flux_model");
      if(!flux_ps) { throw ProblemSetupException("Cannot find flux_model tag",
                                     __FILE__, __LINE__); }

      std::string flux_type;

      if(!flux_ps->getAttribute("type", flux_type)) {
        throw ProblemSetupException("No type for flux_model", __FILE__, __LINE__);
      }

      if(flux_type == "null_flux"){
        return scinew NullFlux(flux_ps);
      }else if(flux_type == "basic_flux"){
        return scinew BasicFlux(flux_ps);
      }else{
        throw ProblemSetupException("Unknown Material Type R ("+flux_type+")",
                                    __FILE__, __LINE__);
      }
    }
  } // End namespace FluxModelFactory
} // End namespace ElectroChem

#endif // CCA_COMPONENTS_ELECTROCHEM_FLUXMODELS_FLUXMODELFACTORY_H

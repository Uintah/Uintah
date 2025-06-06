/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

#include <CCA/Components/OnTheFlyAnalysis/AnalysisModuleFactory.h>
#include <CCA/Components/OnTheFlyAnalysis/lineExtract.h>
#include <CCA/Components/OnTheFlyAnalysis/MinMax.h>
#include <CCA/Components/OnTheFlyAnalysis/meanTurbFluxes.h>
#include <CCA/Components/OnTheFlyAnalysis/momentumAnalysis.h>
#include <CCA/Components/OnTheFlyAnalysis/planeAverage.h>
#include <CCA/Components/OnTheFlyAnalysis/planeExtract.h>
#include <CCA/Components/OnTheFlyAnalysis/statistics.h>
#include <CCA/Components/OnTheFlyAnalysis/spatialAvg.h>
#include <CCA/Components/OnTheFlyAnalysis/turbulentFluxes.h>

#include <sci_defs/uintah_defs.h>

#if !defined( NO_ICE )
  #include <CCA/Components/OnTheFlyAnalysis/vorticity.h>
  #include <CCA/Components/OnTheFlyAnalysis/controlVolFluxes.h>
#endif

#if !defined( NO_MPM )
  #include <CCA/Components/OnTheFlyAnalysis/flatPlate_heatFlux.h>
  #include <CCA/Components/OnTheFlyAnalysis/particleExtract.h>
#endif

#if !defined( NO_ICE ) && !defined( NO_MPM )
  #include <CCA/Components/OnTheFlyAnalysis/1stLawThermo.h>
#endif

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/MaterialManager.h>

#include <string>
#include <vector>
#include <iostream>

using namespace Uintah;

AnalysisModuleFactory::AnalysisModuleFactory()
{
}

AnalysisModuleFactory::~AnalysisModuleFactory()
{
}

//______________________________________________________________________
//
std::vector<AnalysisModule*>
AnalysisModuleFactory::create(const ProcessorGroup* myworld,
                              const MaterialManagerP materialManager,
                              const ProblemSpecP& prob_spec)
{
  // the factory can use either of these xml tags
  ProblemSpecP base_ps;
  ProblemSpecP da_ps = prob_spec->findBlock("DataAnalysis");
  ProblemSpecP pp_ps = prob_spec->findBlock("PostProcess");

  if( da_ps ){
    base_ps = da_ps;
  } 
  else{
    base_ps = pp_ps;
  }


  std::vector<AnalysisModule*> modules;

  if (base_ps) {

    for( ProblemSpecP module_ps = base_ps->findBlock( "Module" );
                      module_ps != nullptr;
                      module_ps = module_ps->findNextBlock( "Module" ) ) {

      if( !module_ps ) {
        throw ProblemSetupException( "\nERROR<DataAnalysis>: Could not find find <Module> tag.\n", __FILE__, __LINE__ );
      }

      std::map<std::string, std::string> attributes;
      module_ps->getAttributes(attributes);
      std::string module = attributes["name"];

      if ( module == "statistics" ) {
        modules.push_back( scinew statistics(          myworld, materialManager, module_ps) );
      }
      else if ( module == "spatialAvg" ) {
        modules.push_back( scinew spatialAvg(          myworld, materialManager, module_ps) );
      }
      else if ( module == "lineExtract" ) {
        modules.push_back(scinew lineExtract(          myworld, materialManager, module_ps ) );
      }
      else if ( module == "planeAverage" ) {
        modules.push_back( scinew planeAverage(        myworld, materialManager, module_ps, true, true, 0 ) );
      }
      else if ( module == "planeExtract" ) {
        modules.push_back( scinew planeExtract(        myworld, materialManager, module_ps ) );
      }
      else if ( module == "meanTurbFluxes" ) {
        modules.push_back( scinew meanTurbFluxes(      myworld, materialManager, module_ps ) );
      }
      else if ( module == "momentumAnalysis" ) {
        modules.push_back( scinew momentumAnalysis(    myworld, materialManager, module_ps ) );
      }
      else if ( module == "minMax" ) {
        modules.push_back( scinew MinMax(              myworld, materialManager, module_ps ) );
      }
      else if ( module == "turbulentFluxes" ) {
        modules.push_back( scinew turbulentFluxes(     myworld, materialManager, module_ps ) );
      }

#if !defined( NO_ICE )
      else if ( module == "vorticity" ) {
        modules.push_back( scinew vorticity(           myworld, materialManager, module_ps ) );
      }
      else if ( module == "controlVolFluxes" ) {
        modules.push_back( scinew controlVolFluxes(    myworld, materialManager, module_ps ) );
      }
#endif

#if !defined( NO_MPM )
      else if ( module == "particleExtract" ) {
        modules.push_back( scinew particleExtract(     myworld, materialManager, module_ps ) );
      }
      else if ( module == "flatPlate_heatFlux" ) {
        modules.push_back( scinew flatPlate_heatFlux(  myworld, materialManager, module_ps ) );
      }
#endif

#if !defined( NO_ICE ) && !defined( NO_MPM )
      else if ( module == "firstLawThermo" ) {
        modules.push_back( scinew FirstLawThermo(      myworld, materialManager, module_ps ) );
      }
#endif

      else {
        std::ostringstream msg;
        msg << "\nERROR<DataAnalysis>: Unknown analysis module : " << module << ".\n";
        throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
      }
    }
  }
  return modules;
}

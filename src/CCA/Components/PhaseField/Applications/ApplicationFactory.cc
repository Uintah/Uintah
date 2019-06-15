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

/**
 * @file CCA/Components/PhaseField/Applications/ApplicationFactory.cc
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#include <CCA/Components/PhaseField/Applications/ApplicationFactory.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;
using namespace PhaseField;

template<> UintahParallelComponentFactory::FactoryMap UintahParallelComponentFactory::RegisteredNames = {};

UintahParallelComponent *
ApplicationFactory::create (
    const ProcessorGroup * myWorld,
    const MaterialManagerP materialManager,
    ProblemSpecP probSpec,
    bool doAMR
)
{
    std::string type;
    if ( !probSpec->getAttribute ( "type", type ) )
        SCI_THROW ( ProblemSetupException ( "Cannot find type attribute in PhaseField block within problem specification file.", __FILE__, __LINE__ ) );
    std::transform ( type.begin(), type.end(), type.begin(), ::tolower );

    // composing application factory name [amr|]<application>|<var>|<dim>|<stn>
    std::string var;
    var = "cc";
    probSpec->getWithDefault ( "var", var, var );

    int dim = 2;
    probSpec->getWithDefault ( "dim", dim, dim );

    std::string stn;
    if ( dim == 2 ) stn = "p5";
    else if ( dim == 3 ) stn = "p7";

    probSpec->getWithDefault ( "dim", dim, dim );
    std::transform ( stn.begin(), stn.end(), stn.begin(), ::tolower );

    std::string application = ( doAMR ? "amr|" : "" ) + type + "|" + var + "|d" + std::to_string ( dim ) + "|" + stn;

    int verbosity;
    probSpec->getWithDefault ( "verbosity", verbosity, 0 );

    UintahParallelComponentBase * ptr = UintahParallelComponentFactory::Create ( application, myWorld, materialManager, verbosity );

    if ( !ptr )
        SCI_THROW ( ProblemSetupException ( "Cannot Create PhaseField Application '" + application + "'", __FILE__, __LINE__ ) );
    return dynamic_cast<UintahParallelComponent *> ( ptr );
}

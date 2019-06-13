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
 * @file CCA/Components/PhaseField/Applications/ApplicationFactory.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_Applications_ApplicationFactory_h
#define Packages_Uintah_CCA_Components_PhaseField_Applications_ApplicationFactory_h

#include <CCA/Components/PhaseField/Factory/Base.h>
#include <CCA/Components/PhaseField/Factory/Factory.h>

#include <Core/Grid/MaterialManagerP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah
{

class ProcessorGroup;
class UintahParallelComponent;

namespace PhaseField
{

/// Base class for UintahParallelComponent
using UintahParallelComponentBase = Base<UintahParallelComponent>;

/// Factory class for UintahParallelComponent
using UintahParallelComponentFactory = Factory<UintahParallelComponent, const ProcessorGroup *, const MaterialManagerP, int>;

/**
 * @brief Factory class for different PhaseField applications
 *
 * Factory class for creating new instances of UintahParallelComponent
 * within the PhaseField component
 */
class ApplicationFactory
{
public:
    /**
    * @brief factory create method
    *
    * Factory method for creating new instances of UintahParallelComponent
    * within the PhaseField component
    *
    * @param myWorld data structure to manage mpi processes
    * @param materialManager data structure to manage materials
    * @param probSpec specifications parsed from ups input file
    * @param doAMR if adaptive mesh refinement is requsted by the input
    */
    static UintahParallelComponent * create (
        const ProcessorGroup * myWorld,
        const MaterialManagerP materialManager,
        ProblemSpecP probSpec,
        bool doAMR
    );
}; // class ApplicationFactory

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_Applications_ApplicationFactory_h

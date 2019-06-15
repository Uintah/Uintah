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
 * @file CCA/Components/PhaseField/Applications/Application.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_Applications_Application_h
#define Packages_Uintah_CCA_Components_PhaseField_Applications_Application_h

#include <CCA/Components/Application/ApplicationCommon.h>
#include <CCA/Components/PhaseField/DataWarehouse/DWInterface.h>
#include <CCA/Components/PhaseField/BoundaryConditions/BCInterface.h>
#include <CCA/Components/PhaseField/AMR/AMRInterface.h>

namespace Uintah
{
namespace PhaseField
{

/**
 * @brief Virtual base for PhaseField applications
 *
 * Wrapper of ApplicationCommon and interfaces
 *
 * @tparam VAR type of variable representation
 * @tparam DIM problem dimension
 * @tparam STN finite-difference stencil
 * @tparam AMR whether to use adaptive mesh refinement
 */
template < VarType VAR, DimType DIM, StnType STN, bool AMR = false > class Application;

/**
 * @brief Virtual base for PhaseField applications (non-AMR implementation)
 *
 * Wrapper of ApplicationCommon and interfaces
 *
 * @implements Application < VAR, DIM, STN, AMR >
 * @tparam VAR type of variable representation
 * @tparam DIM problem dimension
 * @tparam STN finite-difference stencil
 */
template < VarType VAR, DimType DIM, StnType STN>
class Application<VAR, DIM, STN, false>
    : public ApplicationCommon
    , protected DWInterface<VAR, DIM>
    , protected BCInterface<VAR, STN>
{
public:
    using ApplicationCommon::ApplicationCommon;
}; // class Application

/**
 * @brief Virtual base for PhaseField applications (AMR implementation)
 *
 * Wrapper of ApplicationCommon and interfaces
 *
 * @implements Application < VAR, DIM, STN, AMR >
 * @tparam VAR type of variable representation
 * @tparam DIM problem dimension
 * @tparam STN finite-difference stencil
 */
template < VarType VAR, DimType DIM, StnType STN>
class Application<VAR, DIM, STN, true>
    : public ApplicationCommon
    , protected DWInterface<VAR, DIM>
    , protected BCInterface<VAR, STN>
    , protected AMRInterface<VAR, DIM>
{
public:
    using ApplicationCommon::ApplicationCommon;
}; // class Application

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_Applications_Application_h

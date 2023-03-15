/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
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

#ifndef Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_interface1_h
#define Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_interface1_h

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Interface for data-warehouse (variable and dimension dependent implementations)
 *
 * groups together various methods to get info about patches and levels which
 * depend on the different types of variable representation and problem
 * dimensions allowing to choose the relevant implementation at compile time
 *
 * @tparam VAR type of variable representation
 * @tparam DIM problem dimension
 */
template < VarType VAR, DimType DIM >
class dw_interface1;

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#include <CCA/Components/PhaseField/DataWarehouse/detail/dw_interface1_CC_D1.h>
#include <CCA/Components/PhaseField/DataWarehouse/detail/dw_interface1_CC_D2.h>
#include <CCA/Components/PhaseField/DataWarehouse/detail/dw_interface1_CC_D3.h>
#include <CCA/Components/PhaseField/DataWarehouse/detail/dw_interface1_NC_D1.h>
#include <CCA/Components/PhaseField/DataWarehouse/detail/dw_interface1_NC_D2.h>
#include <CCA/Components/PhaseField/DataWarehouse/detail/dw_interface1_NC_D3.h>

#endif // Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_interface1_h

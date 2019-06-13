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
 * @file CCA/Components/PhaseField/DataWarehouse/DWFDViewFactory.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_DWFDViewFactory_h
#define Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_DWFDViewFactory_h

#include <CCA/Components/PhaseField/Factory/Factory.h>
#include <CCA/Components/PhaseField/Views/FDView.h>

namespace Uintah
{
namespace PhaseField
{

/// Factory creator for DWFDView
template<typename Field, StnType STN> using DWFactoryFDView = Factory < FDView<Field, STN>, const typename Field::label_type &, int, const Level * >;

/// Factory base for DWFDView
template<typename Field, StnType STN> using DWBaseFDView = Base< FDView<Field, STN> >;

/// Factory creator implementation for DWFDView Not implemented as there is no need yet
template<typename Field, VarType VAR, StnType STN> struct DWFDViewFactory;

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_DWFDViewFactory_h

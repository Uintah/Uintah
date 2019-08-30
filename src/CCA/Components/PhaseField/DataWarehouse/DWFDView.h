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
 * @file CCA/Components/PhaseField/DataWarehouse/DWFDView.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_DWFDView_h
#define Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_DWFDView_h

#include <CCA/Components/PhaseField/Factory/Implementation.h>
#include <CCA/Components/PhaseField/Views/FDView.h>
#include <CCA/Components/PhaseField/DataWarehouse/detail/dwfd_view.h>

namespace Uintah
{
namespace PhaseField
{

/**
 * @brief Wrapper of DataWarehouse variables for both basic and complex
 * differential operations
 *
 * Factory Implementation for dynamic instantiation
 *
 * @tparam Field type of field (ScalarField < T > or VectorField < T, N >)
 * @tparam STN finite-difference stencil
 * @tparam VAR type of variable representation
 */
template <typename Field, StnType STN, VarType VAR >
class DWFDView
    : virtual public FDView<Field, STN>
    , public Implementation <
    DWFDView<Field, STN, VAR>, FDView<Field, STN>,
    const typename Field::label_type &,
    int, const Level *
    >
    , public detail::dwfd_view<Field, STN, VAR>
{
public: // STATIC MEMBERS

    /// Implementation identifier within Factory
    static const std::string Name;

public: // CONSTRUCTORS/DESTRUCTOR

    /**
     * @brief Constructor
     *
     * Instantiate view components without gathering info from the DataWarehouse
     *
     * @param label list of variable labels for each component
     * @param material material index
     * @param level grid level
     */
    DWFDView (
        const typename Field::label_type & label,
        int material,
        const Level * level
    ) : detail::dwfd_view<Field, STN, VAR> ( label, material, level )
    {}

    /**
     * @brief Constructor
     *
     * Instantiate view components and gather info from dw
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param label list of variable labels for each component
     * @param material material index
     * @param patch grid patch
     */
    DWFDView (
        DataWarehouse * dw,
        const typename Field::label_type & label,
        int material,
        const Patch * patch
    ) : detail::dwfd_view<Field, STN, VAR> ( dw, label, material, patch )
    {}

    /// Default destructor
    virtual ~DWFDView() = default;

    /// Prevent copy (and move) constructor
    DWFDView ( const DWFDView & ) = delete;

    /// Prevent copy (and move) assignment
    /// @return deleted
    DWFDView & operator= ( const DWFDView & ) = delete;

}; // class DWFDView

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_DWFDView_h

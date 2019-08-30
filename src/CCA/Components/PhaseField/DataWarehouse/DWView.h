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
 * @file CCA/Components/PhaseField/DataWarehouse/DWView.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_DWView_h
#define Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_DWView_h

#include <CCA/Components/PhaseField/DataWarehouse/detail/dw_view.h>
#include <CCA/Components/PhaseField/Views/View.h>
#include <CCA/Components/PhaseField/Factory/Implementation.h>

namespace Uintah
{
namespace PhaseField
{

/**
 * @brief Wrapper of DataWarehouse variables
 *
 * Factory Implementation for dynamic instantiation
 *
 * @tparam Field type of field (ScalarField < T > or VectorField < T, N >)
 * @tparam VAR type of variable representation
 * @tparam DIM problem dimension
 */
template<typename Field, VarType VAR, DimType DIM >
class DWView :
    public Implementation < DWView<Field, VAR, DIM>, View<Field>, const typename Field::label_type &, int >,
    virtual public View<Field>,
    public detail::dw_view<Field, VAR, DIM, 0>
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
     */
    DWView (
        const typename Field::label_type & label,
        int material
    ) : detail::dw_view<Field, VAR, DIM, 0> ( label, material )
    {
    }

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
    DWView (
        DataWarehouse * dw,
        const typename Field::label_type & label,
        int material,
        const Patch * patch
    ) : detail::dw_view<Field, VAR, DIM, 0> ( dw, label, material, patch )
    {
    }

    /// Default destructor
    virtual ~DWView() = default;

    /// Prevent copy (and move) constructor
    DWView ( const DWView & ) = delete;

    /// Prevent copy (and move) assignment
    /// @return deleted
    DWView & operator= ( const DWView & ) = delete;

}; // class DWView

} // namespace PhaseField
} // namespace Uintah

// #include <CCA/Components/PhaseField/DataWarehouse/DWViewP5.h>
// #include <CCA/Components/PhaseField/DataWarehouse/DWViewP7.h>

#endif // Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_DWView_h

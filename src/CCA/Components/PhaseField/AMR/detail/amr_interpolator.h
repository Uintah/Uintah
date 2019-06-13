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
 * @file CCA/Components/PhaseField/AMR/detail/amr_interpolator.h
 * @author Jon Matteo Church
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_interpolator_h
#define Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_interpolator_h

#include <CCA/Components/PhaseField/Util/Definitions.h>
#include <CCA/Components/PhaseField/Util/Expressions.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Abstract wrapper of grid variables for interpolation from coarser to
 *        finer levels.
 *
 * Adds to view the possibility to compute multi-grid interpolation
 *
 * @remark All different interpolation strategies must specialize this class and
 *         implement the view< T > class
 *
 * @tparam Field type of Field (should be only ScalarField)
 * @tparam Problem type of PhaseField problem
 * @tparam Index index_sequence of Field within Problem (first element is variable index,
 * following ones, if present, are the component index within the variable)
 */
template<typename Field, typename Problem, typename Index, FCIType FCI, DimType DIM> class amr_interpolator;

/**
 * @brief Abstract wrapper of grid variables for interpolation from coarser to
 * finer levels. (VectorField implementation)
 *
 * Adds to view the possibility to compute multi-grid interpolation
 *
 * @tparam T type of each component of the field at each point
 * @tparam N number of components
 * @tparam Problem type of PhaseField problem
 * @tparam Index index_sequence of Field within Problem (first element is variable index,
 * following ones, if present, are the component index within the variable)
 */
template<typename T, size_t N, typename Problem, typename Index, FCIType FCI, DimType DIM>
class amr_interpolator < VectorField<T, N>, Problem, Index, FCI, DIM >
    : public view_array < amr_interpolator < ScalarField<T>, Problem, Index, FCI, DIM >, ScalarField<T>, N >
{
private: // TYPES

    /// Type of field
    using Field = VectorField<T, N>;

    /// Type of View of each component
    using View = amr_interpolator < ScalarField<T>, Problem, Index, FCI, DIM >;

public:

    /**
     * @brief Constructor
     *
     * Instantiate amr_interpolator components without gathering info from the DataWarehouse
     *
     * @param label list of variable labels for each component
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material material index
     */
    amr_interpolator (
        const typename Field::label_type & label,
        const VarLabel * subproblems_label,
        int material
    )
    {
        for ( size_t i = 0; i < N; ++i )
            this->m_view_ptr[i] = new View ( label[i], subproblems_label, material );
    }

    /**
     * @brief Constructor
     *
     * Instantiate amr_interpolator components and gather info from dw
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param label list of variable labels for each component
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material material index
     * @param patch grid patch
     * @param use_ghosts if ghosts value are to be retrieved
     */
    amr_interpolator (
        DataWarehouse * dw,
        const typename Field::label_type & label,
        const VarLabel * subproblems_label,
        int material,
        const Patch * patch,
        bool use_ghosts = View::use_ghosts_dflt
    )
    {
        for ( size_t i = 0; i < N; ++i )
            this->m_view_ptr[i] = new View ( dw, label[i], material, subproblems_label, patch, use_ghosts );
    }

    /// Destructor
    virtual ~amr_interpolator()
    {
        for ( auto view : this->m_view_ptr )
            delete view;
    }

    /// Prevent copy (and move) constructor
    amr_interpolator ( const amr_interpolator & ) = delete;

    /// Prevent copy (and move) assignment
    amr_interpolator & operator= ( const amr_interpolator & ) = delete;
};

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#include <CCA/Components/PhaseField/AMR/detail/amr_interpolator_I0.h>
#include <CCA/Components/PhaseField/AMR/detail/amr_interpolator_I1_D1.h>
#include <CCA/Components/PhaseField/AMR/detail/amr_interpolator_I1_D2.h>
#include <CCA/Components/PhaseField/AMR/detail/amr_interpolator_I1_D3.h>

#endif // Packages_Uintah_CCA_Components_PhaseField_AMR_detail_amr_interpolator_h

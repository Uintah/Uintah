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
 * @file CCA/Components/PhaseField/DataWarehouse/detail/dwfd_view.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dwfd_view_h
#define Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dwfd_view_h

#include <CCA/Components/PhaseField/DataWarehouse/detail/dw_basic_fd_view.h>
#include <CCA/Components/PhaseField/DataWarehouse/detail/dw_fd_view.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Wrapper of DataWarehouse variables for both basic and complex
 * differential operations
 *
 * Adds to dw_view the possibility to compute finite-difference approximation of
 * of differential operations at internal cells/points
 *
 * @tparam Field type of field (ScalarField < T > or VectorField < T, N >)
 * @tparam STN finite-difference stencil
 * @tparam VAR type of variable representation
 */
template<typename Field, StnType STN, VarType VAR> class dwfd_view;

/**
 * @brief Wrapper of DataWarehouse variables for both basic and complex
 * differential operations (ScalarField implementation)
 *
 * Adds to dw_view the possibility to compute finite-difference approximation of
 * of differential operations at internal cells/points
 *
 * @tparam T type of the field value at each point
 * @tparam STN finite-difference stencil
 * @tparam VAR type of variable representation
 */
template<typename T, StnType STN, VarType VAR>
class dwfd_view < ScalarField<T>, STN, VAR >
    : public dw_basic_fd_view < ScalarField<T>, STN, VAR >
    , public dw_fd_view < ScalarField<T>, STN, VAR >
{
private: // TYPES

    /// Type of field
    using Field = ScalarField<T>;

public: // CONSTRUCTORS/DESTRUCTOR

    /// View is constructed by dw_basic_fd_view
    using dw_basic_fd_view<Field, STN, VAR>::dw_basic_fd_view;

    /// Default destructor
    virtual ~dwfd_view() = default;

    /// Prevent copy (and move) constructor
    dwfd_view ( const dwfd_view & ) = delete;

    /// Prevent copy (and move) assignment
    /// @return deleted
    dwfd_view & operator= ( const dwfd_view & ) = delete;

}; // class dwfd_view

/**
 * @brief Wrapper of DataWarehouse variables for both basic and complex
 * differential operations (VectorField implementation)
 *
 * Adds to dw_view the possibility to compute finite-difference approximation of
 * of differential operations at internal cells/points
 *
 * @tparam T type of each component of the field value at each point
 * @tparam N dimension of the vetor field
 * @tparam STN finite-difference stencil
 * @tparam VAR type of variable representation
 */
template<typename T, size_t N, StnType STN, VarType VAR>
class dwfd_view < VectorField<T, N>, STN, VAR >
    : virtual public view_array < dwfd_view < ScalarField<T>, STN, VAR >, ScalarField<T>, N >
{
private: // TYPES

    /// Type of field
    using Field = VectorField<T, N>;

    /// Type of View of each component
    using View = dwfd_view < ScalarField<T>, STN, VAR >;

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
    dwfd_view (
        const typename Field::label_type & label,
        int material,
        const Level * level
    )
    {
        for ( size_t i = 0; i < N; ++i )
            this->m_view_ptr[i] = scinew View ( label[i], material, level );
    }

    /**
     * @brief Constructor
     *
     * Instantiate view components and gather info from dwr
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param label list of variable labels for each component
     * @param material material index
     * @param patch grid patch
     * @param use_ghosts if ghosts value are to be retrieved
     */
    dwfd_view (
        DataWarehouse * dw,
        const typename Field::label_type & label,
        int material,
        const Patch * patch,
        bool use_ghosts = View::use_ghosts_dflt
    )
    {
        for ( size_t i = 0; i < N; ++i )
            this->m_view_ptr[i] = scinew View ( dw, label[i], material, patch, use_ghosts );
    }

    /// Destructor
    virtual ~dwfd_view()
    {
        for ( auto view : this->m_view_ptr )
            delete view;
    }

    /// Prevent copy (and move) constructor
    dwfd_view ( const dwfd_view & ) = delete;

    /// Prevent copy (and move) assignment
    /// @return deleted
    dwfd_view & operator= ( const dwfd_view & ) = delete;

}; // dw_fd_view

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

// #include <CCA/Components/PhaseField/DataWarehouse/DWFDViewP5.h>
// #include <CCA/Components/PhaseField/DataWarehouse/DWFDViewP7.h>

#endif // Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_DWFDView_h

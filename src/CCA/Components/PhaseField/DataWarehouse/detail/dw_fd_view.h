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
 * @file CCA/Components/PhaseField/DataWarehouse/detail/dw_fd_view.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_fd_view_h
#define Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_fd_view_h

#include <CCA/Components/PhaseField/Views/detail/fd_view.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Wrapper of DataWarehouse variables for complex differential operations
 *
 * Adds to dw_view the possibility to compute finite-difference approximation of
 * of complex differential operations (gradient and laplacian) at internal
 * cells/points
 *
 * @tparam Field type of field (ScalarField < T > or VectorField < T, N >)
 * @tparam STN finite-difference stencil
 * @tparam VAR type of variable representation
 */
template<typename Field, StnType STN, VarType VAR> class dw_fd_view;

/**
 * @brief Wrapper of DataWarehouse variables for complex differential operations
 * (ScalarField implementation)
 *
 * Adds to dw_view the possibility to compute finite-difference approximation of
 * of complex differential operations (gradient and laplacian) at internal
 * cells/points
 *
 * @remark actual finite-differences implementations are in dw_fd
 *
 * @tparam T type of the field value at each point
 * @tparam STN finite-difference stencil
 * @tparam VAR type of variable representation
 */
template<typename T, StnType STN, VarType VAR>
class dw_fd_view < ScalarField<T>, STN, VAR >
    : virtual public fd_view < ScalarField<T>, STN >
{
private: // STATIC MEMBERS

    /// Problem Dimension
    static constexpr DimType DIM = get_stn<STN>::dim;

private: // TYPES

    /// Type of field
    using Field = ScalarField<T>;

    /// Non const type of the field value
    using V = typename std::remove_const<T>::type;

public: // CONSTRUCTORS/DESTRUCTOR

    /// Default constructor
    dw_fd_view () = default;

    /// Default destructor
    virtual ~dw_fd_view() = default;

    /// Prevent copy (and move) constructor
    dw_fd_view ( const dw_fd_view & ) = delete;

    /// Prevent copy (and move) assignment
    dw_fd_view & operator= ( const dw_fd_view & ) = delete;

public: // FD VIEW METHODS

    /**
     * @brief Get gradient value at position
     *
     * @param id position index
     * @return gradient value at id
     */
    virtual inline std::vector<V>
    gradient (
        const IntVector & id
    ) const override
    {
        std::vector<V> res ( DIM );
        res[X] = this->template dx ( id );
        if ( DIM > D1 ) res[Y] = this->dy ( id );
        if ( DIM > D2 ) res[Z] = this->dz ( id );
        return res;
    }

    /**
     * @brief Get laplacian at position
     *
     * @param id position index
     * @return laplacian value at id
     */
    virtual inline typename std::remove_const<T>::type
    laplacian (
        const IntVector & id
    ) const override
    {
        typename std::remove_const<T>::type res = this->dxx ( id );
        if ( DIM > D1 ) res += this->dyy ( id );
        if ( DIM > D2 ) res += this->dzz ( id );
        return res;
    }

}; // class dw_fd_view

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_fd_view_h

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
 * @file CCA/Components/PhaseField/Views/detail/basic_fd_view.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_Views_detail_basic_fd_view_h
#define Packages_Uintah_CCA_Components_PhaseField_Views_detail_basic_fd_view_h

#include <CCA/Components/PhaseField/Util/Definitions.h>
#include <CCA/Components/PhaseField/Views/detail/view.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Abstract wrapper of grid variables for basic finite-difference operations
 *
 * detail implementation of view (variable wrapping) which include also
 * finite difference approximation of basic differential operations (first and
 * second oreder derivatives)
 *
 * @tparam Field type of field (ScalarField < T > or VectorField < T, N >)
 * @tparam STN finite-difference stencil
 */
template<typename Field, StnType STN> class basic_fd_view;

/**
 * @brief Abstract class for basic finite-difference operations on variables
 * (ScalarField implementation)
 *
 * detail implementation of view (variable wrapping) which include also
 * finite difference approximation on basic differential operations (first and
 * second oreder derivatives)
 *
 * @tparam T type of the field value at each point
 * @tparam STN finite-difference stencil
 */
template<typename T, StnType STN>
class basic_fd_view < ScalarField<T>, STN >
    : virtual public view < ScalarField<T> >
{
public: // STATIC MEMBERS

    /// Default value for use_ghost when retrieving data
    static constexpr bool use_ghosts_dflt = true;

private: // TYPES

    /// Type of field
    using Field = ScalarField<T>;

    /// Non const type of the field value
    using V = typename std::remove_const<T>::type;

public: // DESTRUCTOR

    /// Default destructor
    virtual ~basic_fd_view() = default;

public: // VIEW METHODS

    /**
     * @brief Retrieve value from the DataWarehouse for a given patch
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param patch grid patch to retrieve data for
     * @param use_ghosts if ghosts value are to be retrieved
     */
    virtual void set ( DataWarehouse * dw, const Patch * patch, bool use_ghosts = use_ghosts_dflt ) = 0;

    /**
     * @brief Retrieve value from the DataWarehouse for a given region
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param level grid level from which retrieve data
     * @param low lower bound of the region to retrieve
     * @param high higher bound of the region to retrieve
     * @param use_ghosts if ghosts value are to be retrieved
     */
    virtual void set ( DataWarehouse * dw, const Level * level, const IntVector & low, const IntVector & high, bool use_ghosts = use_ghosts_dflt ) = 0;

public: // BASIC FD VIEW METHODS

    /**
     * @brief Get base view
     *
     * @return non const pointer to base view implementation
     */
    virtual view<Field> * get_view() = 0;

    /**
     * @brief Get base view
     *
     * @return const pointer to base view implementation
     */
    virtual const view<Field> * get_view() const = 0;

    /**
     * @brief Partial x derivative
     *
     * First order derivative along x at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual V dx ( const IntVector & id ) const = 0;

    /**
     * @brief Partial y derivative
     *
     * First order derivative along y at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual V dy ( const IntVector & id ) const = 0;

    /**
     * @brief Partial z derivative
     *
     * First order derivative along z at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual V dz ( const IntVector & id ) const = 0;

    /**
     * @brief Partial x second order derivative
     *
     * Second order derivative along x at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual V dxx ( const IntVector & id ) const = 0;

    /**
     * @brief Partial y second order derivative
     *
     * Second order derivative along y at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual V dyy ( const IntVector & id ) const = 0;

    /**
     * @brief Partial z second order derivative
     *
     * Second order derivative along z at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual V dzz ( const IntVector & id ) const = 0;

}; // class basic_fd_view

/**
 * @brief Abstract class for basic finite-difference operations on variables
 * (VectorField implementation)
 *
 * detail implementation of view (variable wrapping) which include also
 * finite difference approximation on basic differential operations (first and
 * second order derivatives)
 *
 * @tparam T type of each component of the field at each point
 * @tparam N number of components
 * @tparam STN finite-difference stencil
 */
template <typename T, size_t N, StnType STN>
class basic_fd_view < VectorField<T, N>, STN >
    : virtual public view_array < basic_fd_view < ScalarField<T>, STN >, ScalarField<T>, N >
{
public: // DESTRUCTOR

    /// Default destructor
    virtual ~basic_fd_view() = default;

}; // class basic_fd_view

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_Views_detail_basic_fd_view_h

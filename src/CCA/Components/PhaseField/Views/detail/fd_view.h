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
 * @file CCA/Components/PhaseField/Views/detail/fd_view.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_Views_detail_fd_view_h
#define Packages_Uintah_CCA_Components_PhaseField_Views_detail_fd_view_h

#include <CCA/Components/PhaseField/Views/detail/basic_fd_view.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Abstract wrapper of grid variables for complex differential operations
 *
 * Adds to view the possibility to compute finite-difference approximation of
 * of complex differential operations (gradient and laplacian) at internal
 * internal cells/points
 *
 * @tparam Field type of field (ScalarField < T > or VectorField < T, N >)
 * @tparam STN finite-difference stencil
 */
template<typename Field, StnType STN> class fd_view;

/**
 * @brief Abstract wrapper of grid variables for complex differential operations
 * (ScalarField implementation)
 *
 * Adds to view the possibility to compute finite-difference approximation of
 * of complex differential operations (gradient and laplacian) at internal
 * internal cells/points
 *
 * @tparam T type of the field value at each point
 * @tparam STN finite-difference stencil
 */
template<typename T, StnType STN>
class fd_view < ScalarField<T>, STN >
    : virtual public basic_fd_view < ScalarField<T>, STN >
{
private: // TYPES

    /// Type of field
    using Field = ScalarField<T>;

    /// Non const type of the field value
    using V = typename std::remove_const<T>::type;

#ifdef HAVE_HYPRE
    /// Type for stencil entries
    using S = typename get_stn<STN>::template type<T>;
#endif

public: // DESTRUCTOR

    /// Default destructor
    virtual ~fd_view() = default;

public: // FD VIEW METHODS

    /**
     * @brief Get gradient value at position
     *
     * @param id position index
     * @return gradient value at id
     */
    virtual std::vector<V> gradient ( const IntVector & id ) const = 0;

    /**
     * @brief Get laplacian at position
     *
     * @param id position index
     * @return laplacian value at id
     */
    virtual V laplacian ( const IntVector & id ) const = 0;

#ifdef HAVE_HYPRE
    virtual std::tuple<S, V> laplacian_sys_hypre ( const IntVector & id ) const = 0;
    virtual V laplacian_rhs_hypre ( const IntVector & id ) const = 0;
#endif

};

/**
 * @brief Abstract wrapper of grid variables for complex differential operations
 * (VectorField implementation)
 *
 * Adds to view the possibility to compute finite-difference approximation of
 * of complex differential operations (gradient and laplacian) at internal
 * internal cells/points
 *
 * @tparam T type of each component of the field at each point
 * @tparam N number of components
 * @tparam STN finite-difference stencil
 */
template <typename T, size_t N, StnType STN >
class fd_view < VectorField<T, N>, STN >
 : virtual public view_array < fd_view < ScalarField<T>, STN >, ScalarField<T>, N >
 {
public: // DESTRUCTOR

    /// Default destructor
    virtual ~fd_view () = default;

}; // class fd_view

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_Views_detail_fd_view_h



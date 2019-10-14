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
 * @file CCA/Components/PhaseField/Views/detail/view.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_Views_detail_view_h
#define Packages_Uintah_CCA_Components_PhaseField_Views_detail_view_h

#include <CCA/Components/PhaseField/Util/Definitions.h>
#include <CCA/Components/PhaseField/DataTypes/Support.h>
#include <CCA/Components/PhaseField/DataTypes/ScalarField.h>
#include <CCA/Components/PhaseField/DataTypes/VectorField.h>
#include <CCA/Components/PhaseField/Views/detail/view_array.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Abstract wrapper for accessing grid variables
 *
 * detail implementation of variable wrapping
 *
 * @remark constant view must use fields with const value type
 *
 * @tparam Field type of field (ScalarField < T > or VectorField < T, N >)
 */
template <typename Field> class view;

/**
 * @brief Abstract class for accessing variables (ScalarField implementation)
 *
 * detail implementation of variable wrapping
 *
 * @remark constant view must use fields with const value type
 *
 * @tparam T type of the field value at each point
 */
template <typename T>
class view < ScalarField<T> >
{
private: // TYPES

    /// Type of field
    using Field = ScalarField<T>;

    /// Non const type of the field value
    using V = typename std::remove_const<T>::type;

public: // TYPES

    /// Type of field
    using field_type = ScalarField<T>;

    /// Default value for use_ghost when retrieving data
    static constexpr bool use_ghosts_dflt = false;

public: // DESTRUCTOR

    /// Default destructor
    virtual ~view() = default;

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

    /**
     * @brief Get a copy of the view
     *
     * @param deep if true inner grid variable is copied as well otherwise the
     * same grid variable is referenced
     *
     * @return new view instance
     */
    virtual view * clone ( bool deep ) const = 0;

    /**
     * @brief Get a copy of the view and apply translate the support
     *
     * @remark It is meant to be used for virtual patches (i.e. periodic boundaries)
     *
     * @param deep if true inner grid variable is copied as well otherwise the
     * same grid variable is referenced
     * @param offset vector specifying the translation of the support
     * @return new view instance
     */
    virtual view * clone ( bool deep, const IntVector & offset ) const = 0;

    /**
     * @brief Get the region for which the view has access to the DataWarehouse
     *
     * @return support of the view
     */
    virtual Support get_support() const = 0;

    /**
     * @brief Check if the view has access to the position with index id
     *
     * @param id position index
     * @return check result
     */
    virtual bool is_defined_at ( const IntVector & id ) const = 0;

    /**
     * @brief Get/Modify value at position with index id (modifications are allowed if T is non const)
     *
     * @param id position index
     * @return reference to field value at id
     */
    virtual T & operator[] ( const IntVector & id ) = 0;

    /**
     * @brief Get value at position
     *
     * @param id position index
     * @return field value at id
     */
    virtual V operator[] ( const IntVector & id ) const = 0;

}; // class view

/**
 * @brief Abstract class for accessing variables (VectorField implementation)
 *
 * detail implementation of variable wrapping
 *
 * @remark constant view must use fields with const value type
 *
 * @tparam T type of each component of the field at each point
 * @tparam N number of components
 */
template <typename T, size_t N >
class view < VectorField<T, N> >
    : virtual public view_array < view < ScalarField<T> >, ScalarField<T>, N >
{
public: // DESTRUCTOR

    /// Default destructor
    virtual ~view() = default;

    /**
     * @brief Get a copy of the view
     *
     * @param deep if true inner grid variable is copied as well otherwise the
     * same grid variable is referenced
     *
     * @return new view instance
     */
    virtual view * clone ( bool deep ) const = 0;

}; // class view

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_detail_view_h

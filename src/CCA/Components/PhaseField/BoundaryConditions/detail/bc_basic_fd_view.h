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
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WAPBRRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/**
 * @file CCA/Components/PhaseField/BoundaryConditions/detail/bc_basic_fd_view.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_bc_basic_fd_view_h
#define Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_bc_basic_fd_view_h

#include <CCA/Components/PhaseField/BoundaryConditions/detail/bc_fd.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Wrapper of DataWarehouse variables for basic differential operations
 *
 * Adds to dw_view the possibility to compute finite-difference approximation of
 * of basic differential operations (first and second order derivatives) at
 * boundary faces
 *
 * @tparam Field type of field (ScalarField < T > or VectorField < T, N >)
 * @tparam STN finite-difference stencil
 * @tparam VAR type of variable representation
 * @tparam P BC, FC, and Patch::Face pack
 */
template<typename Field, StnType STN, VarType VAR, BCF P> class bc_basic_fd_view;

/**
 * @brief Wrapper of DataWarehouse variables for basic differential operations
 * (ScalarField implementation)
 *
 * Adds to dw_view the possibility to compute finite-difference approximation of
 * of basic differential operations (first and second order derivatives) at
 * boundary faces
 *
 * @remark actual finite-differences implementations are in bc_fd
 *
 * @tparam T type of the field value at each point
 * @tparam STN finite-difference stencil
 * @tparam VAR type of variable representation
 * @tparam P BC, FC, and Patch::Face pack
 */
template<typename T, StnType STN, VarType VAR, BCF P>
class bc_basic_fd_view < ScalarField<T>, STN, VAR, P >
    : virtual public basic_fd_view < ScalarField<T>, STN >
    , public bc_fd < ScalarField<T>, STN, VAR, get_bcf<P>::face, get_bcf<P>::bc, get_stn<STN>::ghosts, get_bcf <P>::c2f >  // fd implementation
{
private: // STATIC MEMBERS

    /// Number of ghosts required by STN
    static constexpr int GN = get_stn<STN>::ghosts;

    /// Type of boundary conditions
    static constexpr BC B = get_bcf<P>::bc;

    /// Type of fine/coarse interface conditions
    static constexpr FC C2F = get_bcf<P>::c2f;

    /// Patch face
    static constexpr Patch::FaceType F = get_bcf<P>::face;

private: // TYPES

    /// Type of field
    using Field = ScalarField<T>;

    /// Non const type of the field value
    using V = typename std::remove_const<T>::type;

    /// Type of rhs for the given BC
    using BV = typename get_bc< B >::template value_type< Field >;

protected: // COPY CONSTRUCTOR

    /**
     * @brief Constructor
     *
     * Instantiate a copy of a given view
     *
     * @param copy source view for copying
     * @param deep if true inner grid variable is copied as well otherwise the
     * same grid variable is referenced
     */
    bc_basic_fd_view (
        const bc_basic_fd_view * copy,
        bool deep
    ) : bc_fd<Field, STN, VAR, F, B, GN, C2F> ( copy, deep )
    {}

public: // CONSTRUCTORS/DESTRUCTOR

    /**
     * @brief Constructor
     *
     * Instantiate a view without gathering info from the
     * DataWarehouse
     *
     * @param view view on which the bc has to be imposed
     * @param label variable label
     * @param material material index
     * @param level grid level
     * @param value used when imposing the prescribed BC
     */
    bc_basic_fd_view (
        view<Field> * view,
        const typename Field::label_type & label,
        int material,
        const Level * level,
        const BV & value
    ) : bc_fd<Field, STN, VAR, F, B, GN, C2F> ( view, label, material, level, value )
    {}

    /**
     * @brief Constructor
     *
     * Instantiate a view and gather info from dw
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param view view on which the bc has to be imposed
     * @param label variable label
     * @param material material index
     * @param level grid level
     * @param value used when imposing the prescribed BC
     * @param patch grid patch owning the face
     * @param use_ghosts if ghosts value are to be retrieved
    */
    bc_basic_fd_view (
        const view<Field> * view,
        DataWarehouse * dw,
        const VarLabel * label,
        int material,
        const Level * level,
        const BV & value,
        const Patch * patch,
        bool use_ghosts
    ) : bc_fd<Field, STN, VAR, F, B, GN, C2F> ( view, dw, label, material, level, value, patch, use_ghosts )
    {}

    virtual ~bc_basic_fd_view() = default;

    /// Prevent copy (and move) constructor
    bc_basic_fd_view ( const bc_basic_fd_view & ) = delete;

    /// Prevent copy (and move) assignment
    bc_basic_fd_view & operator= ( const bc_basic_fd_view & ) = delete;

public: // VIEW METHODS

    /**
     * @brief Get a copy of the view
     *
     * @param deep if true inner grid variable is copied as well otherwise the
     * same grid variable is referenced
     *
     * @return new view instance
     */
    virtual view<Field> *
    clone (
        bool deep
    ) const override
    {
        return scinew bc_basic_fd_view ( this, deep );
    };

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
    virtual view<Field> *
    clone (
        bool deep,
        const IntVector & offset
    )
    const override
    {
        return scinew virtual_view<bc_basic_fd_view, Field> ( this, deep, offset );
    };

public: // BASIC FD VIEW METHODS

    /**
     * @brief Get base view (virtual implementation)
     *
     * @remark bc values are computed at runtime thus cannot create non const
     * view, there's nothing to modify in the DataWarehouse
     *
     * @return nothing
     */
    virtual inline view<Field> *
    get_view()
    override VIRT;

    /**
     * @brief Get base view
     *
     * @return const pointer to base view implementation
     */
    virtual inline const view<Field> *
    get_view()
    const override
    {
        return bc_fd<Field, STN, VAR, F, B, GN, C2F>::get_view();
    };

    /**
     * @brief Partial x derivative
     *
     * First order derivative along x at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual V
    dx (
        const IntVector & id
    ) const override
    {
        return this->template d < X > ( id );
    }

    /**
     * @brief Partial y derivative
     *
     * First order derivative along y at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual V
    dy (
        const IntVector & id
    ) const override
    {
        return this->template d < Y > ( id );
    }

    /**
     * @brief Partial z derivative
     *
     * First order derivative along z at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual V
    dz (
        const IntVector & id
    ) const override
    {
        return this->template d < Z > ( id );
    };

    /**
     * @brief Partial x second order derivative
     *
     * Second order derivative along x at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual V
    dxx (
        const IntVector & id
    ) const override
    {
        return this->template d2 < X > ( id );
    }

    /**
     * @brief Partial x second order derivative
     *
     * Second order derivative along y at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual V
    dyy (
        const IntVector & id
    ) const override
    {
        return this->template d2 < Y > ( id );
    }

    /**
     * @brief Partial x second order derivative
     *
     * Second order derivative along z at index id
     *
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    virtual V
    dzz (
        const IntVector & id
    ) const override
    {
        return this->template d2 < Z > ( id );
    }

}; // class bc_basic_fd_view

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_bc_basic_fd_view_h

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
 * @file CCA/Components/PhaseField/DataWarehouse/detail/dw_fd_G1.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_fd_G1_h
#define Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_fd_G1_h

#include <CCA/Components/PhaseField/DataWarehouse/detail/dw_view.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Finite-differences scheme for dw variables (1-ghost in each direction)
 * (ScalarField implementation)
 *
 * Implement P3 (three-point 1D), P5 (five-point 2D) and P7 (seven-point 3D)
 * stencils
 *
 * @tparam T type of the field value at each point
 * @tparam STN finite-difference stencil
 * @tparam VAR type of variable representation
 */
template<typename T, StnType STN, VarType VAR>
class dw_fd < ScalarField<T>, STN, VAR, 1 >
    : virtual public basic_fd_view < ScalarField<T>, STN >
{
public: // STATIC MEMBERS

    /// Default value for use_ghost when retrieving data
    static constexpr bool use_ghosts_dflt = true;

private: // STATIC MEMBERS

    /// Problem dimension
    static constexpr DimType DIM = get_stn<STN>::dim;

    /// Number of ghosts required
    static constexpr int GN = 1;

private: // TYPES

    /// Type of field
    using Field = ScalarField<T>;

    /// Non const type of the field value
    using V = typename std::remove_const<T>::type;

#ifdef HAVE_HYPRE
    /// Stencil entries type
    using S = typename get_stn<STN>::template type<T>;
#endif

private:  // MEMBERS

    /// Wrapper of variable in the DataWarehouse
    dw_view<Field, VAR, DIM, GN> * m_view;

    /// Grid level
    const Level * m_level;

    /// Grid spacing
    Vector m_h;

private: // METHODS

    /**
     * @brief Get const reference to value at position
     *
     * @param id position index
     * @return field value at id
     */
    inline const V &
    value (
        const IntVector & id
    ) const
    {
        return ( *m_view ) [id];
    }

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
    dw_fd (
        const dw_fd * copy,
        bool deep
    ) : m_view ( dynamic_cast < dw_view<Field, VAR, DIM, GN> * > ( copy->m_view->clone ( deep ) ) ),
        m_level ( copy->m_level ),
        m_h ( copy->m_h )
    {}

public: // CONSTRUCTORS/DESTRUCTOR

    /**
     * @brief Constructor
     *
     * Instantiate a view without gathering info from the DataWarehouse
     *
     * @param label variable label
     * @param material material index
     * @param level grid level
     */
    dw_fd (
        const VarLabel * label,
        int material,
        const Level * level
    ) : m_view ( scinew dw_view<Field, VAR, DIM, GN> ( label, material ) ),
        m_level ( level ),
        m_h ( level->dCell() )
    {}

    /**
     * @brief Constructor
     *
     * Instantiate a view and gather info from dw
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param label variable label
     * @param material material index
     * @param patch grid patch
     * @param use_ghosts if ghosts value are to be retrieved
     */
    dw_fd (
        DataWarehouse * dw,
        const VarLabel * label,
        int material,
        const Patch * patch,
        bool use_ghosts = use_ghosts_dflt
    ) : m_view ( scinew dw_view<Field, VAR, DIM, GN> ( dw, label, material, patch, use_ghosts ) ),
        m_level ( patch->getLevel() ),
        m_h ( m_level->dCell() )
    {}

    /// Destructor
    virtual ~dw_fd()
    {
        delete m_view;
    };

    /// Prevent copy (and move) constructor
    dw_fd ( const dw_fd & ) = delete;

    /// Prevent copy (and move) assignment
    /// @return deleted
    dw_fd & operator= ( const dw_fd & ) = delete;

public: // VIEW METHODS

    /**
     * @brief Retrieve value from the DataWarehouse for a given patch
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param patch grid patch to retrieve data for
     * @param use_ghosts if ghosts value are to be retrieved
     */
    virtual inline void
    set (
        DataWarehouse * dw,
        const Patch * patch,
        bool use_ghosts
    ) override
    {
        m_level = patch->getLevel();
        m_h = m_level->dCell();
        m_view->set ( dw, patch, use_ghosts );
    }

    /**
     * @brief Retrieve value from the DataWarehouse for a given region
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param level grid level from which retrieve data
     * @param low lower bound of the region to retrieve
     * @param high higher bound of the region to retrieve
     * @param use_ghosts if ghosts value are to be retrieved
     */
    virtual inline void
    set (
        DataWarehouse * dw,
        const Level * level,
        const IntVector & low,
        const IntVector & high,
        bool use_ghosts
    ) override
    {
        m_view->set ( dw, level, low, high, use_ghosts );
    };

    /**
     * @brief Get the region for which the view has access to the DataWarehouse
     *
     * @return support of the view
     */
    virtual inline Support
    get_support()
    const override
    {
        return m_view->get_support();
    };

    /**
     * @brief Check if the view has access to the position with index id
     *
     * @param id position index
     * @return check result
     */
    virtual inline bool
    is_defined_at (
        const IntVector & id
    ) const override
    {
        return m_view->is_defined_at ( id );
    };

    /**
     * @brief Get/Modify value at position with index id (modifications are allowed if T is non const)
     *
     * @param id position index
     * @return reference to field value at id
     */
    virtual inline T &
    operator[] (
        const IntVector & id
    ) override
    {
        return ( *m_view ) [id];
    };

    /**
     * @brief Get value at position
     *
     * @param id position index
     * @return field value at id
     */
    virtual inline V
    operator[] (
        const IntVector & id
    ) const override
    {
        return value ( id );
    };

public: // BASIC FD VIEW METHODS

    /**
     * @brief Get base view
     *
     * @return non const pointer to base view implementation
     */
    virtual inline view<Field> *
    get_view()
    override
    {
        return m_view;
    };

    /**
     * @brief Get base view
     *
     * @return const pointer to base view implementation
     */
    virtual inline const view<Field> *
    get_view()
    const override
    {
        return m_view;
    };

public: // DW FD MEMBERS

    /**
     * @brief First order derivative
     *
     * First order derivative along DIR at index id
     *
     * @tparam DIR Direction along with derivative is approximated
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    template <DirType DIR>
    inline T
    d (
        const IntVector & id
    ) const
    {
        IntVector im ( id ), ip ( id );
        im[DIR] -= 1;
        ip[DIR] += 1;
        return ( value ( ip ) - value ( im ) ) / ( 2. * m_h[DIR] );
    }

    /**
     * @brief Second order derivative
     *
     * Second order derivative along DIR at index id
     *
     * @tparam DIR Direction along with derivative is approximated
     * @param id index where to evaluate the finite-difference
     * @return approximated value at id
     */
    template <DirType DIR>
    inline T
    d2 (
        const IntVector & id
    ) const
    {
        IntVector im ( id ), ip ( id );
        im[DIR] -= 1;
        ip[DIR] += 1;
        return ( value ( ip ) + value ( im ) - 2. * value ( id ) ) / ( m_h[DIR] * m_h[DIR] );
    }

#ifdef HAVE_HYPRE
    template <DirType DIR>
    inline void
    add_d2_sys_hypre (
        const IntVector & _DOXYARG ( id ),
        S & stencil_entries,
        T & /*rhs*/
    ) const
    {
        double h2 = m_h[DIR] * m_h[DIR];
        stencil_entries[2 * DIR] += 1. / h2;
        stencil_entries[2 * DIR + 1] += 1. / h2;
        stencil_entries.p += -2. / h2;
    }

    template <DirType DIR>
    inline void
    add_d2_rhs_hypre (
        const IntVector & _DOXYARG ( id ),
        T & /*rhs*/
    ) const
    {}
#endif

}; // class dw_fd_G1

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_fd_G1_h

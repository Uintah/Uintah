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
 * @file CCA/Components/PhaseField/DataWarehouse/detail/dw_view.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_view_h
#define Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_view_h

#include <CCA/Components/PhaseField/Views/detail/view.h>
#include <CCA/Components/PhaseField/Views/detail/virtual_view.h>

#include <CCA/Components/PhaseField/DataTypes/SubProblemsP.h>
#include <CCA/Components/PhaseField/DataTypes/Variable.h>

#include <CCA/Components/PhaseField/DataWarehouse/DWInterface.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Class for accessing variables from the DataWarehouse
 *
 * detail implementation of DataWarehouse variable wrapping
 *
 * @remark constant view must use fields with const value type
 *
 * @tparam Field type of field (ScalarField < T > or VectorField < T, N >)
 * @tparam VAR type of variable representation
 * @tparam DIM problem dimension
 * @tparam GN number of ghosts required
 */
template<typename Field, VarType VAR, DimType DIM, size_t GN> class dw_view;

/**
 * @brief Class for accessing variables from the DataWarehouse
 * (ScalarField implementation)
 *
 * detail implementation of DataWarehouse variable wrapping
 *
 * @remark constant view must use fields with const value type
 *
 * @tparam T type of the field value at each point
 * @tparam VAR type of variable representation
 * @tparam DIM problem dimension
 * @tparam GN number of ghosts required
 */
template<typename T, VarType VAR, DimType DIM, size_t GN>
class dw_view < ScalarField<T>, VAR, DIM, GN >
    : virtual public view < ScalarField<T> >
{
public: // STATIC MEMBERS

    /// Default value for use_ghosts when retrieving data
    static constexpr bool use_ghosts_dflt = false;

private: // STATIC MEMBERS

    /// Type of ghosts
    static constexpr Ghost::GhostType GT = GN ? get_var<VAR>::ghost_type : Ghost::None;

private: // TYPES

    /// Type of field
    using Field = ScalarField<T>;

    /// Non const type of the field value
    using V = typename std::remove_const<T>::type;

private: // FRIENDS

    /// Grant virtual_view's (with virtual support) wrapper access
    friend virtual_view<dw_view, Field>;

private: // MEMBERS

    /// Variable label in the DataWarehouse
    typename Field::label_type m_label;

    /// Material index in the DataWarehouse
    const int m_material;

    /// Grid Level
    const Level * m_level;

    /// Underlying Grid Variable
    Variable<VAR, T> * m_variable;

    /// View to the variable
    /// (points the variable itself or to the kokkos view of it for gpu builds)
#ifndef UINTAH_ENABLE_KOKKOS
    Variable<VAR, T> * m_view;
#else
    KokkosView3<T> * m_view;
#endif

private: // COPY CONSTRUCTOR

    /**
     * @brief Constructor
     *
     * Instantiate a copy of a given view
     *
     * @param copy source view for copying
     * @param deep if true inner grid variable is copied as well otherwise the
     * same grid variable is referenced
     */
    dw_view (
        const dw_view * copy,
        bool deep
    ) : m_label ( copy->m_label ),
        m_material ( copy->m_material ),
        m_level ( copy->m_level ),
        m_variable ( ( copy->m_variable && deep ) ? scinew Variable<VAR, T> ( *copy->m_variable ) : nullptr ),
#ifndef UINTAH_ENABLE_KOKKOS
        m_view ( m_variable )
#else
        m_view ( m_variable ? m_variable->getKokkosView() : nullptr )
#endif
    {}

private: // METHODS

    /**
     * @brief Create grid variable over a patch
     *
     * Instantiate the underlying grid variable a retrieve patch data
     * (const implementation)
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param patch grid patch to retrieve data for
     * @param use_ghosts if ghosts value are to be retrieved
     * @return new instance of Variable
     */
    template<bool is_const>
    typename std::enable_if<is_const, Variable<VAR, T> *>::type
    create_patch_variable (
        DataWarehouse * dw,
        const Patch * patch,
        bool use_ghosts
    )
    {
        Variable<VAR, T> * var = scinew Variable<VAR, T>;
        if ( use_ghosts )
            dw->get ( *var, m_label, m_material, patch, GT, GN );
        else
            dw->get ( *var, m_label, m_material, patch, Ghost::None, 0 );
        return var;
    }

    /**
     * @brief Create grid variable over a patch
     *
     * Instantiate the underlying grid variable a retrieve patch data
     * (non const implementation)
     *
     * @remark if variable does not exists in dw it is allocated therein
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param patch grid patch to retrieve data for
     * @param use_ghosts if ghosts value are to be retrieved
     * @return new instance of Variable
     */
    template<bool is_const>
    typename std::enable_if < !is_const, Variable<VAR, T> * >::type
    create_patch_variable (
        DataWarehouse * dw,
        const Patch * patch,
        bool use_ghosts
    )
    {
        Variable<VAR, T> * var = scinew Variable<VAR, T>;
        if ( use_ghosts )
        {
            if ( dw->exists ( m_label, m_material, patch ) )
                dw->getModifiable ( *var, m_label, m_material, patch, GT, GN );
            else
                dw->allocateAndPut ( *var, m_label, m_material, patch, GT, GN );
        }
        else
        {
            if ( dw->exists ( m_label, m_material, patch ) )
                dw->getModifiable ( *var, m_label, m_material, patch, Ghost::None, 0 );
            else
                dw->allocateAndPut ( *var, m_label, m_material, patch, Ghost::None, 0 );
        }
        return var;
    }

    /**
     * @brief Create grid variable over a region
     *
     * Instantiate the underlying grid variable a retrieve region data
     * (const implementation)
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param level grid level from which retrieve data
     * @param low lower bound of the region to retrieve
     * @param high higher bound of the region to retrieve
     * @param use_ghosts if ghosts value are to be retrieved
     * @return new instance of Variable
     */
    template<bool is_const>
    typename std::enable_if<is_const, Variable<VAR, T> *>::type
    create_region_variable (
        DataWarehouse * dw,
        const Level * level,
        const IntVector & low,
        const IntVector & high,
        bool use_ghosts
    )
    {
        Variable<VAR, T> * var = scinew Variable<VAR, T>;
        if ( use_ghosts )
            dw->getRegion ( *var, m_label, m_material, level, low - get_dim<DIM>::template scalar_vector<GN>(), high + get_dim<DIM>::template scalar_vector<GN>() );
        else
            dw->getRegion ( *var, m_label, m_material, level, low, high );
        return var;
    }

    /**
     * @brief Create grid variable over a region
     *
     * Instantiate the underlying grid variable a retrieve region data
     * (non const implementation)
     *
     * @remark does nothing since views of grid variables over region must not
     * allow data modifications
     *
     * @param dw unused
     * @param level unused
     * @param low unused
     * @param high unused
     * @param use_ghosts unused
     * @return null pointer
     */
    template<bool is_const>
    typename std::enable_if < !is_const, Variable<VAR, T> * >::type
    create_region_variable (
        DataWarehouse * _DOXYARG ( dw ),
        const Level * _DOXYARG ( level ),
        const IntVector & _DOXYARG ( low ),
        const IntVector & _DOXYARG ( high ),
        bool _DOXYARG ( use_ghosts )
    )
    {
        ASSERTFAIL ( "cannot create non const view over region" );
        return nullptr;
    }

public: // CONSTRUCTORS/DESTRUCTOR

    /**
     * @brief Constructor
     *
     * Instantiate a view without gathering info from the DataWarehouse
     *
     * @param label variable label
     * @param material material index
     */
    dw_view (
        const typename Field::label_type & label,
        int material
    ) : m_label ( label ),
        m_material ( material ),
        m_level ( nullptr ),
        m_variable ( nullptr ),
        m_view ( nullptr )
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
    dw_view (
        DataWarehouse * dw,
        const typename Field::label_type & label,
        int material,
        const Patch * patch,
        bool use_ghosts = use_ghosts_dflt
    ) : m_label ( label ),
        m_material ( material ),
        m_level ( patch->getLevel() ),
        m_variable ( create_patch_variable<std::is_const<T>::value> ( dw, patch, use_ghosts ) ),
#ifndef UINTAH_ENABLE_KOKKOS
        m_view ( m_variable )
#else
        m_view ( m_variable->getKokkosView() )
#endif
    {}

    /// Destructor
    ~dw_view()
    {
        delete m_variable;
    }

    /// Prevent copy (and move) constructor
    dw_view ( const dw_view & ) = delete;

    /// Prevent copy (and move) assignment
    /// @return deleted
    dw_view & operator= ( const dw_view & ) = delete;

public: // VIEW METHODS

    /**
     * @brief Retrieve values from the DataWarehouse for a given patch
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param patch grid patch to retrieve data for
     * @param use_ghosts if ghosts value are to be retrieved
     */
    virtual void
    set (
        DataWarehouse * dw,
        const Patch * patch,
        bool use_ghosts
    ) override
    {
        delete m_variable;
        m_level = patch->getLevel();
        m_variable = create_patch_variable<std::is_const<T>::value> ( dw, patch, use_ghosts );
#ifndef UINTAH_ENABLE_KOKKOS
        m_view = m_variable;
#else
        m_view = m_variable->getKokkosView();
#endif
    };

    /**
     * @brief Retrieve values from the DataWarehouse for a given region
     *
     * @param dw DataWarehouse from which data is retrieved
     * @param level grid level from which retrieve data
     * @param low lower bound of the region to retrieve
     * @param high higher bound of the region to retrieve
     * @param use_ghosts if ghosts value are to be retrieved
     */
    virtual void
    set (
        DataWarehouse * dw,
        const Level * level,
        const IntVector & low,
        const IntVector & high,
        bool use_ghosts
    ) override
    {
        delete m_variable;
        m_level = level;
        m_variable = create_region_variable<std::is_const<T>::value> ( dw, level, low, high, use_ghosts );
#ifndef UINTAH_ENABLE_KOKKOS
        m_view = m_variable;
#else
        m_view = m_variable->getKokkosView();
#endif
    };

    /**
     * @brief Get a copy of the view
     *
     * @param deep if true inner grid variable is copied as well otherwise the
     * same grid variable is referenced
     *
     * @return new view instance
     */
    virtual inline view<Field> *
    clone (
        bool deep
    )
    const override
    {
        return scinew dw_view ( this, deep );
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
    virtual inline view<Field> *
    clone (
        bool deep,
        const IntVector & offset
    )
    const override
    {
        return scinew virtual_view<dw_view, Field> ( this, deep, offset );
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
        return {{ m_variable->getLowIndex(), m_variable->getHighIndex() }};
    };

    /**
     * @brief Check if the view has access to the position with index id
     *
     * @param id position index
     * @return check result
     */
    virtual bool
    is_defined_at (
        const IntVector & id
    ) const override
    {
        IntVector low { m_variable->getLowIndex() }, high { m_variable->getHighIndex() };

        return ( low[X] <= id[X] && id[X] < high[X] ) &&
               ( low[Y] <= id[Y] && id[Y] < high[Y] ) &&
               ( low[Z] <= id[Z] && id[Z] < high[Z] );
    };

    /**
     * @brief Get/Modify value at position with index id (modifications are allowed if T is non const)
     *
     * @param id position index
     * @return reference to field value at id
     */
    virtual T &
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
    virtual V
    operator[] (
        const IntVector & id
    ) const override
    {
        return ( *m_view ) [id];
    };

public: // DW METHODS

    /**
     * @brief initialize variable in the DW
     *
     * set the field equal to value at each point in the domain
     *
     * @remark T must not be const
     *
     * @param value initialization value
     */
    inline void
    initialize (
        const V & value
    ) const
    {
        static_assert ( !std::is_const<T>::value, "cannot initialize const view" );
        ASSERTMSG ( m_variable, "cannot initialize view of uninitialized variable" );
        m_variable->initialize ( value );
    }

}; // dw_view

/**
 * @brief Class for accessing variables from the DataWarehouse
 * (VectorField implementation)
 *
 * detail implementation of DataWarehouse variable wrapping
 *
 * @remark constant view must use fields with const value type
 *
 * @tparam T type of each component of the field value at each point
 * @tparam N dimension of the vetor field
 * @tparam VAR type of variable representation
 * @tparam DIM problem dimension
 * @tparam GN number of ghosts required
 */
template<typename T, size_t N, VarType VAR, DimType DIM, size_t GN>
class dw_view < VectorField<T, N>, VAR, DIM, GN >
    : virtual public view < VectorField<T, N> >
    , virtual public view_array < dw_view < ScalarField<T>, VAR, DIM, GN >, ScalarField<T>, N >
{
public: // STATIC MEMBERS


private: // TYPES

    /// Type of field
    using Field = VectorField<T, N>;

    /// Type of View of each component
    using View = dw_view < ScalarField<T>, VAR, DIM, GN >;

    /// Non const type of the field value
    using V = typename std::remove_const<T>::type;

private: // COPY CONSTRUCTOR

    /**
     * @brief Constructor
     *
     * Instantiate a copy of a given view
     *
     * @param copy source view for copying
     * @param deep if true inner grid variable is copied as well otherwise the
     * same grid variable is referenced
     */
    dw_view (
        const dw_view * copy,
        bool deep
    )
    {
        for ( size_t i = 0; i < N; ++i )
        {
            const auto & v = ( *copy ) [i];
            this->m_view_ptr[i] = v.clone ( deep );
        }
    }

public: // CONSTRUCTORS/DESTRUCTOR

    /**
     * @brief Constructor
     *
     * Instantiate view components without gathering info from the DataWarehouse
     *
     * @param label list of variable labels for each component
     * @param material material index
     */
    dw_view (
        const typename Field::label_type & label,
        int material
    )
    {
        for ( size_t i = 0; i < N; ++i )
            this->m_view_ptr[i] = scinew View ( label[i], material );
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
     * @param use_ghosts if ghosts value are to be retrieved
     */
    dw_view (
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
    virtual ~dw_view ()
    {
        for ( auto view : this->m_view_ptr )
            delete view;
    }

    /// Prevent copy (and move) constructor
    dw_view ( const dw_view & ) = delete;

    /// Prevent copy (and move) assignment
    /// @return deleted
    dw_view & operator= ( const dw_view & ) = delete;

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
    )
    const override
    {
        return scinew dw_view ( this, deep );
    };

public: // DW METHODS

    /**
     * @brief initialize variable in the DW
     *
     * set the field equal to value at each point in the domain
     *
     * @remark T must not be const
     *
     * @param value initialization value
     */
    void
    initialize (
        const V & value
    ) const
    {
        for ( auto & view : this->m_view_ptr )
            dynamic_cast < View * > ( view )->initialize ( value );
    }

    /// Resolve ambiguous operator[]
    using view_array < View, ScalarField<T>, N >::operator [];

}; // dw_view

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif //Packages_Uintah_CCA_Components_PhaseField_DataWarehouse_detail_dw_view_h

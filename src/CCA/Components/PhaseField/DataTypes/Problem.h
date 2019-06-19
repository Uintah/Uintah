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
 * @file CCA/Components/PhaseField/DataTypes/Problem.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_Problem_h
#define Packages_Uintah_CCA_Components_PhaseField_Problem_h

#include <CCA/Components/PhaseField/DataTypes/BCInfo.h>
#include <CCA/Components/PhaseField/Views/View.h>
#include <CCA/Components/PhaseField/Views/FDView.h>
#include <CCA/Components/PhaseField/DataWarehouse/DWFDView.h>
#include <CCA/Components/PhaseField/BoundaryConditions/BCFDViewFactory.h>

namespace Uintah
{
namespace PhaseField
{

/**
 * @brief PhaseField Problem
 *
 * It represent one portion of the global problem on which all implementations
 * required to solve a timestep are fixed (i.e: there is no need to check boundary
 * conditions/variables extents/neighbors...)
 * @tparam VAR type of variable representation
 * @tparam STN finite-difference stencil
 * @tparam Field list of type of fields (ScalarField < T > or VectorialField < T, N >)
 */
template <VarType VAR, StnType STN, typename... Field>
class Problem
{
public: // STATIC MEMBERS

    /// Problem data representation
    static constexpr VarType Var = VAR;

    /// Problem dimension
    static constexpr DimType Dim = get_stn<STN>::dim;

    /// Finite difference stencil
    static constexpr StnType Stn = STN;


    /// class name as used by ApplicationFactory
    static const std::string Name;


public: // TYPE HELPERS

    /**
     * @brief Field Type Helper
     *
     * Allow to get the type of the Field represented by the I-th Problem variable
     *
     * @tparam I index of the variable within the Problem
     * @tparam J component indexes of the variable (used with higher dimensional fields)
     */
    template<size_t I, size_t... J>
    struct get_field
    {
        /// Type hekpof the Field represented by the [J..]-th component of the I-th Problem variable
        using type = typename std::tuple_element < I, std::tuple<Field...> >::type::template elem_type<J...>;
    };

    /**
     * @brief Field Type Helper (ScalarField implementation)
     *
     * Allow to get the type of the Field represented by the I-th Problem variable
     *
     * @tparam I index of the variable within the Problem
     */
    template<size_t I>
    struct get_field<I>
    {
        /// Type helper of the Field represented by the I-th Problem variable
        using type = typename std::tuple_element < I, std::tuple<Field...> >::type;
    };

    /// View type for the I-th Problem variable (J-th component)
    template<size_t I, size_t... J> using get_view_type = View < typename get_field<I, J...>::type >;

    /// FDView type for the I-th Problem variable (J-th component)
    template<size_t I, size_t... J> using get_fd_view_type = FDView < typename get_field<I, J...>::type, STN >;


private: // MEMBERS

    /// Grid Level on wich the Problem is defined
    const Level * m_level;

    /// Lower bound of the region where the problem is defined
    IntVector m_low;

    /// Higher bound of the region where the problem is defined
    IntVector m_high;

    /// Faces on which the problem is defined (empty for inner problems)
    const std::list< Patch::FaceType > m_face;

    /// Labels for problem variables
    const std::tuple< typename Field::label_type... > m_labels;

    /// Container for the finite-differences implementations for the problem variables
    const std::tuple< std::unique_ptr< FDView<Field, STN> > ... > m_fd_view;

private: // INDEXED CONSTRUCTOR

    /**
     * @brief Boundary Problem constructor
     *
     * Instantiate all subviews required to handle all given variables in the
     * given region without retrieving data.
     *
     * @remark There is no need to recreate these
     * views until the geometry is unchanged
     *
     * @tparam I indices for variables
     * @param unused to allow template argument deduction
     * @param labels list of the labels for the variables to handle
     * @param subproblems_label label for subproblems in the DataWarehouse
     * @param material index in the DataWarehouse
     * @param level grid level
     * @param low lower bound of the region to handle
     * @param high higher bound of the region to handle
     * @param faces list of faces the new (sub)problem have to handle
     * @param bcs boundary info for each variable and face
     */
    template < size_t ... I >
    Problem (
        index_sequence<I...> _DOXYARG ( unused ),
        const typename Field::label_type & ... labels,
        const VarLabel * subproblems_label,
        int material,
        const Level * level,
        IntVector low,
        IntVector high,
        const std::list<Patch::FaceType> & faces,
        std::vector< BCInfo<Field> > ... bcs
    ) : m_level ( level ),
        m_low ( low ),
        m_high ( high ),
        m_face ( faces ),
        m_labels { labels... },
             m_fd_view ( std::unique_ptr< FDView<Field, STN> > { BCFDViewFactory<Problem, I>::create ( labels, subproblems_label, material, level, faces, bcs ) } ... )
    {}

public: // CONSTRUCTORS/DESTRUCTOR

    /**
     * @brief Inner Problem constructor
     *
     * Instantiate all the views required to handle all given variables in the
     * given region without retrieving data.
     *
     * @remark There is no need to recreate the
     * views until the geometry is unchanged
     *
     * @param labels list of the labels for the variables to handle
     * @param material index in the DataWarehouse
     * @param level grid level
     * @param low lower bound of the region to handle
     * @param high higher bound of the region to handle
     */
    Problem (
        const typename Field::label_type & ... labels,
        int material,
        const Level * level,
        IntVector low,
        IntVector high
    ) : m_level ( level ),
        m_low ( low ),
        m_high ( high ),
        m_face (),
        m_labels { labels... },
             m_fd_view { std::unique_ptr< FDView<Field, STN> > { scinew DWFDView<Field, STN, VAR> ( labels, material, level ) } ... }
    {}

    /**
     * @brief Boundary constructor
     *
     * Instantiate all subviews required to handle all given variables in the
     * given region without retrieving data. There is no need to recreate the
     * views until the geometry is unchanged
     *
     * @param labels list of the labels for the variables to handle
     * @param subproblems_label label for subproblems in the DataWarehouse
     * @param material index in the DataWarehouse
     * @param level grid level
     * @param low lower bound of the region to handle
     * @param high higher bound of the region to handle
     * @param faces list of faces the new (sub)problem have to handle
     * @param bcs boundary info for each variable and face
     */
    Problem (
        const typename Field::label_type & ... labels,
        const VarLabel * subproblems_label,
        int material,
        const Level * level,
        IntVector low,
        IntVector high,
        const std::list<Patch::FaceType> & faces,
        std::vector< BCInfo<Field> > ... bcs
    ) : Problem ( make_index_sequence<sizeof... ( labels ) > {}, labels..., subproblems_label, material, level, low, high, faces, bcs... )
    {
    }

    /**
     * @brief Get Codimension
     *
     * Return the codimension of the variety on which the problem is defined
     *
     * @return the number of faces that define the variety
     */
    inline int
    get_codim()
    const
    {
        return m_face.size();
    }

    /**
     * @brief Get Problem range
     *
     * @return Range on which the Problem is defined
     */
    inline BlockRange
    get_range()
    const
    {
        return { m_low, m_high };
    }

    /**
     * @brief Get Problem lower bound
     *
     * @return low index
     */
    inline const IntVector &
    get_low()
    const
    {
        return m_low;
    }

    /**
     * @brief Get Problem higher bound
     *
     * @return high index
     */
    inline const IntVector &
    get_high()
    const
    {
        return m_high;
    }

    /**
     * @brief Get problem faces
     *
     * @return list of faces which the problem region belongs to
     */
    inline
    const std::list<Patch::FaceType> &
    get_faces()
    const
    {
        return m_face;
    }

    /**
     * @brief Get a View
     *
     * Get the View of the I-th Problem variable without retrieving data
     *
     * @tparam I index of the variable within the Problem
     * @return a view to the variable
     */
    template<size_t I>
    inline get_view_type<I> &
    get_view ()
    const
    {
        return * std::get<I> ( m_fd_view )->get_view();
    }

    /**
     * @brief Get a View
     *
     * Get the View of the I-th Problem variable and retrieve the data from dw
     *
     * @tparam I index of the variable within the Problem
     * @param dw DataWarehouse to use for retrieving data
     * @return a view to the variable
     */
    template<size_t I>
    inline get_view_type<I> &
    get_view (
        DataWarehouse * dw
    ) const
    {
        get_view_type<I> * view { std::get<I> ( m_fd_view )->get_view() };
        view->set ( dw, m_level, m_low, m_high );
        return *view;
    }

    /**
     * @brief Get a View
     *
     * Get the View of the J-th component of the I-th Problem variable without
     * retrieving data
     *
     * @tparam I index of the variable within the Problem
     * @tparam J index of the component within the variable
     * @return a view to the variable
     */
    template<size_t I, size_t J>
    inline get_view_type<I, J> &
    get_view (
    ) const
    {
        return * std::get<I> ( m_fd_view )->operator[] ( J ).get_view();
    }

    /**
     * @brief Get a View
     *
     * Get the View of the J-th component of the I-th Problem variable and
     * retrieve the data from dw
     *
     * @tparam I index of the variable within the Problem
     * @tparam J index of the component within the variable
     * @param dw DataWarehouse to use for retrieving data
     * @return a view to the variable
     */
    template<size_t I, size_t J>
    inline get_view_type<I, J> &
    get_view (
        DataWarehouse * dw
    ) const
    {
        get_view_type<I, J> * view = std::get<J> ( std::get<I> ( m_fd_view ) )->get_view();
        view->set ( dw, m_level, m_low, m_high );
        return *view;
    }

    /**
     * @brief Get a FDView
     *
     * Get the FDView of the I-th Problem variable without retrieving data
     *
     * @tparam I index of the variable within the Problem
     * @return a view to the variable that implements finite-differences
     */
    template< size_t I >
    inline get_fd_view_type<I> &
    get_fd_view ()
    const
    {
        return * std::get<I> ( m_fd_view ).get();
    }

    /**
     * @brief Get a FDView
     *
     * Get the FDView of the I-th Problem variable and retrieves the data from dw
     *
     * @tparam I index of the variable within the Problem
     * @param dw DataWarehouse to use for retrieving data
     * @return a view to the variable that implements finite-differences
     */
    template< size_t I >
    inline get_fd_view_type<I> &
    get_fd_view (
        DataWarehouse * dw
    ) const
    {
        get_fd_view_type<I> * view { std::get<I> ( m_fd_view ).get() };
        view->set ( dw, m_level, m_low, m_high );
        return *view;
    }

    /**
     * @brief Get a FDView
     *
     * Get the FDView of the J-th component of the I-th Problem variable
     * without retrieving data
     *
     * @tparam I index of the variable within the Problem
     * @tparam J index of the component within the variable
     * @return a view to the variable that implements finite-differences
     */
    template<size_t I, size_t J>
    inline get_fd_view_type<I, J> &
    get_fd_view ()
    const
    {
        return * std::get<I> ( m_fd_view ).get();
    }

    /**
     * @brief Get a FDView
     *
     * Get the FDView of the J-th component of the I-th Problem variable and
     * retrieves the data from dw
     *
     * @tparam I index of the variable within the Problem
     * @tparam J index of the component within the variable
     * @param dw DataWarehouse to use for retrieving data
     * @return a view to the variable that implements finite-differences
     */
    template<size_t I, size_t J>
    inline get_fd_view_type<I, J> &
    get_fd_view (
        DataWarehouse * dw
    ) const
    {
        get_fd_view_type<I> * view { std::get<I> ( m_fd_view ).get() };
        view->set ( dw, m_level, m_low, m_high );
        return *view;
    }

}; // class Problem

/**
 * @brief Insertion operator between Problem and std::ostream
 *
 * Used for debug purposes
 *
 * @param os output stream reference
 * @param p problem to output
 * @return output stream reference for concatenation
 */
template <PhaseField::VarType VAR, PhaseField::StnType STN, typename... T>
std::ostream &
operator<< (
    std::ostream & os,
    const PhaseField::Problem<VAR, STN, T...> & p
)
{
    if ( p.get_codim() )
    {
        os << "Boundary Problem (";
        auto faces = p.get_faces();
        for ( const auto & f : faces )
        {
            os << f;
            if ( f != faces.back() ) os << ",";
        }
        os << ") range " << p.get_range();
    }
    else
        os << "Inner Problem : range " << p.get_range();
    return os;
}

} // namespace PhaseFieldSTNSTN
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_Problem_h

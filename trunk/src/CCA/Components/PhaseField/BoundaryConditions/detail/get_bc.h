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
 * @file CCA/Components/PhaseField/BoundaryConditions/detail/get_bc.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_get_bc_h
#define Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_get_bc_h

#include <CCA/Components/PhaseField/DataTypes/BCInfo.h>

#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Components/PhaseField/Util/Expressions.h>
#include <CCA/Components/PhaseField/DataTypes/ScalarField.h>
#include <CCA/Components/PhaseField/DataTypes/VectorField.h>

namespace Uintah
{
namespace PhaseField
{
namespace detail
{

/**
 * @brief Get boundary conditions on one face static functor
 *
 * @tparam Field type of field (ScalarField < T > or VectorField < T, N >)
 */
template<typename Field>
class get_bc
#if _DOXYGEN
{
    /// Execute the functor
    /// @return Retrieved boundary conditions information
    inline static BCInfo<Field> exec (
        Patch const *,
        const Patch::FaceType &,
        const int &,
        const int &,
        const std::array<const VarLabel *, N> &,
        const std::map<std::string, FC> *,
        bool &
    ) = 0;
}
#endif
;

/**
 * @brief Get boundary conditions on one face static functor
 * (ScalarField implementation)
 *
 * Allows to get the boundary value for a given variable label on a given
 * boundary face (face, child) of the given patch.
 *
 * @remark Refactor of getBCValue function in Core/Grid/BoundaryConditions/BCUtils.h
 *
 * @implements get_bc < Field >
 */
template<typename T>
class get_bc< ScalarField<T> >
{
    /// Type of field
    using Field = ScalarField<T>;

public:
    /**
     * @brief Execute the functor
     *
     * If patch getBCType is Patch::Coarse it also populate the output c2f field
     * with the relevant value from the c2f map in input
     * Otherwise, if getBCType is specified but not "simmetry", retrieves the
     * values required to enforce the boundary conditions specified in input.
     *
     * @param patch grid patch to be checked
     * @param face face to check
     * @param child bc index (it is possible to specify multiple bc for the same variable)
     * @param material problem material index
     * @param label variable label to check
     * @param c2f which fine/coarse interface conditions to use on each variable
     * @param[in,out] flag flag to check if any bc is applied to the given face
     * @return Retrieved boundary conditions information
     */
    BCInfo<Field>
    static exec (
        Patch const * patch,
        const Patch::FaceType & face,
        const int & child,
        const int & material,
        const VarLabel * const & label,
        const std::map<std::string, FC> * c2f,
        bool & flag
    )
    {
        BCInfo<Field> bci { T{}, BC::Unknown, FC::None };

        const std::string & desc = label->getName();
        if ( patch->getBCType ( face ) == Patch::Coarse )
        {
            ASSERTMSG ( c2f, "null c2f map in get_bc" );
            ASSERTMSG ( c2f->find ( desc ) != c2f->end(), "desc not found in c2f map in get_bc" );
            bci.bc = BC::FineCoarseInterface;
            bci.c2f = c2f->at ( desc );
        }
        else if ( patch->getBCType ( face ) == Patch::Neighbor )
        {
            bci.bc = BC::None;
        }
        else if ( const BCDataArray * data = patch->getBCDataArray ( face ) )
        {
            if ( const BoundCondBase * info = data->getBoundCondData ( material, desc, child ) )
            {
                if ( const BoundCond<double> * _info = dynamic_cast<const BoundCond<double> *> ( info ) )
                {
                    bci.bc = str_to_bc ( _info->getBCType() );
                    bci.value = _info ->getValue();
                }
                delete info;
            }
            else if ( const BoundCondBase * info = data->getBoundCondData ( material, "Symmetric", child ) )
            {
                if ( info->getBCType() == "symmetry" )
                    bci.bc = BC::Symmetry;
                delete info;
            }
        }

        switch ( bci.bc )
        {
        case BC::Unknown:
        {
            std::ostringstream msg;
            msg << "\n ERROR: Unknown BC condition on patch " << *patch << " face " << face << std::endl;
            SCI_THROW ( InvalidValue ( msg.str(), __FILE__, __LINE__ ) );
        }
        break;
        case BC::None:
            break;
        default:
            flag = true;
        }
        return bci;
    }
};

/**
 * @brief Get boundary conditions on one face static functor
 * (VectorField implementation)
 *
 * Allows to get the boundary value for a given variable label on a given
 * boundary face (face, child) of the given patch.
 *
 * @implements get_bc < Field >
 */
template<typename T, size_t N>
class get_bc < VectorField<T, N> >
{
    /// Type of field
    using Field = VectorField<T, N>;

    /**
     * @brief Execute the functor (internal indexed implementation)
     *
     * For each component call the ScalarField implementation
     *
     * @tparam J list of components' indices
     * @param patch grid patch to be checked
     * @param face face to check
     * @param child bc index (it is possible to specify multiple bc for the same variable)
     * @param material problem material index
     * @param label variable label to check
     * @param c2f which fine/coarse interface conditions to use on each variable
     * @param[in,out] flag flag to check if any bc is applied to the given face
     * @return Retrieved boundary conditions information
     */
    template < size_t ... J >
    inline static BCInfo<Field>
    exec (
        index_sequence<J...>,
        Patch const * patch,
        const Patch::FaceType & face,
        const int & child,
        const int & material,
        const typename Field::label_type & label,
        const std::map<std::string, FC> * c2f,
        bool & flag
    )
    {
        return { get_bc< ScalarField<T> >::exec ( patch, face, child, material, std::get<J> ( label ), c2f, flag )... };
    }

public:
    /**
     * @brief Execute the functor
     *
     * For each component call the internal indexed implementation
     *
     * @param patch grid patch to be checked
     * @param face face to check
     * @param child bc index (it is possible to specify multiple bc for the same variable)
     * @param material problem material index
     * @param label variable label to check
     * @param c2f which fine/coarse interface conditions to use on each variable
     * @param[in,out] flag flag to check if any bc is applied to the given face
     * @return Retrieved boundary conditions information
     */
    inline static BCInfo<Field>
    exec (
        Patch const * patch,
        const Patch::FaceType & face,
        const int & child,
        const int & material,
        const std::array<const VarLabel *, N> & label,
        const std::map<std::string, FC> * c2f,
        bool & flag
    )
    {
        return exec ( make_index_sequence<N> {}, patch, face, child, material, label, c2f, flag );
    }
};

} // namespace detail
} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_detail_get_bc_h

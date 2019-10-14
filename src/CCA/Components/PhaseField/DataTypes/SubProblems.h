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
 * @file CCA/Components/PhaseField/DataTypes/SubProblems.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_DataTypes_SubProblems_h
#define Packages_Uintah_CCA_Components_PhaseField_DataTypes_SubProblems_h

#include <CCA/Components/PhaseField/DataTypes/Problem.h>

namespace Uintah
{
namespace PhaseField
{

template < VarType VAR, StnType STN> class BCInterface;

/**
 * @brief PhaseField Problem Container
 *
 * Stores the list of all Problem's in which a Patch is partitioned
 *
 * @tparam Problem type of PhaseField problem
 */
template<typename Problem> struct SubProblems;

/**
 * @brief PhaseField Problem Container (Problem implementation)
 *
 * @tparam VAR type of variable representation
 * @tparam STN finite-difference stencil
 * @tparam Field list of type of fields (ScalarField < T> or VectorField < T, N >)
 */
template<VarType VAR, StnType STN, typename... Field>
struct SubProblems < Problem<VAR, STN, Field...> >
        : public RefCounted
        , public std::list < Problem<VAR, STN, Field...> >
{
    /**
     * @brief Constructor
     *
     * Instantiate the list the views required to handle all given variables
     * over the given patch without retrieving data.
     *
     * @remark There is no need to recreate the
     * views until the geometry is unchanged
     *
     * @param bci interface to problem for bc related funcitonalities
     * @param labels list of the labels for the variables to handle
     * @param subproblems_label label for subproblems in the DataWarehouse
     * @param material index in the DataWarehouse
     * @param patch grid level
     * @param c2f mapping between variable names and C2F condition for amr grids
     */
    SubProblems (
        const BCInterface<VAR, STN> * bci,
        const typename Field::label_type & ...  labels,
        const VarLabel * subproblems_label,
        int material,
        const Patch * patch,
        const std::map<std::string, FC> * c2f = nullptr
    ) : std::list<Problem<VAR, STN, Field...>> ( bci->template partition_patch <Field...> ( labels..., subproblems_label, material, patch, c2f ) )
    {
    };
}; // struct SubProblems

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_DataTypes_SubProblems_h

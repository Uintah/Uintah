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
 * @file CCA/Components/PhaseField/BoundaryConditions/BCFDViewFactory.h
 * @author Jon Matteo Church [j.m.church@leeds.ac.uk]
 * @date 2018/12
 */

#ifndef Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_BCViewFactory_h
#define Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_BCViewFactory_h

#include <CCA/Components/PhaseField/Util/Definitions.h>
#include <CCA/Components/PhaseField/Util/Expressions.h>
#include <CCA/Components/PhaseField/Factory/Base.h>
#include <CCA/Components/PhaseField/Factory/Factory.h>
#include <CCA/Components/PhaseField/DataTypes/BCInfo.h>
#include <CCA/Components/PhaseField/DataTypes/ScalarField.h>
#include <CCA/Components/PhaseField/DataTypes/VectorField.h>
#include <CCA/Components/PhaseField/Views/FDView.h>

#include <Core/Exceptions/ProblemSetupException.h>

namespace Uintah
{
namespace PhaseField
{

/// Factory creator for BCFDView
template<typename Field, StnType STN> using BCFactoryFDView = Factory < FDView<Field, STN>, const typename Field::label_type &, const VarLabel *, int, const Level *, const std::vector< BCInfo<Field> > & >;

/// Factory base for BCFDView
template<typename Field, StnType STN> using BCBaseFDView = Base< FDView<Field, STN> >;

/**
 * @brief Factory creator implementation for BCFDView
 *
 * @tparam Problem type of PhaseField problem
 * @tparam I index of Field within Problem
 */
template<typename Problem, size_t I>
class BCFDViewFactory
{
    /// Type of field
    using Field = typename Problem::template get_field<I>::type;

private: // STATIC MEMBERS

    /// Problem variable representation
    static constexpr VarType VAR = Problem::Var;

    /// Finite-difference stencil
    static constexpr StnType STN = Problem::Stn;

public:

    /**
     * @brief factory static method for intantiating a BCView< T, S >
     *
     * @param label label of variable in the DataWarehouse
     * @param subproblems_label label of subproblems in the DataWarehouse
     * @param material index of material in the DataWarehouse
     * @param level level on which data is retrieved
     * @param faces list of the faces where to impose boundary conditions
     * @param bcs vector with info on the boundary conditions (ordered as in faces)
     * @return FDView< T, S >* newly instantiated view
     */
    static FDView<Field, STN> *
    create (
        const typename Field::label_type & label,
        const VarLabel * subproblems_label,
        int material,
        const Level * level,
        const std::list<Patch::FaceType> & faces,
        const std::vector < BCInfo<Field> > & bcs
    )
    {
        BCBaseFDView<Field, STN> * ptr = nullptr;
        std::string bcview = Problem::Name + "|" + std::to_string ( I ) + "|" + var_to_str ( VAR ) + "|";

        auto face = faces.begin();
        auto bci = bcs.begin();
        while ( face != faces.end() && bci != bcs.end() )
        {
            bcview += Patch::getFaceName ( *face ) + "|" + ( bci->bc == BC::FineCoarseInterface ? fc_to_str ( bci->c2f ) : bc_to_str ( bci->bc ) ) + "|";
            ++face;
            ++bci;
        }
        ptr = BCFactoryFDView<Field, STN>::Create ( bcview, label, subproblems_label, material, level, bcs );
        if ( !ptr )
        {
            SCI_THROW ( ProblemSetupException ( "Cannot Create BCView '" + bcview + "'", __FILE__, __LINE__ ) );
        }

        return dynamic_cast< FDView<Field, STN>* > ( ptr );
    }

}; // class BCFDViewFactory

} // namespace PhaseField
} // namespace Uintah

#endif // Packages_Uintah_CCA_Components_PhaseField_BoundaryConditions_BCViewFactory_h

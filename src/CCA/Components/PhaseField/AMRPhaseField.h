/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef Packages_Uintah_CCA_Components_PhaseField_AMRPhaseField_h
#define Packages_Uintah_CCA_Components_PhaseField_AMRPhaseField_h

#include <Core/Grid/Variables/PerPatch.h>
#include <CCA/Components/PhaseField/PhaseField.h>
#include <CCA/Components/Regridder/PerPatchVars.h>

namespace Uintah
{

namespace PF
{

using FlagVariable = CCVariable<int>;

#ifdef UINTAH_ENABLE_KOKKOS
using FlagView = KokkosView3<int>;
#else
using FlagView = FlagVariable & ;
#endif

template<VariableType VariableType>
IntVector map_to_coarser ( Level const * /*level*/, IntVector const & /*i*/ )
{
    static_assert ( static_false<VariableType>::value, "map_to_coarser<VariableType> not implemented" );
    return { 0, 0, 0 };
}

template<>
IntVector map_to_coarser<CellCentered> ( Level const * level, IntVector const & i )
{
    return level->mapCellToCoarser ( i );
}

template<>
IntVector map_to_coarser<NodeCentered> ( Level const * level, IntVector const & i )
{
    return level->mapNodeToCoarser ( i );
}

template<VariableType VariableType>
IntVector map_to_finer ( Level const * level, IntVector const & i )
{
    static_assert ( static_false<VariableType>::value, "map_to_finer<VariableType> not implemented" );
    return { 0, 0, 0 };
}

template<>
IntVector map_to_finer<CellCentered> ( Level const * level, IntVector const & i )
{
    return level->mapCellToFiner ( i );
}

template<>
IntVector map_to_finer<NodeCentered> ( Level const * level, IntVector const & i )
{
    return level->mapNodeToFiner ( i );
}

template<VariableType VariableType, int NumGhosts, int Dimension>
void refine ( int i_fine, int j_fine, int k_fine, Level const * level_fine, Level const * level_coarse, ConstView<VariableType> psi_coarse, ConstView<VariableType> u_coarse, View<VariableType> psi_fine, View<VariableType> u_fine )
{
    static_assert ( static_false<VariableType>::value, "refine<VariableType, NumGhosts, Dimension> not implemented" );
};

template<>
void refine<CellCentered, 1, 2> ( int i_fine, int j_fine, int k_fine, Level const * level_fine, Level const * level_coarse, ConstView<CellCentered> psi_coarse, ConstView<CellCentered> u_coarse, View<CellCentered> psi_fine, View<CellCentered> u_fine )
{
    IntVector cell_fine ( i_fine, j_fine, k_fine );
    IntVector cell_coarse ( level_fine->mapCellToCoarser ( cell_fine ) );

    psi_fine[cell_fine] = psi_coarse[cell_coarse];
    u_fine[cell_fine] = u_coarse[cell_coarse];
};

template<>
void refine<CellCentered, 1, 3> ( int i_fine, int j_fine, int k_fine, Level const * level_fine, Level const * level_coarse, ConstView<CellCentered> psi_coarse, ConstView<CellCentered> u_coarse, View<CellCentered> psi_fine, View<CellCentered> u_fine )
{
    IntVector cell_fine ( i_fine, j_fine, k_fine );
    IntVector cell_coarse ( level_fine->mapCellToCoarser ( cell_fine ) );

    psi_fine[cell_fine] = psi_coarse[cell_coarse];
    u_fine[cell_fine] = u_coarse[cell_coarse];
};

template<>
void refine<NodeCentered, 1, 2> ( int i_fine, int j_fine, int k_fine, Level const * level_fine, Level const * level_coarse, ConstView<NodeCentered> psi_coarse, ConstView<NodeCentered> u_coarse, View<NodeCentered> psi_fine, View<NodeCentered> u_fine )
{
    IntVector node_fine ( i_fine, j_fine, k_fine );
    IntVector node_coarse ( level_fine->mapNodeToCoarser ( node_fine ) );
    Point point_fine ( level_fine->getNodePosition ( node_fine ) );
    Point point_coarse ( level_coarse->getNodePosition ( node_coarse ) );
    Vector dist = ( point_fine.asVector() - point_coarse.asVector() ) / level_coarse->dCell();
    double w00 ( 1 ), w10 ( 1 ), w01 ( 1 ), w11 ( 1 );
    IntVector n00 ( node_coarse ), n01 ( node_coarse ), n10 ( node_coarse ), n11 ( node_coarse );
    if ( dist.x() < 0. ) {
        n00[0] = n01[0] -= 1;
        w00 *= -dist.x();
        w01 *= -dist.x();
        w10 *= ( 1 + dist.x() );
        w11 *= ( 1 + dist.x() );
    } else if ( dist.x() > 0. ) {
        n10[0] = n11[0] += 1;
        w00 *= ( 1 - dist.x() );
        w01 *= ( 1 - dist.x() );
        w10 *= dist.x();
        w11 *= dist.x();
    } else {
        w10 *= 0.;
        w11 *= 0.;
    }

    if ( dist.y() < 0. ) {
        n00[1] = n10[1] -= 1;
        w00 *= -dist.y();
        w10 *= -dist.y();
        w01 *= ( 1 + dist.y() );
        w11 *= ( 1 + dist.y() );
    } else if ( dist.y() > 0. ) {
        n01[1] = n11[1] += 1;
        w00 *= ( 1 - dist.y() );
        w10 *= ( 1 - dist.y() );
        w01 *= dist.y();
        w11 *= dist.y();
    } else {
        w01 *= 0.;
        w11 *= 0.;
    }

    psi_fine[node_fine] = w00 * psi_coarse[n00] +
                          w01 * psi_coarse[n01] +
                          w10 * psi_coarse[n10] +
                          w11 * psi_coarse[n11];
    u_fine[node_fine] = w00 * u_coarse[n00] +
                        w01 * u_coarse[n01] +
                        w10 * u_coarse[n10] +
                        w11 * u_coarse[n11];
};

template<>
void refine<NodeCentered, 1, 3> ( int i_fine, int j_fine, int k_fine, Level const * level_fine, Level const * level_coarse, ConstView<NodeCentered> psi_coarse, ConstView<NodeCentered> u_coarse, View<NodeCentered> psi_fine, View<NodeCentered> u_fine )
{
    IntVector node_fine ( i_fine, j_fine, k_fine );
    IntVector node_coarse ( level_fine->mapNodeToCoarser ( node_fine ) );
    Point point_fine ( level_fine->getNodePosition ( node_fine ) );
    Point point_coarse ( level_coarse->getNodePosition ( node_coarse ) );
    Vector dist = ( point_fine.asVector() - point_coarse.asVector() ) / level_coarse->dCell();
    double w000 ( 1 ), w100 ( 1 ), w010 ( 1 ), w110 ( 1 ), w001 ( 1 ), w101 ( 1 ), w011 ( 1 ), w111 ( 1 );
    IntVector n000 ( node_coarse ), n010 ( node_coarse ), n100 ( node_coarse ), n110 ( node_coarse ), n001 ( node_coarse ), n011 ( node_coarse ), n101 ( node_coarse ), n111 ( node_coarse );
    if ( dist.x() < 0. ) {
        n000[0] = n001[0] = n010[0] = n011[0] -= 1;
        w000 *= -dist.x();
        w001 *= -dist.x();
        w010 *= -dist.x();
        w011 *= -dist.x();
        w100 *= ( 1 + dist.x() );
        w101 *= ( 1 + dist.x() );
        w110 *= ( 1 + dist.x() );
        w111 *= ( 1 + dist.x() );
    } else if ( dist.x() > 0. ) {
        n100[0] = n101[0] = n110[0] = n111[0] += 1;
        w000 *= ( 1 - dist.x() );
        w001 *= ( 1 - dist.x() );
        w010 *= ( 1 - dist.x() );
        w011 *= ( 1 - dist.x() );
        w100 *= dist.x();
        w101 *= dist.x();
        w110 *= dist.x();
        w111 *= dist.x();
    } else {
        w100 = 0.;
        w101 = 0.;
        w110 = 0.;
        w111 = 0.;
    }

    if ( dist.y() < 0. ) {
        n000[1] = n100[1] = n001[1] = n101[1] -= 1;
        w000 *= -dist.y();
        w001 *= -dist.y();
        w100 *= -dist.y();
        w101 *= -dist.y();
        w010 *= ( 1 + dist.y() );
        w011 *= ( 1 + dist.y() );
        w110 *= ( 1 + dist.y() );
        w111 *= ( 1 + dist.y() );
    } else if ( dist.y() > 0. ) {
        n010[1] = n110[1] = n011[1] = n111[1] += 1;
        w000 *= ( 1 - dist.y() );
        w001 *= ( 1 - dist.y() );
        w100 *= ( 1 - dist.y() );
        w101 *= ( 1 - dist.y() );
        w010 *= dist.y();
        w011 *= dist.y();
        w110 *= dist.y();
        w111 *= dist.y();
    } else {
        w010 = 0.;
        w011 = 0.;
        w110 = 0.;
        w111 = 0.;
    }

    if ( dist.z() < 0. ) {
        n000[2] = n010[2] = n100[2] = n110[2] -= 1;
        w000 *= -dist.z();
        w010 *= -dist.z();
        w100 *= -dist.z();
        w110 *= -dist.z();
        w001 *= ( 1 + dist.z() );
        w011 *= ( 1 + dist.z() );
        w101 *= ( 1 + dist.z() );
        w111 *= ( 1 + dist.z() );
    } else if ( dist.z() > 0. ) {
        n001[2] = n011[2] = n101[2] = n111[2] += 1;
        w000 *= ( 1 - dist.z() );
        w010 *= ( 1 - dist.z() );
        w100 *= ( 1 - dist.z() );
        w110 *= ( 1 - dist.z() );
        w001 *= dist.z();
        w011 *= dist.z();
        w101 *= dist.z();
        w111 *= dist.z();
    } else {
        w001 = 0.;
        w011 = 0.;
        w101 = 0.;
        w111 = 0.;
    }
    psi_fine[node_fine] = w000 * psi_coarse[n000] +
                          w001 * psi_coarse[n001] +
                          w010 * psi_coarse[n010] +
                          w011 * psi_coarse[n011] +
                          w100 * psi_coarse[n100] +
                          w101 * psi_coarse[n101] +
                          w110 * psi_coarse[n110] +
                          w111 * psi_coarse[n111];

    u_fine[node_fine] = w000 * u_coarse[n000] +
                        w001 * u_coarse[n001] +
                        w010 * u_coarse[n010] +
                        w011 * u_coarse[n011] +
                        w100 * u_coarse[n100] +
                        w101 * u_coarse[n101] +
                        w110 * u_coarse[n110] +
                        w111 * u_coarse[n111];
};

template<VariableType VariableType, int NumGhosts, int Dimension>
void coarsen ( int i_fine, int j_fine, int k_fine, Level const * level_fine, Level const * level_coarse, ConstView<VariableType> psi_coarse, ConstView<VariableType> u_coarse, View<VariableType> psi_fine, View<VariableType> u_fine )
{
    static_assert ( static_false<VariableType>::value, "coarsen<VariableType, NumGhosts, Dimension> not implemented" );
};

template<>
void coarsen<CellCentered, 1, 2> ( int i_coarse, int j_coarse, int k_coarse, Level const * level_coarse, Level const * level_fine, ConstView<CellCentered> psi_fine, ConstView<CellCentered> u_fine, View<CellCentered> psi_coarse, View<CellCentered> u_coarse )
{
    IntVector l_coarse ( i_coarse, j_coarse, k_coarse ); // bottom-left-lower corner
    IntVector h_coarse = l_coarse + IntVector ( 1, 1, 1 );
    IntVector l_fine ( level_coarse->mapNodeToFiner ( l_coarse ) );
    IntVector h_fine = l_fine + level_fine->getRefinementRatio(); // TODO check this

    double sum_psi ( 0. ), sum_u ( 0. ), cnt ( 0. );
    for ( CellIterator it ( l_fine, h_fine ); !it.done(); ++it ) {
        sum_psi += psi_fine[*it];
        sum_u += u_fine[*it];
        cnt += 1.;
    }
    psi_coarse ( i_coarse, j_coarse, k_coarse ) = sum_psi / cnt;
    u_coarse ( i_coarse, j_coarse, k_coarse ) = sum_u / cnt;
};

template<>
void coarsen<CellCentered, 1, 3> ( int i_coarse, int j_coarse, int k_coarse, Level const * level_coarse, Level const * level_fine, ConstView<CellCentered> psi_fine, ConstView<CellCentered> u_fine, View<CellCentered> psi_coarse, View<CellCentered> u_coarse )
{
    IntVector l_coarse ( i_coarse, j_coarse, k_coarse ); // bottom-left-lower corner
    IntVector h_coarse = l_coarse + IntVector ( 1, 1, 1 );
    IntVector l_fine ( level_coarse->mapNodeToFiner ( l_coarse ) );
    IntVector h_fine = l_fine + level_fine->getRefinementRatio(); // TODO check this

    double sum_psi ( 0. ), sum_u ( 0. ), cnt ( 0. );
    for ( CellIterator it ( l_fine, h_fine ); !it.done(); ++it ) {
        sum_psi += psi_fine[*it];
        sum_u += u_fine[*it];
        cnt += 1.;
    }
    psi_coarse ( i_coarse, j_coarse, k_coarse ) = sum_psi / cnt;
    u_coarse ( i_coarse, j_coarse, k_coarse ) = sum_u / cnt;
};

template<>
void coarsen<NodeCentered, 1, 2> ( int i_coarse, int j_coarse, int k_coarse, Level const * level_coarse, Level const * level_fine, ConstView<NodeCentered> psi_fine, ConstView<NodeCentered> u_fine, View<NodeCentered> psi_coarse, View<NodeCentered> u_coarse )
{
    IntVector node_coarse ( i_coarse, j_coarse, k_coarse );
    IntVector node_fine ( level_coarse->mapNodeToFiner ( node_coarse ) );

    Point point_coarse ( level_coarse->getNodePosition ( node_coarse ) );
    Point point_fine ( level_fine->getNodePosition ( node_fine ) );

    assert ( ( point_fine.asVector() - point_coarse.asVector() ).length() == 0 );

    psi_coarse[node_coarse] = psi_fine[node_fine];
    u_coarse[node_coarse] = u_fine[node_fine];
};

template<>
void coarsen<NodeCentered, 1, 3> ( int i_coarse, int j_coarse, int k_coarse, Level const * level_coarse, Level const * level_fine, ConstView<NodeCentered> psi_fine, ConstView<NodeCentered> u_fine, View<NodeCentered> psi_coarse, View<NodeCentered> u_coarse )
{
    IntVector node_coarse ( i_coarse, j_coarse, k_coarse );
    IntVector node_fine ( level_coarse->mapNodeToFiner ( node_coarse ) );

    Point point_coarse ( level_coarse->getNodePosition ( node_coarse ) );
    Point point_fine ( level_fine->getNodePosition ( node_fine ) );

    assert ( ( point_fine.asVector() - point_coarse.asVector() ).length() == 0 );

    psi_coarse[node_coarse] = psi_fine[node_fine];
    u_coarse[node_coarse] = u_fine[node_fine];
};

template<VariableType VariableType, int NumGhosts, int Dimension>
void error_estimate ( int i_fine, int j_fine, int k_fine, Patch const * patch, ConstView<VariableType> psi, FlagView flag_refine, bool & refine_patch, double const & refine_threshold )
{
    static_assert ( static_false<VariableType>::value, "error_estimate<VariableType, NumGhosts, Dimension> not implemented" );
};

template<>
void error_estimate<CellCentered, 1, 2> ( int i, int j, int k, Patch const * patch, ConstView<CellCentered> psi, FlagView flag_refine, bool & refine_patch, double const & refine_threshold )
{
    IntVector l ( psi.getLowIndex() ), h ( psi.getHighIndex() );
    Vector d ( patch->dCell() );

    int im = ( i > l.x() ) ? i - 1 : i, ip = ( i < h.x() - 1 ) ? i + 1 : i;
    int jm = ( j > l.y() ) ? j - 1 : j, jp = ( j < h.y() - 1 ) ? j + 1 : j;
    double psi_x = ( psi ( ip, j, k ) - psi ( im, j, k ) ) / ( d.x() * ( ip - im ) );
    double psi_y = ( psi ( i, jp, k ) - psi ( i, jm, k ) ) / ( d.y() * ( jp - jm ) );
    double grad_psi_norm2 = psi_x * psi_x + psi_y * psi_y;

    bool tmp = grad_psi_norm2 > refine_threshold;
    flag_refine ( i, j, k ) = tmp;
    refine_patch |= tmp;
}

template<>
void error_estimate<CellCentered, 1, 3> ( int i, int j, int k, Patch const * patch, ConstView<CellCentered> psi, FlagView flag_refine, bool & refine_patch, double const & refine_threshold )
{
    IntVector l ( psi.getLowIndex() ), h ( psi.getHighIndex() );
    Vector d ( patch->dCell() );

    int im = ( i > l.x() ) ? i - 1 : i, ip = ( i < h.x() - 1 ) ? i + 1 : i;
    int jm = ( j > l.y() ) ? j - 1 : j, jp = ( j < h.y() - 1 ) ? j + 1 : j;
    int km = ( k > l.z() ) ? k - 1 : k, kp = ( k < h.z() - 1 ) ? k + 1 : k;
    double psi_x = ( psi ( ip, j, k ) - psi ( im, j, k ) ) / ( d.x() * ( ip - im ) );
    double psi_y = ( psi ( i, jp, k ) - psi ( i, jm, k ) ) / ( d.y() * ( jp - jm ) );
    double psi_z = ( psi ( i, j, kp ) - psi ( i, j, km ) ) / ( d.z() * ( kp - km ) );
    double grad_psi_norm2 = psi_x * psi_x + psi_y * psi_y + psi_z * psi_z;

    bool tmp = grad_psi_norm2 > refine_threshold;
    flag_refine ( i, j, k ) = tmp;
    refine_patch |= tmp;
}

template<>
void error_estimate<NodeCentered, 1, 2> ( int i, int j, int k, Patch const * patch, ConstView<NodeCentered> psi, FlagView flag_refine, bool & refine_patch, double const & refine_threshold )
{
    Vector d ( patch->dCell() );

    double psi_x_0 = ( psi ( i + 1, j,     k ) - psi ( i,     j,     k ) ) / d.x();
    double psi_y_0 = ( psi ( i,     j + 1, k ) - psi ( i,     j,     k ) ) / d.x();
    double grad_psi_00 = psi_x_0 * psi_x_0 + psi_y_0 * psi_y_0;
    double psi_x_1 = ( psi ( i + 1, j + 1, k ) - psi ( i,     j + 1, k ) ) / d.x();
    double psi_y_1 = ( psi ( i + 1, j + 1, k ) - psi ( i + 1, j,     k + 1 ) ) / d.x();
    double grad_psi_11 = psi_x_1 * psi_x_1 + psi_y_1 * psi_y_1;

    bool tmp = grad_psi_00 > refine_threshold ||
               grad_psi_11 > refine_threshold;
    flag_refine ( i, j, k ) = tmp;
    refine_patch |= tmp;
}

template<>
void error_estimate<NodeCentered, 1, 3> ( int i, int j, int k, Patch const * patch, ConstView<NodeCentered> psi, FlagView flag_refine, bool & refine_patch, double const & refine_threshold )
{
    Vector d ( patch->dCell() );

    double psi_x_00 = ( psi ( i + 1, j,   k ) - psi ( i,   j,   k ) ) / d.x();
    double psi_y_00 = ( psi ( i,   j + 1, k ) - psi ( i,   j,   k ) ) / d.x();
    double psi_z_00 = ( psi ( i,   j,   k + 1 ) - psi ( i,   j,   k ) ) / d.x();
    double grad_psi_000 = psi_x_00 * psi_x_00 + psi_y_00 * psi_y_00 + psi_z_00 * psi_z_00;
    double psi_x_10 = ( psi ( i + 1, j + 1, k ) - psi ( i,   j + 1, k ) ) / d.x();
    double psi_y_10 = ( psi ( i + 1, j + 1, k ) - psi ( i + 1, j,   k ) ) / d.x();
    double psi_z_11 = ( psi ( i + 1, j + 1, k + 1 ) - psi ( i + 1, j + 1, k ) ) / d.x();
    double grad_psi_110 = psi_x_10 * psi_x_10 + psi_y_10 * psi_y_10 + psi_z_11 * psi_z_11;
    double psi_x_01 = ( psi ( i + 1, j,   k + 1 ) - psi ( i,   j, k + 1 ) ) / d.x();
    double psi_y_11 = ( psi ( i + 1, j + 1, k + 1 ) - psi ( i + 1, j, k + 1 ) ) / d.x();
    double psi_z_10 = ( psi ( i + 1, j,   k + 1 ) - psi ( i + 1, j, k ) ) / d.x();
    double grad_psi_101 = psi_x_01 * psi_x_01 + psi_y_11 * psi_y_11 + psi_z_10 * psi_z_10;
    double psi_x_11 = ( psi ( i + 1, j + 1, k + 1 ) - psi ( i, j + 1, k + 1 ) ) / d.x();
    double psi_y_01 = ( psi ( i,   j + 1, k + 1 ) - psi ( i, j  , k + 1 ) ) / d.x();
    double psi_z_01 = ( psi ( i,   j + 1, k + 1 ) - psi ( i, j + 1, k ) ) / d.x();
    double grad_psi_011 = psi_x_11 * psi_x_11 + psi_y_01 * psi_y_01 + psi_z_01 * psi_z_01;

    bool tmp = grad_psi_000 > refine_threshold ||
               grad_psi_110 > refine_threshold ||
               grad_psi_101 > refine_threshold ||
               grad_psi_011 > refine_threshold;
    flag_refine ( i, j, k ) = tmp;
    refine_patch |= tmp;
}

template<int Dimension>
void error_estimate_test ( int i, int j, int k, Patch const * patch, FlagView flag_refine, bool & refine_patch )
{
    bool tmp = true;
    IntVector l, h;
    patch->getLevel()->findCellIndexRange ( l, h );
    if ( Dimension == 2 )
        tmp = ( h.x() + 3 * l.x() < 4 * i && 4 * i < l.x() + 3 * h.x() ) &&
              ( h.y() + 3 * l.y() < 4 * j && 4 * j < l.y() + 3 * h.y() );
    if ( Dimension == 3 )
        tmp = ( h.x() + 3 * l.x() < 4 * i && 4 * i < l.x() + 3 * h.x() ) &&
              ( h.y() + 3 * l.y() < 4 * j && 4 * j < l.y() + 3 * h.y() ) &&
              ( h.z() + 3 * l.z() < 4 * k && 4 * k < l.z() + 3 * h.z() );
    flag_refine ( i, j, k ) = tmp;
    refine_patch |= tmp;
}

}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
class AMRPhaseField : public PhaseField<VariableType, NumGhosts, Dimension>
{
protected:
    using Variable = PF::Variable<VariableType>;
    using ConstVariable = PF::ConstVariable<VariableType>;
    using FlagVariable = PF::FlagVariable;
    using FlagView = PF::FlagView;

    using PhaseField<VariableType, NumGhosts, Dimension>::get_view;

    using PhaseField<VariableType, NumGhosts, Dimension>::get_low;
    using PhaseField<VariableType, NumGhosts, Dimension>::get_high;

    template <typename... Args>
    static BlockRange get_flag_inner_range ( Args && ... args )
    {
        return PF::get_inner_range<PF::CellCentered, NumGhosts, Dimension> ( std::forward<Args> ( args )... );
    }

    template <typename... Args>
    static IntVector map_to_coarser ( Args && ... args )
    {
        return PF::map_to_coarser<VariableType> ( std::forward<Args> ( args )... );
    }

    template <typename... Args>
    static IntVector map_to_finer ( Args && ... args )
    {
        return PF::map_to_finer<VariableType> ( std::forward<Args> ( args )... );
    }

    template <typename... Args>
    static void refine ( Args && ... args )
    {
        return PF::refine<VariableType, NumGhosts, Dimension> ( std::forward<Args> ( args )... );
    }

    template <typename... Args>
    static void coarsen ( Args && ... args )
    {
        return PF::coarsen<VariableType, NumGhosts, Dimension> ( std::forward<Args> ( args )... );
    }

    template <typename... Args>
    void error_estimate ( Args && ... args )
    {
        return PF::error_estimate<VariableType, NumGhosts, Dimension> ( std::forward<Args> ( args )... );
    }

    using PhaseField<VariableType, NumGhosts, Dimension>::psi_label;
    using PhaseField<VariableType, NumGhosts, Dimension>::u_label;
    using PhaseField<VariableType, NumGhosts, Dimension>::state;
    using PhaseField<VariableType, NumGhosts, Dimension>::dbg_out1;
    using PhaseField<VariableType, NumGhosts, Dimension>::dbg_out2;
    using PhaseField<VariableType, NumGhosts, Dimension>::dbg_out3;
    using PhaseField<VariableType, NumGhosts, Dimension>::dbg_out4;
    using PhaseField<VariableType, NumGhosts, Dimension>::GhostType;

#ifdef UINTAH_ENABLE_KOKKOS
    FlagView get_view ( FlagVariable & var )
    {
        return var.getKokkosView();
    }
#else
    FlagView get_view ( FlagVariable & var )
    {
        return var;
    }
#endif

protected:
    double refine_threshold;

public:
    AMRPhaseField ( ProcessorGroup const * myworld, int verbosity = 0 );
    virtual ~AMRPhaseField ();

protected:
    AMRPhaseField ( AMRPhaseField const & ) = delete;
    AMRPhaseField & operator= ( AMRPhaseField const & ) = delete;

public:
    virtual void problemSetup ( ProblemSpecP const & params, ProblemSpecP const & restart_prob_spec, GridP & grid, SimulationStateP & state ) override;
    virtual void scheduleRefine ( PatchSet const * patches, SchedulerP & sched ) override;
    virtual void scheduleRefineInterface ( LevelP const & level_fine, SchedulerP & sched, bool need_old_coarse, bool need_new_coarse )
    {
        /*TODO*/
    };
    virtual void scheduleCoarsen ( LevelP const & level_coarse, SchedulerP & sched ) override;
    virtual void scheduleErrorEstimate ( LevelP const & level_coarse, SchedulerP & sched ) override;
    virtual void scheduleInitialErrorEstimate ( LevelP const & level_coarse, SchedulerP & sched ) override;

protected:
    void task_refine ( ProcessorGroup const * myworld, PatchSubset const * patches_fine, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new );
    void task_coarsen ( ProcessorGroup const * myworld, PatchSubset const * patches_coarse, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new );
    virtual void task_error_estimate ( ProcessorGroup const * myworld, PatchSubset const * patches, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new );
};

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
class AMRPhaseFieldTest : public AMRPhaseField<VariableType, NumGhosts, Dimension>
{
    using Variable = PF::Variable<VariableType>;
    using ConstVariable = PF::ConstVariable<VariableType>;
    using FlagVariable = PF::FlagVariable;
    using FlagView = PF::FlagView;
    using PhaseField<VariableType, NumGhosts, Dimension>::psi_label;
    using PhaseField<VariableType, NumGhosts, Dimension>::u_label;
    using PhaseField<VariableType, NumGhosts, Dimension>::state;
    using PhaseField<VariableType, NumGhosts, Dimension>::dbg_out1;
    using PhaseField<VariableType, NumGhosts, Dimension>::dbg_out2;
    using PhaseField<VariableType, NumGhosts, Dimension>::dbg_out3;
    using PhaseField<VariableType, NumGhosts, Dimension>::dbg_out4;
    using PhaseField<VariableType, NumGhosts, Dimension>::get_view;
    using AMRPhaseField<VariableType, NumGhosts, Dimension>::get_view;
    using AMRPhaseField<VariableType, NumGhosts, Dimension>::get_flag_inner_range;
    using AMRPhaseField<VariableType, NumGhosts, Dimension>::AMRPhaseField;
    template <typename... Args>
    void error_estimate_test ( Args && ... args )
    {
        return PF::error_estimate_test<Dimension> ( std::forward<Args> ( args )... );
    }
    virtual void task_error_estimate ( ProcessorGroup const * myworld, PatchSubset const * patches, MaterialSubset const * matls, DataWarehouse * dw_old, DataWarehouse * dw_new ) override;
};

extern template class AMRPhaseField <PF::CellCentered, 1, 2>;
extern template class AMRPhaseField <PF::NodeCentered, 1, 2>;
extern template class AMRPhaseField <PF::CellCentered, 1, 3>;
extern template class AMRPhaseField <PF::NodeCentered, 1, 3>;
extern template class AMRPhaseFieldTest <PF::CellCentered, 1, 2>;
extern template class AMRPhaseFieldTest <PF::NodeCentered, 1, 2>;
extern template class AMRPhaseFieldTest <PF::CellCentered, 1, 3>;
extern template class AMRPhaseFieldTest <PF::NodeCentered, 1, 3>;

using AMRCCPhaseField2D = AMRPhaseField <PF::CellCentered, 1, 2>;
using AMRNCPhaseField2D = AMRPhaseField <PF::NodeCentered, 1, 2>;
using AMRCCPhaseField3D = AMRPhaseField <PF::CellCentered, 1, 3>;
using AMRNCPhaseField3D = AMRPhaseField <PF::NodeCentered, 1, 3>;

using AMRCCPhaseField2DTest = AMRPhaseFieldTest <PF::CellCentered, 1, 2>;
using AMRNCPhaseField2DTest = AMRPhaseFieldTest <PF::NodeCentered, 1, 2>;
using AMRCCPhaseField3DTest = AMRPhaseFieldTest <PF::CellCentered, 1, 3>;
using AMRNCPhaseField3DTest = AMRPhaseFieldTest <PF::NodeCentered, 1, 3>;

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
AMRPhaseField<VariableType, NumGhosts, Dimension>::AMRPhaseField ( ProcessorGroup const * myworld, int verbosity )
    : PhaseField<VariableType, NumGhosts, Dimension> ( myworld, verbosity )
{}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
AMRPhaseField<VariableType, NumGhosts, Dimension>::~AMRPhaseField ()
{}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void AMRPhaseField<VariableType, NumGhosts, Dimension>::problemSetup ( ProblemSpecP const & params, ProblemSpecP const & restart_prob_spec, GridP & grid, SimulationStateP & state )
{
    PhaseField<VariableType, NumGhosts, Dimension>::problemSetup ( params, restart_prob_spec, grid, state );

    ProblemSpecP diffusion = params->findBlock ( "PhaseField" );
    diffusion->require ( "refine_threshold", refine_threshold );
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void AMRPhaseField<VariableType, NumGhosts, Dimension>::scheduleRefine ( PatchSet const * patches, SchedulerP & sched )
{
    if ( getLevel ( patches )->getIndex() == 0 ) {
        return;
    }

    Task * task = scinew Task ( "AMRPhaseField::task_refine", this, &AMRPhaseField::task_refine );
    if ( VariableType == PF::CellCentered ) {
        task->requires ( Task::NewDW, psi_label, 0, Task::CoarseLevel, 0, Task::NormalDomain, Ghost::None, 0 );
        task->requires ( Task::NewDW, u_label, 0, Task::CoarseLevel, 0, Task::NormalDomain, Ghost::None, 0 );
    } else {
        task->requires ( Task::NewDW, psi_label, 0, Task::CoarseLevel, 0, Task::NormalDomain, Ghost::AroundNodes, 1 );
        task->requires ( Task::NewDW, u_label, 0, Task::CoarseLevel, 0, Task::NormalDomain, Ghost::AroundNodes, 1 );
    }
    task->computes ( psi_label );
    task->computes ( u_label );
    sched->addTask ( task, patches, state->allMaterials() );
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void AMRPhaseField<VariableType, NumGhosts, Dimension>::scheduleCoarsen ( LevelP const & level_coarse, SchedulerP & sched )
{
    Task * task = scinew Task ( "AMRPhaseField::task_coarsen", this, &AMRPhaseField::task_coarsen );
    task->requires ( Task::NewDW, psi_label, 0, Task::FineLevel, 0, Task::NormalDomain, Ghost::None, 0 );
    task->requires ( Task::NewDW, u_label, 0, Task::FineLevel, 0, Task::NormalDomain, Ghost::None, 0 );
    task->modifies ( psi_label );
    task->modifies ( u_label );
    sched->addTask ( task, level_coarse->eachPatch(), state->allMaterials() );
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void AMRPhaseField<VariableType, NumGhosts, Dimension>::scheduleErrorEstimate ( LevelP const & level_coarse, SchedulerP & sched )
{
    Task * task = scinew Task ( "AMRPhaseField::task_error_estimate", this, &AMRPhaseField::task_error_estimate );
    task->requires ( Task::NewDW, psi_label, Ghost::Ghost::None, 0 ); // this is actually the old value of this
    task->modifies ( state->get_refineFlag_label(), state->refineFlagMaterials() );
    task->modifies ( state->get_refinePatchFlag_label(), state->refineFlagMaterials() );
    sched->addTask ( task, level_coarse->eachPatch(), state->allMaterials() );
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void AMRPhaseField<VariableType, NumGhosts, Dimension>::scheduleInitialErrorEstimate ( LevelP const & level_coarse, SchedulerP & sched )
{
    scheduleErrorEstimate ( level_coarse, sched );
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void AMRPhaseField<VariableType, NumGhosts, Dimension>::task_refine ( ProcessorGroup const * /*myworld*/, PatchSubset const * patches_fine, MaterialSubset const * /*matls*/, DataWarehouse * /*dw_old*/, DataWarehouse * dw_new )
{
    dbg_out1 << "==== AMRPhaseField::task_refine ====" << std::endl;

    const Level * level_fine = getLevel ( patches_fine );
    const Level * level_coarse = level_fine->getCoarserLevel().get_rep();

    for ( int p = 0; p < patches_fine->size(); ++p ) {
        const Patch * patch_fine = patches_fine->get ( p );
        dbg_out2 << "== Fine Patch: " << *patch_fine << std::endl;

        Variable psi_fine, u_fine;
        dw_new->allocateAndPut ( psi_fine, psi_label, 0, patch_fine );
        dw_new->allocateAndPut ( u_fine, u_label, 0, patch_fine );
        dbg_out4 << "psi_fine \t window " << psi_fine.getLowIndex() << psi_fine.getHighIndex() << std::endl;
        dbg_out4 << "u_fine \t window " << u_fine.getLowIndex() << u_fine.getHighIndex() << std::endl;

        IntVector l_fine = get_low ( patch_fine );
        IntVector h_fine = get_high ( patch_fine );

        IntVector l_coarse = map_to_coarser ( level_fine, l_fine );
        IntVector h_coarse = map_to_coarser ( level_fine, h_fine );
        if ( VariableType == PF::NodeCentered ) { // Extening in order to select nodes on right edges
            h_coarse += IntVector ( 1, 1, Dimension == 3 ? 1 : 0 );
        }

        dbg_out4 << "fine range" << BlockRange ( l_fine, h_fine ) << std::endl;
        dbg_out4 << "coarse range" << BlockRange ( l_coarse, h_coarse ) << std::endl;

        ConstVariable psi_coarse, u_coarse;
        dw_new->getRegion ( psi_coarse, psi_label, 0, level_coarse, l_coarse, h_coarse );
        dw_new->getRegion ( u_coarse, u_label, 0, level_coarse, l_coarse, h_coarse );
        dbg_out4 << "psi_coarse \t window " << psi_coarse.getLowIndex() << psi_coarse.getHighIndex() << std::endl;
        dbg_out4 << "u_coarse \t window " << u_coarse.getLowIndex() << u_coarse.getHighIndex() << std::endl;

        BlockRange range_fine ( l_fine, h_fine );
        dbg_out3 << "= Iterating over fine range" << range_fine << std::endl;
        parallel_for ( range_fine, [level_fine, level_coarse, &psi_coarse, &u_coarse, &psi_fine, &u_fine, this] ( int i, int j, int k )->void { refine ( i, j, k, level_fine, level_coarse, get_view ( psi_coarse ), get_view ( u_coarse ), get_view ( psi_fine ), get_view ( u_fine ) ); } );
    }

    dbg_out2 << std::endl;
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void
AMRPhaseField<VariableType, NumGhosts, Dimension>::task_coarsen ( ProcessorGroup const * /*myworld*/, const PatchSubset * patches_coarse, const MaterialSubset * /*matls*/, DataWarehouse * dw_old, DataWarehouse * dw_new )
{
    dbg_out1 << "==== AMRPhaseField::task_coarsen " << std::endl;

    const Level * level_coarse = getLevel ( patches_coarse );
    const Level * level_fine = level_coarse->getFinerLevel().get_rep();

    for ( int p = 0; p < patches_coarse->size(); ++p ) {
        const Patch * patch_coarse = patches_coarse->get ( p );
        dbg_out2 << "== Coarse Patch: " << *patch_coarse << std::endl;

        Variable psi_coarse, u_coarse;
        dw_new->getModifiable ( psi_coarse, psi_label, 0, patch_coarse );
        dw_new->getModifiable ( u_coarse, u_label, 0, patch_coarse );
        dbg_out4 << "psi_coarse \t window " << psi_coarse.getLowIndex() << psi_coarse.getHighIndex() << std::endl;
        dbg_out4 << "u_coarse \t window " << u_coarse.getLowIndex() << u_coarse.getHighIndex() << std::endl;

        IntVector l_coarse = get_low ( patch_coarse );
        IntVector h_coarse = get_high ( patch_coarse );

        IntVector l_fine = map_to_finer ( level_coarse, l_coarse );
        IntVector h_fine = map_to_finer ( level_coarse, h_coarse );

        Level::selectType patches_fine;
        level_fine->selectPatches ( l_fine, h_fine, patches_fine );

        for ( int i = 0; i < patches_fine.size(); ++i ) {
            const Patch * patch_fine = patches_fine[i];
            dbg_out3 << "= Fine Patch " << *patch_fine << std::endl;

            ConstVariable psi_fine, u_fine;
            dw_new->get ( psi_fine, psi_label, 0, patch_fine, Ghost::None, 0 );
            dw_new->get ( u_fine, u_label, 0, patch_fine, Ghost::None, 0 );
            dbg_out4 << "psi_fine \t window " << psi_fine.getLowIndex() << psi_fine.getHighIndex() << std::endl;
            dbg_out4 << "u_fine \t window " << u_fine.getLowIndex() << u_fine.getHighIndex() << std::endl;

            BlockRange range_coarse (
                Max ( l_coarse, map_to_coarser ( level_fine, get_low ( patch_fine ) ) ),
                Min ( h_coarse, map_to_coarser ( level_fine, get_high ( patch_fine ) ) )
            );

            dbg_out3 << "= Iterating over coarse cells window " << range_coarse << std::endl;
            parallel_for ( range_coarse, [level_coarse, level_fine, &psi_fine, &u_fine, &psi_coarse, &u_coarse, this] ( int i, int j, int k )->void { coarsen ( i, j, k, level_coarse, level_fine, get_view ( psi_fine ), get_view ( u_fine ), get_view ( psi_coarse ), get_view ( u_coarse ) ); } );
        }
    }

    dbg_out2 << std::endl;
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void AMRPhaseField<VariableType, NumGhosts, Dimension>::task_error_estimate ( ProcessorGroup const * /*myworld*/, PatchSubset const * patches, MaterialSubset const * /*matls*/, DataWarehouse * /*dw_old*/, DataWarehouse * dw_new )
{
    dbg_out1 << "==== AMRPhaseField::task_error_estimate " << std::endl;

    for ( int p = 0; p < patches->size(); ++p ) {
        const Patch * patch = patches->get ( p );
        dbg_out2 << "== Patch: " << *patch << std::endl;

        FlagVariable flag_refine;
        PerPatch<PatchFlagP> flag_refine_patch;
        dw_new->getModifiable ( flag_refine, state->get_refineFlag_label(), 0, patch );
        dw_new->get ( flag_refine_patch, state->get_refinePatchFlag_label(), 0, patch );
        dbg_out4 << "flag_refine \t window " << flag_refine.getLowIndex() << flag_refine.getHighIndex() << std::endl;

        PatchFlag * patch_flag_refine = flag_refine_patch.get().get_rep();
        ConstVariable psi;
        dw_new->get ( psi, psi_label, 0, patch, Ghost::None, 0 );
        dbg_out4 << "psi \t window " << psi.getLowIndex() << psi.getHighIndex() << std::endl;

        bool refine_patch = false;
        BlockRange range = get_flag_inner_range ( patch );
        dbg_out3 << "= Iterating over inner cells window " << range << std::endl;
        parallel_for ( range, [patch, &psi, &flag_refine, &refine_patch, this] ( int i, int j, int k )->void { error_estimate ( i, j, k, patch, get_view ( psi ), get_view ( flag_refine ), refine_patch, refine_threshold ); } );

        if ( refine_patch ) {
            patch_flag_refine->set();
        }
    }

    dbg_out2 << std::endl;
}

template<PF::VariableType VariableType, int NumGhosts, int Dimension>
void AMRPhaseFieldTest<VariableType, NumGhosts, Dimension>::task_error_estimate ( ProcessorGroup const * /*myworld*/, PatchSubset const * patches, MaterialSubset const * /*matls*/, DataWarehouse * /*dw_old*/, DataWarehouse * dw_new )
{
    dbg_out1 << "==== AMRPhaseFieldTest::task_error_estimate " << std::endl;

    for ( int p = 0; p < patches->size(); ++p ) {
        const Patch * patch = patches->get ( p );
        dbg_out2 << "== Patch: " << *patch << std::endl;

        FlagVariable flag_refine;
        PerPatch<PatchFlagP> flag_refine_patch;
        dw_new->getModifiable ( flag_refine, state->get_refineFlag_label(), 0, patch );
        dw_new->get ( flag_refine_patch, state->get_refinePatchFlag_label(), 0, patch );
        dbg_out4 << "flag_refine \t window " << flag_refine.getLowIndex() << flag_refine.getHighIndex() << std::endl;

        PatchFlag * patch_flag_refine = flag_refine_patch.get().get_rep();
        ConstVariable psi;
        dw_new->get ( psi, psi_label, 0, patch, Ghost::None, 0 );
        dbg_out4 << "psi \t window " << psi.getLowIndex() << psi.getHighIndex() << std::endl;

        bool refine_patch = false;
        BlockRange range = get_flag_inner_range ( patch );
        dbg_out3 << "= Iterating over inner cells window " << range << std::endl;
        parallel_for ( range, [patch, &psi, &flag_refine, &refine_patch, this] ( int i, int j, int k )->void { error_estimate_test ( i, j, k, patch, get_view ( flag_refine ), refine_patch ); } );

        if ( refine_patch ) {
            patch_flag_refine->set();
        }
    }

    dbg_out2 << std::endl;
}
}

#endif








/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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

#include <CCA/Components/Wasatch/Expressions/Coordinate.h>

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <CCA/Components/Wasatch/FieldTypes.h>

//-- Uintah Includes --//

#include <Core/Grid/Variables/VarTypes.h>  // delt_vartype
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>

namespace WasatchCore {
  template < typename FieldT >
  Coordinates<FieldT>::Coordinates(const int idir)
  : Expr::Expression<FieldT>(),
    idir_(idir),
  shift_(Uintah::Vector(0,0,0))
  {}
  
  //--------------------------------------------------------------------

  template < typename FieldT >
  Coordinates<FieldT>::~Coordinates()
  {}
  
  //--------------------------------------------------------------------
  
  template < typename FieldT >
  void
  Coordinates<FieldT>::bind_operators( const SpatialOps::OperatorDatabase& opDB )
  {
    patchContainer_ = opDB.retrieve_operator<UintahPatchContainer>();
  }
  
  //--------------------------------------------------------------------

  template < typename FieldT >
  void
  Coordinates<FieldT>::evaluate()
  {
    using namespace SpatialOps;
    FieldT& phi = this->value();

    const Uintah::Patch* patch = patchContainer_->get_uintah_patch();
    const Uintah::Vector spacing = patch->dCell();
    const Direction stagLoc = get_staggered_location<FieldT>();
    switch (stagLoc) {
      case XDIR:
        shift_[0] = -spacing[0]*0.5;
        shift_[1] = 0.0;
        shift_[2] = 0.0;
        break;
      case YDIR:
        shift_[0] = 0.0;
        shift_[1] = -spacing[1]*0.5;
        shift_[2] = 0.0;
        break;
      case ZDIR:
        shift_[0] = 0.0;
        shift_[1] = 0.0;
        shift_[2] = -spacing[2]*0.5;
        break;
      case NODIR:
        shift_[0] = 0.0;
        shift_[1] = 0.0;
        shift_[2] = 0.0;
      default:
        break;
    }
    const Uintah::IntVector patchCellOffset = patch->getExtraCellLowIndex(1);

    // also touch ghost cells to avoid a communication on volume fractions
    for(Uintah::CellIterator iter(patch->getExtraCellIterator(1)); !iter.done(); iter++)
    {
      Uintah::IntVector iCell = *iter;
      const Uintah::Point xyz( patch->getCellPosition(iCell) );
      const Uintah::IntVector localUintahIJK = iCell - patchCellOffset;
      // now go to local indexing
      const SpatialOps::IntVec localIJK(localUintahIJK[0], localUintahIJK[1], localUintahIJK[2]);
      phi(localIJK) = xyz(idir_) + shift_[idir_];
    }
  }
    
  //--------------------------------------------------------------------

  template < typename FieldT >
  Coordinates<FieldT>::Builder::Builder( const Expr::Tag& result )
  : ExpressionBuilder(result)
  {
    if (result.name() == "XSVOL" ||
        result.name() == "XXVOL" ||
        result.name() == "XYVOL" ||
        result.name() == "XZVOL") idir_ = 0;
    else if (result.name() == "YSVOL" ||
        result.name() == "YXVOL" ||
        result.name() == "YYVOL" ||
        result.name() == "YZVOL") idir_ = 1;
    else if (result.name() == "ZSVOL" ||
             result.name() == "ZXVOL" ||
             result.name() == "ZYVOL" ||
             result.name() == "ZZVOL") idir_ = 2;
  }
  
  //--------------------------------------------------------------------
  
  template < typename FieldT >
  Expr::ExpressionBase*
  Coordinates<FieldT>::Builder::build() const
  {
    return new Coordinates<FieldT>(idir_);
  }
  
  //
  template class Coordinates<SVolField>;
  template class Coordinates<XVolField>;
  template class Coordinates<YVolField>;
  template class Coordinates<ZVolField>;
} // namespace WasatchCore

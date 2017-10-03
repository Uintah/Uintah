/*
 * The MIT License
 *
 * Copyright (c) 2017 The University of Utah
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

/*
 *  NOTE: we have split the implementations into several files to reduce the size
 *        of object files when compiling with nvcc, since we were crashing the
 *        linker in some cases.
 */

#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>
#include <spatialops/structured/SpatialMask.h>

namespace WasatchCore{

  // ###################################################################

  template< typename FieldT >
  void
  BCCopier<FieldT>::
  evaluate()
  {
    using namespace SpatialOps;
    FieldT& lhs =  this->value();
    const FieldT& src = src_->field_ref();
    if( this->spatialMask_ ){
      if( this->isStaggeredNormal_ ){
        masked_assign(STAGGERED_MASK, lhs, src);
      } else {
        masked_assign(*this->spatialMask_, lhs, src);
      }
    }
  }

  // ###################################################################
  // EXPLICIT INSTANTIATION
#include <CCA/Components/Wasatch/FieldTypes.h>
#define INSTANTIATE_BC_PROFILES(VOLT)   \
    template class BCCopier<VOLT>;

  INSTANTIATE_BC_PROFILES(SVolField)
  INSTANTIATE_BC_PROFILES(XVolField)
  INSTANTIATE_BC_PROFILES(YVolField)
  INSTANTIATE_BC_PROFILES(ZVolField)

} // namespace WasatchCore

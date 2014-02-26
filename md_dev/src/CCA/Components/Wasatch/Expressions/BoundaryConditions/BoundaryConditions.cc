/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
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

#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/BoundaryConditions.h>

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
void
ConstantBC<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    std::vector<SpatialOps::structured::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::structured::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    if(this->isStaggered_) {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii){
        f(*ig) = bcValue_;
        f(*ii) = bcValue_;
      }
    } else {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        f(*ig) = ( bcValue_- ci * f(*ii) ) / cg;
      }
    }
  }
}

// ###################################################################

template< typename FieldT >
void
LinearBC<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    double bcVal = 0.0;
    std::vector<SpatialOps::structured::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::structured::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    if(this->isStaggered_) {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        bcVal  = a_ * (*x_)(*ii) + b_;
        f(*ii) = bcVal;
        f(*ig) = bcVal;
      }
    } else {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        f(*ig) = ( ( a_ * (*x_)(*ig) + b_ ) - ci*f(*ii) ) / cg;
      }
    }
  }
}

// ###################################################################

template< typename FieldT >
void
ParabolicBC<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    double x = 0.0;
    double bcVal = 0.0;
    std::vector<SpatialOps::structured::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::structured::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    if(this->isStaggered_) {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        x = (*x_)(*ii) - x0_;
        bcVal = a_ * x*x + b_ * x + c_;
        f(*ii) = bcVal;
        f(*ig) = bcVal; // if the field is staggered, set the extra-cell value equal to the "boundary" value
      }
    } else {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        x = (*x_)(*ig) - x0_;
        f(*ig) = ( (a_ * x*x + b_ * x + c_) - ci*f(*ii) ) / cg;
      }
    }
  }
}

// ###################################################################

template< typename FieldT >
void
PowerLawBC<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    double bcVal = 0.0;
    std::vector<SpatialOps::structured::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::structured::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    if(this->isStaggered_) {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        bcVal  = phic_ * std::pow( 1.0 - std::fabs( (*x_)(*ig) - x0_ ) / R_ , 1.0/n_ );
        f(*ii) = bcVal;
        f(*ig) = bcVal;
      }
    } else {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        bcVal = phic_ * std::pow( 1.0 - std::fabs( (*x_)(*ig) - x0_ ) / R_ , 1.0/n_ );
        f(*ig) = ( bcVal - ci*f(*ii) ) / cg;
      }
    }
  }
}

// ###################################################################

template< typename FieldT >
void
BCCopier<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    std::vector<SpatialOps::structured::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::structured::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    if (this->isStaggered_) {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        f(*ig) = (*src_)(*ig);
        //f(*ii) = (*src_)(*ii);
      }
    } else {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        f(*ig) = (*src_)(*ig);
      }
    }
  }  
}

// ###################################################################
// EXPLICIT INSTANTIATION
#include <CCA/Components/Wasatch/FieldTypes.h>
#define INSTANTIATE_BC_PROFILES(VOLT) \
template class ConstantBC<VOLT>;      \
template class ConstantBC<SpatialOps::structured::FaceTypes<VOLT>::XFace>;      \
template class ConstantBC<SpatialOps::structured::FaceTypes<VOLT>::YFace>;      \
template class ConstantBC<SpatialOps::structured::FaceTypes<VOLT>::ZFace>;      \
template class LinearBC<VOLT>;        \
template class LinearBC<SpatialOps::structured::FaceTypes<VOLT>::XFace>;        \
template class LinearBC<SpatialOps::structured::FaceTypes<VOLT>::YFace>;        \
template class LinearBC<SpatialOps::structured::FaceTypes<VOLT>::ZFace>;        \
template class ParabolicBC<VOLT>;     \
template class ParabolicBC<SpatialOps::structured::FaceTypes<VOLT>::XFace>;     \
template class ParabolicBC<SpatialOps::structured::FaceTypes<VOLT>::YFace>;     \
template class ParabolicBC<SpatialOps::structured::FaceTypes<VOLT>::ZFace>;     \
template class PowerLawBC<VOLT>;     \
template class PowerLawBC<SpatialOps::structured::FaceTypes<VOLT>::XFace>;     \
template class PowerLawBC<SpatialOps::structured::FaceTypes<VOLT>::YFace>;     \
template class PowerLawBC<SpatialOps::structured::FaceTypes<VOLT>::ZFace>;     \
template class BCCopier<VOLT>;

INSTANTIATE_BC_PROFILES(SVolField)
INSTANTIATE_BC_PROFILES(XVolField)
INSTANTIATE_BC_PROFILES(YVolField)
INSTANTIATE_BC_PROFILES(ZVolField)

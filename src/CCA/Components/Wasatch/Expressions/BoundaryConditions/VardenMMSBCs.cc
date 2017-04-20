/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

#include <CCA/Components/Wasatch/Expressions/BoundaryConditions/VardenMMSBCs.h>

#include <math.h>

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
void
VarDen1DMMSDensity<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const TimeField& time = t_->field_ref();
  const double ci = this->ci_;
  const double cg = this->cg_;
  const double t = time[0];

  const double bcValue = -1 / ( (5/(exp(1125/( t + 10)) * (2 * t + 5)) - 1)/rho0_ - 5/(rho1_ * exp(1125 / (t + 10)) * (2 * t + 5)));
  if( (this->vecGhostPts_) && (this->vecInteriorPts_) ){
    std::vector<SpatialOps::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
      f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
    }
  }
}

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
void
VarDen1DMMSMixtureFraction<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const TimeField& time = t_->field_ref();
  const double ci = this->ci_;
  const double cg = this->cg_;
  
  const double t = time[0];  // this breaks GPU.
  
  if( (this->vecGhostPts_) && (this->vecInteriorPts_) ){
    std::vector<SpatialOps::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    const double bcValue = ( (5. / (2. * t +5.)) * exp(-1125 / (10. + t)) );
    for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
      f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
    }
  }
}

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
void
VarDen1DMMSMomentum<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  const TimeField& time = t_->field_ref();
  const double t = time[0];
  
  if( (this->vecGhostPts_) && (this->vecInteriorPts_) ){
    std::vector<IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    if( this->isStaggeredNormal_ ){
      if( side_==PLUS_SIDE ){
        const double bcValue = (5 * t * sin((30 * M_PI )/(3 * t + 30)))/(( (t * t) + 1)*((5 / (exp(1125/(t + 10))*(2 * t + 5)) - 1)/rho0_ - 5/(rho1_ * exp(1125/(t + 10))*(2 * t + 5))));
        for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
          f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
          f(*ii) = ( bcValue - ci*f(*ig) ) / cg;
        }
      }
      else if( side_ == MINUS_SIDE ){
        const double bcValue = (5 * t * sin((-30 * M_PI )/(3 * t + 30)))/(( (t * t) + 1)*((5 / (exp(1125/(t + 10))*(2 * t + 5)) - 1)/rho0_ - 5/(rho1_ * exp(1125/(t + 10))*(2 * t + 5))));
        for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
          f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
          f(*ii) = ( bcValue - ci*f(*ii) ) / cg;
        }
      }
    }
    else{
      if( side_==PLUS_SIDE ){
        const double bcValue = (5 * t * sin((30 * M_PI )/(3 * t + 30)))/(( (t * t) + 1)*((5 / (exp(1125/(t + 10))*(2 * t + 5)) - 1)/rho0_ - 5/(rho1_ * exp(1125/(t + 10))*(2 * t + 5))));
        for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
          f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
        }
      }
      else if( side_ == MINUS_SIDE ){
        const double bcValue = (5 * t * sin((-30 * M_PI )/(3 * t + 30)))/(( (t * t) + 1)*((5 / (exp(1125/(t + 10))*(2 * t + 5)) - 1)/rho0_ - 5/(rho1_ * exp(1125/(t + 10))*(2 * t + 5))));
        for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
          f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
        }
      }
    }
  }
}

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
void
VarDen1DMMSSolnVar<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const TimeField& time = t_->field_ref();
  const double ci = this->ci_;
  const double cg = this->cg_;
  const double t = time[0];  // this breaks GPU
  
  if( (this->vecGhostPts_) && (this->vecInteriorPts_) ){
    std::vector<SpatialOps::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    const double bcValue = -5/(exp(1125/(t + 10))*(2 * t + 5) * ((5/(exp(1125/(t + 10))*(2 * t + 5)) - 1)/rho0_ - 5/(rho1_ * exp(1125/(t + 10))*(2 * t + 5))));
    for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
      f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
    }
  }
}

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
void
VarDen1DMMSVelocity<FieldT>::evaluate()
{
  using namespace SpatialOps;
  
  FieldT& f = this->value();
  const TimeField& time = t_->field_ref();
  const double ci = this->ci_;
  const double cg = this->cg_;
  const double t = time[0];  // this breaks GPU
  
  if( (this->vecGhostPts_) && (this->vecInteriorPts_) ){
    std::vector<IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    if( this->isStaggeredNormal_ ){
      if( side_== PLUS_SIDE ){
        const double bcValue = ( ((-5 * t)/( t * t + 1)) * sin(10 * M_PI / (t + 10) ) );
        for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
          f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
          f(*ii) = ( bcValue - ci*f(*ig) ) / cg;
        }
      }
      else if( side_ == MINUS_SIDE ){
        const double bcValue = ( ((-5 * t)/( t * t + 1)) * sin(-10 * M_PI / (t + 10) ) );
        for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
          f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
          f(*ii) = ( bcValue - ci*f(*ii) ) / cg;
        }
      }
    } else {
      if( side_== PLUS_SIDE ){
        const double bcValue = ( ((-5 * t)/( t * t + 1)) * sin(10 * M_PI / (t + 10) ) );
        for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
          f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
        }
      }
      else if( side_ == MINUS_SIDE ){
        const double bcValue = ( ((-5 * t)/( t * t + 1)) * sin(-10 * M_PI / (t + 10) ) );
        for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
          f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
        }
      }
    }
  }
}

//------------------

#define INSTANTIATE_VARDEN_MMS_BCS( VOLT )            \
  template class VarDen1DMMSDensity<VOLT>;            \
  template class VarDen1DMMSMixtureFraction<VOLT>;    \
  template class VarDen1DMMSSolnVar<VOLT>;            \
  template class VarDen1DMMSMomentum<VOLT>;           \
  template class VarDen1DMMSVelocity<VOLT>;           

INSTANTIATE_VARDEN_MMS_BCS(SVolField)
INSTANTIATE_VARDEN_MMS_BCS(XVolField)
INSTANTIATE_VARDEN_MMS_BCS(YVolField)
INSTANTIATE_VARDEN_MMS_BCS(ZVolField)

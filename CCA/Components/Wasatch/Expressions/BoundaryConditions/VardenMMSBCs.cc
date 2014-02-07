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
  const double ci = this->ci_;
  const double cg = this->cg_;
  const double t = (*t_)[0];

  const double bcValue = -1 / ( (5/(exp(1125/( t + 10)) * (2 * t + 5)) - 1)/rho0_ - 5/(rho1_ * exp(1125 / (t + 10)) * (2 * t + 5)));
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    std::vector<SpatialOps::structured::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::structured::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
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
  const double ci = this->ci_;
  const double cg = this->cg_;
  
  const double t = (*t_)[0];  // this breaks GPU.
  
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    std::vector<SpatialOps::structured::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::structured::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
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
  namespace SS = SpatialOps::structured;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  const double t = (*t_)[0];
  
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    std::vector<SS::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SS::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    if (this->isStaggered_) {
      if (side_==SS::PLUS_SIDE) {
        const double bcValue = (5 * t * sin((30 * M_PI )/(3 * t + 30)))/(( (t * t) + 1)*((5 / (exp(1125/(t + 10))*(2 * t + 5)) - 1)/rho0_ - 5/(rho1_ * exp(1125/(t + 10))*(2 * t + 5))));
        for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
          f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
          f(*ii) = ( bcValue - ci*f(*ig) ) / cg;
        }
      } else if (side_ == SS::MINUS_SIDE) {
        const double bcValue = (5 * t * sin((-30 * M_PI )/(3 * t + 30)))/(( (t * t) + 1)*((5 / (exp(1125/(t + 10))*(2 * t + 5)) - 1)/rho0_ - 5/(rho1_ * exp(1125/(t + 10))*(2 * t + 5))));
        for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
          f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
          f(*ii) = ( bcValue - ci*f(*ii) ) / cg;
        }
      }
    } else {
      if (side_==SS::PLUS_SIDE) {
        const double bcValue = (5 * t * sin((30 * M_PI )/(3 * t + 30)))/(( (t * t) + 1)*((5 / (exp(1125/(t + 10))*(2 * t + 5)) - 1)/rho0_ - 5/(rho1_ * exp(1125/(t + 10))*(2 * t + 5))));
        for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
          f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
        }
      } else if (side_ == SS::MINUS_SIDE) {
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
  const double ci = this->ci_;
  const double cg = this->cg_;
  const double t = (*t_)[0];  // this breaks GPU
  
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    std::vector<SpatialOps::structured::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::structured::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
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
  namespace SS = SpatialOps::structured;
  
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  const double t = (*t_)[0];  // this breaks GPU
  
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    std::vector<SS::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SS::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    if (this->isStaggered_) {
      if (side_== SS::PLUS_SIDE) {
        const double bcValue = ( ((-5 * t)/( t * t + 1)) * sin(10 * M_PI / (t + 10) ) );
        for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
          f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
          f(*ii) = ( bcValue - ci*f(*ig) ) / cg;
        }
      } else if (side_ == SS::MINUS_SIDE) {
        const double bcValue = ( ((-5 * t)/( t * t + 1)) * sin(-10 * M_PI / (t + 10) ) );
        for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
          f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
          f(*ii) = ( bcValue - ci*f(*ii) ) / cg;
        }
      }
    } else {
      if (side_== SS::PLUS_SIDE) {
        const double bcValue = ( ((-5 * t)/( t * t + 1)) * sin(10 * M_PI / (t + 10) ) );
        for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
          f(*ig) = ( bcValue - ci*f(*ii) ) / cg;
        }
      } else if (side_ == SS::MINUS_SIDE) {
        const double bcValue = ( ((-5 * t)/( t * t + 1)) * sin(-10 * M_PI / (t + 10) ) );
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
VarDenCorrugatedMMSMixFracBC<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  const double t = (*(this->t_))[0];  // this breaks GPU
  const FieldT& x = *(this->x_);
  const FieldT& y = *(this->y_);
  
  const double k = this->k_;
  const double uf = this->uf_;
  const double vf = this->vf_;
  const double w = this->w_;
  const double a = this->a_;
  const double b = this->b_;
  const double r0 = this->r0_;
  const double r1 = this->r1_;

  
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    std::vector<SpatialOps::structured::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::structured::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    const double dn1 = 1 + r0/r1;
    const double dn2 = 1 - r0/r1;
    for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
      const double xh = uf * t - x(*ii) + a * cos(k * (vf * t - y(*ii)));
      const double s0 = tanh(b * xh * exp(-w*t));
      const double bcVal = (1.0+s0)/(dn1 + s0*dn2);
      f(*ig) = ( bcVal - ci*f(*ii) ) / cg;
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
VarDenCorrugatedMMSRhofBC<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  const double t = (*(this->t_))[0];  // this breaks GPU
  const FieldT& x = *(this->x_);
  const FieldT& y = *(this->y_);
  
  const double k = this->k_;
  const double uf = this->uf_;
  const double vf = this->vf_;
  const double w = this->w_;
  const double a = this->a_;
  const double b = this->b_;
  const double r1 = this->r1_;
  
  
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    std::vector<SpatialOps::structured::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::structured::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index

    for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
      const double xh = uf * t - x(*ii) + a * cos(k * (vf * t - y(*ii)));
      const double s0 = tanh(b * xh * exp(-w*t));
      const double bcVal = 0.5*r1*(1.0 + s0);
      f(*ig) = ( bcVal - ci*f(*ii) ) / cg;
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
VarDenCorrugatedMMSVelocityBC<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  const double t = (*(this->t_))[0];  // this breaks GPU
  const FieldT& x = *(this->x_);
  const FieldT& y = *(this->y_);
  
  const double k = this->k_;
  const double uf = this->uf_;
  const double vf = this->vf_;
  const double w = this->w_;
  const double a = this->a_;
  const double b = this->b_;
  const double r0 = this->r0_;
  const double r1 = this->r1_;
  
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    std::vector<SpatialOps::structured::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::structured::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    if (this->isStaggered_) {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        const double xh = uf * t - x(*ii) + a * cos(k * (vf * t - y(*ii)));
        const double s0 = exp(2.0*b*xh*exp(-w * t)) + 1.0;
        const double rho = 0.5*(r0 + r1 + (r1-r0)*tanh( b*exp(-w*t)*xh ));
        const double bcVal = (r1 - r0) / rho * ( - w * xh + (w*xh - uf)/s0 + w*log(s0)/(2.0*b*exp(-w*t)));
        f(*ii) = bcVal;
        f(*ig) = bcVal;
      }
    } else {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        const double xh = uf * t - x(*ii) + a * cos(k * (vf * t - y(*ii)));
        const double s0 = exp(2.0*b*xh*exp(-w * t)) + 1.0;
        const double rho = 0.5*(r0 + r1 + (r1-r0)*tanh( b*exp(-w*t)*xh ));
        const double bcVal = (r1 - r0) / rho * ( - w * xh + (w*xh - uf)/s0 + w*log(s0)/(2.0*b*exp(-w*t)));
        f(*ig) = ( bcVal - ci*f(*ii) ) / cg;
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
VarDenCorrugatedMMSMomBC<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  const double t = (*(this->t_))[0];  // this breaks GPU
  const FieldT& x = *(this->x_);
  const FieldT& y = *(this->y_);
  
  const double k = this->k_;
  const double uf = this->uf_;
  const double vf = this->vf_;
  const double w = this->w_;
  const double a = this->a_;
  const double b = this->b_;
  const double r0 = this->r0_;
  const double r1 = this->r1_;
  
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    std::vector<SpatialOps::structured::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::structured::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    if (this->isStaggered_) {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        const double xh = uf * t - x(*ii) + a * cos(k * (vf * t - y(*ii)));
        const double s0 = exp(2.0*b*xh*exp(-w * t)) + 1.0;
        const double bcVal = (r1 - r0) * ( - w * xh + (w*xh - uf)/s0 + w*log(s0)/(2.0*b*exp(-w*t)));
        f(*ii) = bcVal;
        f(*ig) = bcVal;
      }
    } else {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        const double xh = uf * t - x(*ii) + a * cos(k * (vf * t - y(*ii)));
        const double s0 = exp(2.0*b*xh*exp(-w * t)) + 1.0;
        const double bcVal = (r1 - r0) * ( - w * xh + (w*xh - uf)/s0 + w*log(s0)/(2.0*b*exp(-w*t)));
        f(*ig) = ( bcVal - ci*f(*ii) ) / cg;
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
VarDenCorrugatedMMSyMomBC<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  const double t = (*(this->t_))[0];  // this breaks GPU
  const FieldT& x = *(this->x_);
  const FieldT& y = *(this->y_);
  
  const double k = this->k_;
  const double uf = this->uf_;
  const double vf = this->vf_;
  const double w = this->w_;
  const double a = this->a_;
  const double b = this->b_;
  const double r0 = this->r0_;
  const double r1 = this->r1_;
  
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    std::vector<SpatialOps::structured::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::structured::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    if (this->isStaggered_) {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        const double xh = uf * t - x(*ii) + a * cos(k * (vf * t - y(*ii)));
        const double rho = 0.5*(r0 + r1 + (r1-r0)*tanh( b*exp(-w*t)*xh ));
        const double bcVal = rho * vf;
        f(*ii) = bcVal;
        f(*ig) = bcVal;
      }
    } else {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        const double xh = uf * t - x(*ii) + a * cos(k * (vf * t - y(*ii)));
        const double rho = 0.5*(r0 + r1 + (r1-r0)*tanh( b*exp(-w*t)*xh ));
        const double bcVal = rho * vf;
        f(*ig) = ( bcVal - ci * f(*ii) ) / cg;
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
VarDenCorrugatedMMSRho<FieldT>::evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;
  const double t = (*(this->t_))[0];  // this breaks GPU
  const FieldT& x = *(this->x_);
  const FieldT& y = *(this->y_);
  
  const double k = this->k_;
  const double uf = this->uf_;
  const double vf = this->vf_;
  const double w = this->w_;
  const double a = this->a_;
  const double b = this->b_;
  const double r0 = this->r0_;
  const double r1 = this->r1_;
  
  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    std::vector<SpatialOps::structured::IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<SpatialOps::structured::IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    if (this->isStaggered_) {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        const double xh = uf * t - x(*ii) + a * cos(k * (vf * t - y(*ii)));
        const double bcVal = 0.5*(r0 + r1 + (r1-r0)*tanh( b*exp(-w*t)*xh ));
        f(*ii) = bcVal;
        f(*ig) = bcVal;
      }
    } else {
      for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
        const double xh = uf * t - x(*ii) + a * cos(k * (vf * t - y(*ii)));
        const double bcVal = 0.5*(r0 + r1 + (r1-r0)*tanh( b*exp(-w*t)*xh ));
        f(*ig) = ( bcVal - ci*f(*ii) ) / cg;
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
  template class VarDen1DMMSVelocity<VOLT>;           \
  template class VarDenCorrugatedMMSBCBase<VOLT>; \
  template class VarDenCorrugatedMMSVelocityBC<VOLT>; \
  template class VarDenCorrugatedMMSMomBC<VOLT>;\
  template class VarDenCorrugatedMMSMixFracBC<VOLT>;  \
  template class VarDenCorrugatedMMSRhofBC<VOLT>;     \
  template class VarDenCorrugatedMMSyMomBC<VOLT>;     \
  template class VarDenCorrugatedMMSRho<VOLT>;


INSTANTIATE_VARDEN_MMS_BCS(SVolField)
INSTANTIATE_VARDEN_MMS_BCS(XVolField)
INSTANTIATE_VARDEN_MMS_BCS(YVolField)
INSTANTIATE_VARDEN_MMS_BCS(ZVolField)

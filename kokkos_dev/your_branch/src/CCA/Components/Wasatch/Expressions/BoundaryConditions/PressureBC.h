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

#ifndef PressureBC_Expr_h
#define PressureBC_Expr_h

#include <expression/Expression.h>
#include <CCA/Components/Wasatch/TagNames.h>

template< typename FieldT >
class PressureBC
: public BoundaryConditionBase<FieldT>
{
  PressureBC( const Expr::Tag& velTag ) :
  velTag_ (velTag)
  {}
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& resultTag,
            const Expr::Tag& velTag ) :
    ExpressionBuilder(resultTag),
    velTag_ (velTag)
    {}
    Expr::ExpressionBase* build() const{ return new PressureBC(velTag_); }
  private:
    const Expr::Tag velTag_;
  };
  
  ~PressureBC(){}
  void advertise_dependents( Expr::ExprDeps& exprDeps )
  {
    exprDeps.requires_expression( velTag_ );
    const Wasatch::TagNames& tagNames = Wasatch::TagNames::self();
    exprDeps.requires_expression( tagNames.timestep );
  }
  
  void bind_fields( const Expr::FieldManagerList& fml )
  {
    u_ = &fml.template field_ref<FieldT>( velTag_ );
    const Wasatch::TagNames& tagNames = Wasatch::TagNames::self();
    dt_ = &fml.template field_ref<double>( tagNames.timestep );
  }

  void evaluate()
  {
    using namespace SpatialOps;
    using namespace SpatialOps::structured;
    FieldT& f = this->value();
    
    if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {      
      std::vector<IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
      std::vector<IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
      const IntVec& offset = this->bndNormal_;
      const double sign = (offset[0] >=0 && offset[1] >= 0 && offset[2] >= 0) ? 1.0 : -1.0;
      
      if(this->isStaggered_) {
        for( ; ii != (this->vecInteriorPts_)->end(); ++ii, ++ig ){
          const double ub = (*u_)(*ii);             // boundary cell
          const double ui = (*u_)(*ii - offset);    // interior cell
          const double fi = f(*ii - offset);

          // uncomment the following two lines if you want to use second order one-sided difference
          //const double uii = (*u_)(*ii - offset*2); // interior interior cell
          //const double fii = f(*ii - offset*2);     // interior interior partial RHS
          if ( sign*ub < 0.0 ) { // flow coming in
            f(*ii) = (1.0/ *dt_)*( -ub + ui ) + fi;
            // uncomment the line below to use second-order one-sided difference
            //f(*ii) = (1.0/ *dt_)*( -ub + 4.0/3.0*ui - 1.0/3.0*uii) + 4.0/3.0*fi - 1.0/3.0*fii;
          } else { // flow coming out
            f(*ii) = -(1.0/ *dt_) * ub;
          }
        }
        
        if (this->interiorEdgePoints_) {
          std::vector<IntVec>::const_iterator ic = (this->interiorEdgePoints_)->begin(); // ii is the interior flat index
          for (; ic != (this->interiorEdgePoints_)->end(); ++ic) {
            const double ub  = (*u_)(*ic);            // boundary cell
            f(*ic) = -(1.0/ *dt_) * ub;
          }
        }

      } else {
        std::ostringstream msg;
        msg << "ERROR: You cannot use the OutflowBC boundary expression with non staggered fields!"
        << " This boundary condition can only be applied on the momentum partial RHS." << std::endl;
        std::cout << msg.str() << std::endl;
        throw std::runtime_error(msg.str());
      }
    }
  }
  
private:
  const FieldT* u_;
  const double* dt_;
  const Expr::Tag velTag_;
};

#endif // PressureBC_Expr_h

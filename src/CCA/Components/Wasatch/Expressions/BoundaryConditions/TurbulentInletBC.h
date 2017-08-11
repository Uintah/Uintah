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

#ifndef TrubulentInlet_Expr_h
#define TrubulentInlet_Expr_h

#include <expression/Expression.h>
#include <Core/IO/UintahZlibUtil.h>
#include <Core/IO/UintahIFStreamUtil.h>
#include <istream>
#include <CCA/Components/Wasatch/TagNames.h>

template< typename FieldT >
class TurbulentInletBC
: public WasatchCore::BoundaryConditionBase<FieldT>
{
  typedef typename SpatialOps::SingleValueField TimeField;
  TurbulentInletBC( const std::string inputFileName,
                    const std::string velDir,
                    const int period,
                    const double timePeriod);
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& resultTag,
             const std::string inputFileName,
             const std::string velDir,
             const int period,
             const double timePeriod )
    : ExpressionBuilder(resultTag),
      inputFileName_(inputFileName),
      velDir_(velDir),
      period_(period),
      timePeriod_(timePeriod)
    {}

    Expr::ExpressionBase* build() const{
      return new TurbulentInletBC(inputFileName_, velDir_, period_, timePeriod_);
    }
  private:
    const std::string inputFileName_;
    const std::string velDir_;
    const int period_;
    const double timePeriod_;
  };
  
  ~TurbulentInletBC(){}
  void evaluate();
  
private:
  DECLARE_FIELDS(TimeField, t_, dt_)
  const std::string velDir_;
  const int period_;
  const double timePeriod_;
  std::vector< std::vector< std::vector<double> > > fluct_; // velocity fluctuations
  SpatialOps::IntVec minC_; //store indicies of lowest corner value of bounding box aroudn inlet
  int NT_, jSize_, kSize_, iComponent_, jComponent_;

  double coord_;
  double dx_;
  bool firsttimestep_;
  // member functions
  int calculate_time_index();
};

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename FieldT >
TurbulentInletBC<FieldT>::
TurbulentInletBC( const std::string inputFileName,
                  const std::string velDir,
                  const int period,
                  const double timePeriod )
  : velDir_(velDir),
    period_(period),
    timePeriod_(timePeriod)
{
  const WasatchCore::TagNames& tagNames = WasatchCore::TagNames::self();
   t_ = this->template create_field_request<TimeField>(tagNames.time);
   dt_ = this->template create_field_request<TimeField>(tagNames.timestep);
  
  std::ifstream ifs( inputFileName.c_str() );

  if( !ifs ) {
    proc0cout << "Error opening file for Turbulent Inlet: " << inputFileName << std::endl;
    throw Uintah::ProblemSetupException("Unable to open the given input file: " + inputFileName, __FILE__, __LINE__);
  }
  
  std::string face;
  Uintah::getValue(ifs,face);
  if (face.compare("x-")==0 || face.compare("x+") ==0) {
    iComponent_ = 1;
    jComponent_ = 2;
  } else if (face.compare("y-")==0 || face.compare("y+") ==0) {
    iComponent_ = 0;
    jComponent_ = 2;
  } else if (face.compare("z-")==0 || face.compare("z+") ==0) {
    iComponent_ = 0;
    jComponent_ = 1;
  }
  
  Uintah::getValue(ifs, NT_);
  Uintah::getValue(ifs, jSize_);
  Uintah::getValue(ifs, kSize_);
  Uintah::getValue(ifs,minC_[0]);
  Uintah::getValue(ifs,minC_[1]);
  Uintah::getValue(ifs,minC_[2]);
  int nPts;
  Uintah::getValue(ifs,nPts);
  Uintah::getValue(ifs,dx_);
  
  fluct_.resize(NT_,std::vector< std::vector<double> >(jSize_,std::vector<double>(kSize_)));
  int t,j,k;
  double u,v,w;
  for (int n = 0; n<nPts; n++) {
    Uintah::getValue(ifs,t);
    Uintah::getValue(ifs,j);
    Uintah::getValue(ifs,k);
    Uintah::getValue(ifs,u);
    Uintah::getValue(ifs,v);
    Uintah::getValue(ifs,w);
    fluct_[t][j][k] = (velDir_.compare("X")==0) ? u : (velDir_.compare("Y")==0 ? v : w);
  }
  firsttimestep_ = true;
}

//--------------------------------------------------------------------

template< typename FieldT >
int
TurbulentInletBC<FieldT>::calculate_time_index() {
  const TimeField& t = t_->field_ref();
  // floor - get the number of dxs into the data
  int tindex =(int) std::floor(t[0]/dx_);
  
  while (tindex >= NT_- 1) {
    tindex -= (NT_-1);
  }

  // coordinate relative to the current turbulent data interval
  coord_ = t[0] - tindex*dx_;

  // OPTIONAL: make sure we match the data point at the begining of new turbulent data
  //if (coord_ < *dt_) coord_ = 0.0;
  
  return tindex;
}

//--------------------------------------------------------------------

template< typename FieldT >
void
TurbulentInletBC<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  FieldT& f = this->value();
  const double ci = this->ci_;
  const double cg = this->cg_;  

  const int tIndex = calculate_time_index();

  IntVec globalCellIJK;

  if ( (this->vecGhostPts_) && (this->vecInteriorPts_) ) {
    std::vector<IntVec>::const_iterator ig = (this->vecGhostPts_)->begin();    // ig is the ghost flat index
    std::vector<IntVec>::const_iterator ii = (this->vecInteriorPts_)->begin(); // ii is the interior flat index
    for( ; ig != (this->vecGhostPts_)->end(); ++ig, ++ii ){
      globalCellIJK = *ig - minC_ + this->patchCellOffset_;
      // linear interpolation between the turbulent-data points
      const double y0 = fluct_[tIndex    ][ globalCellIJK[iComponent_] ][ globalCellIJK[jComponent_] ];
      const double y1 = fluct_[tIndex + 1][ globalCellIJK[iComponent_] ][ globalCellIJK[jComponent_] ];
      const double a = (y1-y0)/dx_;
      const double bcValue = a*coord_ + y0;
      //double bcValue_ = fluct_[tIndex][globalCellIJK[iComponent_]][globalCellIJK[jComponent_]];
      f(*ig) = ( bcValue - ci * f(*ii) ) / cg;
      if (this->isStaggeredNormal_) f(*ii) = ( bcValue - ci * f(*ig) ) / cg;
    }
  }
}

//--------------------------------------------------------------------

#endif // TrubulentInletBC_Expr_h

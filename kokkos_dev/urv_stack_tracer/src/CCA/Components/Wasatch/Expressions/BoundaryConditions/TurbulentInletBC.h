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

#ifndef TrubulentInlet_Expr_h
#define TrubulentInlet_Expr_h

#include <expression/Expression.h>
#include <Core/IO/UintahZlibUtil.h>
#include <Core/IO/UintahIFStreamUtil.h>
#include <istream>
#include <CCA/Components/Wasatch/TagNames.h>

template< typename FieldT >
class TurbulentInletBC
: public BoundaryConditionBase<FieldT>
{
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
            const double timePeriod) :
    ExpressionBuilder(resultTag),
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
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  // member variables
  const double *t_;
  const double *dt_;
  const Expr::Tag timeTag_, timestepTag_;
  const std::string velDir_;  
  const int period_;
  const double timePeriod_;
  std::vector< std::vector< std::vector<double> > > fluct_; // velocity fluctuations
  std::vector<int> minC_; //store indicies of lowest corner value of bounding box aroudn inlet
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
  : timeTag_    ( Wasatch::TagNames::self().time ),
    timestepTag_( Wasatch::TagNames::self().timestep ),
    velDir_(velDir),
    period_(period),
    timePeriod_(timePeriod)
{
  using namespace std;
  using namespace Uintah;
  
  ifstream ifs( inputFileName.c_str() );

  if( !ifs ) {
    proc0cout << "Error opening file for Turbulent Inlet: " << inputFileName << endl;
    throw ProblemSetupException("Unable to open the given input file: " + inputFileName, __FILE__, __LINE__);
  }
  
  std::string face;
  getValue(ifs,face);
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
  
  getValue(ifs, NT_);
  getValue(ifs, jSize_);
  getValue(ifs, kSize_);
  minC_.resize(3);
  getValue(ifs,minC_[0]);
  getValue(ifs,minC_[1]);
  getValue(ifs,minC_[2]);
  int nPts;
  getValue(ifs,nPts);
  getValue(ifs,dx_);
  
  fluct_.resize(NT_,vector< vector<double> >(jSize_,vector<double>(kSize_)));
  int t,j,k;
  double u,v,w;
  for (int n = 0; n<nPts; n++) {
    getValue(ifs,t);
    getValue(ifs,j);
    getValue(ifs,k);
    getValue(ifs,u);
    getValue(ifs,v);
    getValue(ifs,w);
    fluct_[t][j][k] = (velDir_.compare("X")==0) ? u : (velDir_.compare("Y")==0 ? v : w);
  }
  firsttimestep_ = true;
}

//--------------------------------------------------------------------

template< typename FieldT >
void
TurbulentInletBC<FieldT>::advertise_dependents(Expr::ExprDeps& exprDeps)
{
  exprDeps.requires_expression( timeTag_     );
  exprDeps.requires_expression( timestepTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
void
TurbulentInletBC<FieldT>::bind_fields(const Expr::FieldManagerList& fml)
{
  const typename Expr::FieldMgrSelector<double>::type& fm = fml.template field_manager<double>();
  t_  = &fm.field_ref( timeTag_     );
  dt_ = &fm.field_ref( timestepTag_ );
}

//--------------------------------------------------------------------

template< typename FieldT >
int
TurbulentInletBC<FieldT>::calculate_time_index() {
  
  // floor - get the number of dxs into the data
  int tindex =(int) std::floor(*t_/dx_);
  
  while (tindex >= NT_- 1) {
    tindex -= (NT_-1);
  }

  // coordinate relative to the current turbulent data interval
  coord_ = *t_ - tindex*dx_;

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

  int tIndex = calculate_time_index();

  SpatialOps::structured::IntVec localCellIJK;
  SpatialOps::structured::IntVec globalCellIJK;  
  
  std::vector<int>::const_iterator ig = (this->flatGhostPoints_).begin();    // ig is the ghost flat index
  std::vector<int>::const_iterator ii = (this->flatInteriorPoints_).begin(); // ii is the interior flat index

  for( ; ig != (this->flatGhostPoints_).end(); ++ig, ++ii ){
    // get i,j,k index from flat index
    localCellIJK  = f.window_with_ghost().ijk_index_from_local(*ig);
    globalCellIJK = localCellIJK - SpatialOps::structured::IntVec(1,1,1) - minC_ + this->patchCellOffset_;

    // linear interpolation between the turbulent-data points
    const double y0 = fluct_[tIndex][globalCellIJK[iComponent_]][globalCellIJK[jComponent_]];
    const double y1 = fluct_[tIndex + 1][globalCellIJK[iComponent_]][globalCellIJK[jComponent_]];
    const double a = (y1-y0)/dx_;
    const double bcValue = a*coord_ + y0;
    
    //double bcValue_ = fluct_[tIndex][globalCellIJK[iComponent_]][globalCellIJK[jComponent_]];
    f[*ig] = ( bcValue - ci * f[*ii] ) / cg;
  }
}

//--------------------------------------------------------------------

#endif // TrubulentInletBC_Expr_h

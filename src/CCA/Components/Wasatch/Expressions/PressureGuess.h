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

#ifndef PRESSURE_GUESS
#define PRESSURE_GUESS

#include <vector>
#include <iostream>

//-- Boost --//
#include <boost/shared_ptr.hpp>

//-- ExprLib Includes --//
#include <expression/Expression.h>

//-- SpatialOps Includes --//
#include <spatialops/structured/FVStaggered.h>

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/TimeIntegratorTools.h>
#include <CCA/Components/Wasatch/Wasatch.h>
#include <CCA/Components/Wasatch/OldVariable.h>

/**
 *  \class PressureGuess
 *  \ingroup Expressions
 *  \author Mokbel Karam
 *  \brief Calculates the pseudo-pressure approximations at the 
 *         intermediate stages of Low-Cost Runge-Kutta integrators 
 *         (for now it is used only for incompressible flow).
 * 
 * This expression relies on pseudo-pressures from previous time levels.
 */


class PressureApproximationsHelper; //forward declaration

class PressureGuess
: public Expr::Expression<SpatialOps::SVolField>
{
  typedef SpatialOps::SVolField PFieldT;
  typedef SpatialOps::SingleValueField TimeField;
  typedef boost::shared_ptr<const Expr::FieldRequest<PFieldT> > fieldRequestPtr;
  typedef boost::shared_ptr<const Expr::FieldRequest<SpatialOps::SpatialField<SpatialOps::SingleValue> > > timeFieldRequestPtr;

  const int numtimesteps_; 
  Expr::TagList oldPressureTags_;

  
//   DECLARE_FIELDS(PFieldT, pressure_NM_0_);
  DECLARE_FIELDS(TimeField, rkStage_)
  DECLARE_VECTOR_OF_FIELDS( PFieldT, old_pressure_Fields_ ) 

  const WasatchCore::TimeIntegrator* timeIntInfo_;

  //used to help in the construction of pressure approximations and used in the evaluate function.
  PressureApproximationsHelper * pressure_approximation_helper_;
  
  // constructor
  PressureGuess(const int order);

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const int order_;
  public:
    /**
     *  \param result the result of this expression
     *  \param order the order of the approximation
     */
    Builder( const Expr::Tag& result,
            const int order);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  };

  ~PressureGuess();
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
};

/**
 * @brief The following is the implementation of the 
 *        strategy pattern for the pressure approximations construction
 * 
 */

class PressureApproximationsInterface{
    /**
     * @brief Interface to all the classes that compute pressure approximation for the RK integrators.
     * 
     */
public:
    typedef boost::shared_ptr<const Expr::FieldRequest<SpatialOps::SpatialField<SpatialOps::SVol> > > fieldRequestPtr;

    virtual ~PressureApproximationsInterface(){}
    /**
     * @brief functions responsible for the computation of the pressure approximations at different stages.
     * 
     * @param pressureGuessValue  reference to the pressure guess field
     * @param old_pressure_Fields  vector with all the field requests for all the old pressure values required
     * @param timeIntInfo pointer to the time integrator information
     */
    virtual void compute_approximation_stage_1(SVolField& pressureGuessValue, std::vector<fieldRequestPtr> old_pressure_Fields,const WasatchCore::TimeIntegrator* timeIntInfo)  = 0;
    virtual void compute_approximation_stage_2(SVolField& pressureGuessValue, std::vector<fieldRequestPtr> old_pressure_Fields,const WasatchCore::TimeIntegrator* timeIntInfo)  = 0;
};

class PressureApproximationsHelper{
private:
    PressureApproximationsInterface * pressureApproximation_;
public:
    typedef SpatialOps::SingleValueField TimeField;
    typedef boost::shared_ptr<const Expr::FieldRequest<SpatialOps::SpatialField<SpatialOps::SVol> > > fieldRequestPtr;

    PressureApproximationsHelper(PressureApproximationsInterface * pressureApproximation = nullptr): pressureApproximation_(pressureApproximation){}
    ~PressureApproximationsHelper(){delete pressureApproximation_;}

    void set_approximation_strategy(PressureApproximationsInterface * pressureApproximation){
        delete this->pressureApproximation_;
        this->pressureApproximation_ = pressureApproximation;
    }
    
    /**
     * @brief This function is called from inside the evaluate function of the PressureGuess expression.
     * 
     * @param RKStage a reference to the single value field 
     * @param pressureGuessValue reference to the pressure guess field
     * @param old_pressure_Fields vector with all the field requests for all the old pressure values required
     * @param timeIntInfo pointer to the time integrator information
     */
    void compute_approximation(const TimeField& RKStage, SVolField& pressureGuessValue, std::vector<fieldRequestPtr> old_pressure_Fields,const WasatchCore::TimeIntegrator* timeIntInfo) 
    {   
        const double& rkStage_val = *RKStage.begin();
        if (rkStage_val==1)
            pressureApproximation_->compute_approximation_stage_1(pressureGuessValue,old_pressure_Fields,timeIntInfo);
        else if (rkStage_val==2)
            pressureApproximation_->compute_approximation_stage_2(pressureGuessValue,old_pressure_Fields,timeIntInfo);
    }
};

/**
 * @brief the implementation of the pressure approximations
 * 
 */

class RK2Approx: public PressureApproximationsInterface{

    public:
    void compute_approximation_stage_1(SVolField& pressureGuessValue, std::vector<fieldRequestPtr> oldPressureFields,const WasatchCore::TimeIntegrator* timeIntInfo)  {
        // std::cout<<"pressure guess field request vector size: "<<oldPressureFields.size()<<std::endl;
        const SVolField& pn = oldPressureFields[0]->field_ref();
        pressureGuessValue <<= pn;
    }
    void compute_approximation_stage_2(SVolField& pressureGuessValue, std::vector<fieldRequestPtr> oldPressureFields,const WasatchCore::TimeIntegrator* timeIntInfo)  {
        const SVolField& pn = oldPressureFields[0]->field_ref();
        pressureGuessValue <<= 0.0;
    }
};


class RK3Approx: public PressureApproximationsInterface{

    public:
    void compute_approximation_stage_1(SVolField& pressureGuessValue, std::vector<fieldRequestPtr> oldPressureFields,const WasatchCore::TimeIntegrator* timeIntInfo)  {

        const SVolField& pn = oldPressureFields[0]->field_ref();
        const SVolField& pnm1 = oldPressureFields[1]->field_ref();

        pressureGuessValue <<= (3*pn-pnm1)/2.0 ;
    }
    void compute_approximation_stage_2(SVolField& pressureGuessValue, std::vector<fieldRequestPtr> oldPressureFields,const WasatchCore::TimeIntegrator* timeIntInfo)  {
        // integrator's coefficients
        const WasatchCore::TimeIntegrator& timeintinfo = *timeIntInfo;
        const double b2 = timeintinfo.beta[1];

        const SVolField& pn = oldPressureFields[0]->field_ref();
        const SVolField& pnm1 = oldPressureFields[1]->field_ref();

        pressureGuessValue <<= 2.0 * b2 * (3*pn-pnm1)/2.0  +  b2 * (pn-pnm1);
    }
};


#endif // PRESSURE_GUESS

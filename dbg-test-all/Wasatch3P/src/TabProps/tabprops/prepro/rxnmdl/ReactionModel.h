/*
 * Copyright (c) 2014 The University of Utah
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

#ifndef REACTION_MODEL_H
#define REACTION_MODEL_H

#include <string>
#include <map>
#include <vector>
#include <cassert>

#include <cantera/Cantera.h>
#include <cantera/IdealGasMix.h>

#include <tabprops/prepro/TableBuilder.h>


//====================================================================


//--------------------------------------------------------------------
/** @class ReactionModel
 *  @brief Base class for creation of tabular reaction models.
 *
 *  @author James C. Sutherland @date January, 2006
 *
 *  This is a base class for reaction models.  Any existing reaction model
 *  could be wrapped in this base class.  The intention is that this helps
 *  construct a StateTable, which contains the desired quantities for either
 *  input to a mixing model or use in a cfd calculation.
 *
 *  Alternatively, you may wish to deal directly with the TableBuilder
 *  interface for constructing tables.
 *
 *  Currently, Cantera is required for evaluation of thermophysical
 *  properties.  This requirement could be lifted in favor of a general method
 *  of evaluating such properties.  Then Cantera could be one of a set of
 *  options...
 */
class ReactionModel{

 public:

  /**
   *  Set up a basic reaction model by creating a tablebuilder.
   *
   *  @param gas : Cantera object for evaluating properties of an ideal gas
   *  mixture.
   *
   *  @param indepVarNames : Names of the independent variables for this
   *  model.
   *
   *  @param interpolationOrder : Order of accuracy for interpolants in the
   *  resulting table.  Default is third order.
   *
   *  @param name : the name for this model.  By default, the table filename will
   *  use this as its prefix.
   */
  ReactionModel( Cantera_CXX::IdealGasMix & gas,
                 const std::vector<std::string> & indepVarNames,
                 const int interpolationOrder,
                 const std::string& name )
  : modelName_( name ),
    gasProps_( gas ),
    nSpec_( gas.nSpecies() ),
    interpOrder_( interpolationOrder ),
    tableBuilder_( gas, indepVarNames, interpolationOrder )
  {
    tableBuilder_.set_filename( modelName_ );
  }

  virtual ~ReactionModel(){}

  inline void select_for_output( const StateVarEvaluator::StateVars & var ){
    tableBuilder_.request_output( var );
  }

  inline void select_species_for_output( const std::string & specName,
                                         const StateVarEvaluator::StateVars sv )
  {
    assert( sv == StateVarEvaluator::SPECIES || sv == StateVarEvaluator::MOLEFRAC );
    tableBuilder_.request_output( sv, specName );
  }


  /** @brief Query the model name */
  inline const std::string & model_name() const{ return modelName_; }

  /**
   *  @brief Drive solution of relevant equations to implement the model.
   */
  virtual void implement() = 0;

//--------------------------------------------------------------------
 protected:

  /* the name of the reaction model */
  const std::string modelName_;
  Cantera_CXX::IdealGasMix & gasProps_;
  const int nSpec_;
  const int interpOrder_; ///< The order of interpolant to use in the table.
  TableBuilder tableBuilder_;

//--------------------------------------------------------------------
 private:

  ReactionModel( const ReactionModel &  );  // no copying
};

//====================================================================

#endif

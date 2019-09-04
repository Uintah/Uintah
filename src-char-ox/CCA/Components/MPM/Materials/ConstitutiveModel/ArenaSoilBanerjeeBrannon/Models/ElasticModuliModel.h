/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 * Copyright (c) 2013-2014 Callaghan Innovation, New Zealand
 * Copyright (c) 2015 Parresia Research Limited, New Zealand
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

#ifndef __ELASTICITY_MODEL_H__
#define __ELASTICITY_MODEL_H__


#include <CCA/Components/MPM/Materials/ConstitutiveModel/ArenaSoilBanerjeeBrannon/Models/ModelStateBase.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>

namespace Vaango {

  using ParameterDict = std::map<std::string, double>;

  /*! \class ElasticModuli
   *  \brief A struct containing the instantaneous bulk and shear modulus
   *  \author Biswajit Banerjee, 
  */
  struct ElasticModuli {
    double bulkModulus;
    double shearModulus;

    ElasticModuli(const double& bulk, const double& shear) 
      : bulkModulus(bulk), shearModulus(shear) { }
  };

  /*! \class ElasticModuliModel
   *  \brief A generic wrapper for various elasticity models
   *  \author Biswajit Banerjee, 
   *
   * Provides an abstract base class for various isotropic elasticity models
  */

  class ElasticModuliModel {

  public:
         
    //! Construct a elasticity model and initialize value
    /*! This is an abstract base class. */
    ElasticModuliModel();

    //! Destructor of elasticity model.  
    /*! Virtual to ensure correct behavior */
    virtual ~ElasticModuliModel();

    virtual void outputProblemSpec(Uintah::ProblemSpecP& ps) = 0;
             
    /////////////////////////////////////////////////////////////////////////
    /*!
      \brief Get the model parameters
     */
    /////////////////////////////////////////////////////////////////////////
    virtual ParameterDict getParameters() const = 0 ;

    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Get the elastic moduli
    */
    /////////////////////////////////////////////////////////////////////////
    virtual ElasticModuli getInitialElasticModuli() const = 0;
    virtual
    ElasticModuli getCurrentElasticModuli(const ModelStateBase* state) = 0; // not const
                                                                            // modifies d_bulk
    virtual ElasticModuli getElasticModuliLowerBound() const = 0;
    virtual ElasticModuli getElasticModuliUpperBound() const = 0;

    /////////////////////////////////////////////////////////////////////////
    /*! 
      \brief For partially saturated materials, get the drained and 
             partially saturated moduli
    */
    /////////////////////////////////////////////////////////////////////////
    virtual 
    void computeDrainedModuli(const double& I1_bar, 
                              const double& ev_p_bar,
                              double& KK,
                              double& GG) {};
    virtual
    void computePartialSaturatedModuli(const double& I1_eff_bar, 
                                       const double& pw_bar,
                                       const double& ev_p_bar,
                                       const double& phi,
                                       const double& S_w,
                                       double& KK,
                                       double& GG) {};
    
  };
} // End namespace Uintah
      
#endif  // __ELASTICITY_MODEL_H__


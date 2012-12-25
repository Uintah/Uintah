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

#ifndef OscillatingCylinder_Expr_h
#define OscillatingCylinder_Expr_h

#include <expression/Expression.h>
#include <spatialops/FieldReductions.h>

/**
 *  \class Oscillating Cylinder
 *  \author Tony Saad
 *  \date April, 2012
 *  \brief Implements an oscillating cylinder that can be used as volume
 fraction expression to illustrate the use of a moving geometry in a flowfield.
 */
class OscillatingCylinder : public Expr::Expression<SVolField>
{
public:
  
  struct Builder : public Expr::ExpressionBuilder
  {
    Builder(const Expr::Tag& result,
            const std::string axis,
            const std::vector<double> origin,
            const std::vector<double> oscillatingdir,            
            const double insideValue = 1.0,
            const double outsideValue = 0.0,
            const double radius = 0.1,
            const double frequency = 6.28318530717959,
            const double amplitude = 0.0);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const std::string axis_;
    const std::vector<double> origin_;
    const std::vector<double> oscillatingdir_;    
    const double insidevalue_, outsidevalue_, radius_;
    const double frequency_, amplitude_;
  };
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
  
private:
  
  OscillatingCylinder(const std::string axis,
                const std::vector<double> origin,  // origin on the minus face perpendicular to axis
                const std::vector<double> oscillatingdir,                      
                const double insideValue,
                const double outsideValue,
                const double radius,
                const double frequency,
                const double amplitude);
  Expr::Tag tag1_, tag2_;
  const std::vector<double> origin_;
  const std::vector<double> oscillatingdir_;
  const double insidevalue_, outsidevalue_, radius_;
  const double frequency_, amplitude_;
  const SVolField* field1_;
  const SVolField* field2_;
  const double* t_;
  Expr::Tag timet_;
};
#endif

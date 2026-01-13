/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

#ifndef ICE_VISCOSITY_H
#define ICE_VISCOSITY_H

#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

class DataWarehouse;
class ICELabel;
class Material;
class Patch;


class Viscosity {

//______________________________________________________________________
//
protected:


  enum callOrder{
    First = 0,              // model must be called before all others
    Middle = 1,             //
    Last = 2                // model must be called last
  };

private:

  callOrder m_callOrder;                         // models calling order

  std::string m_modelName;

  bool m_isViscosityDefined={false};             // if a patch has at least one cell that has a non-zero velocity

//______________________________________________________________________
//
public:
            // constructors
  Viscosity( ProblemSpecP& ps);

  Viscosity( ProblemSpecP & boxps,
             const GridP  & grid );

            // destructor
  virtual ~Viscosity();

  //__________________________________
  //
  virtual void
  outputProblemSpec(ProblemSpecP& vModels_ps) = 0;

  //__________________________________
  //           methods compute the viscosity
  virtual void
  computeDynViscosity(const Patch         * patch,
                      CCVariable<double>  & temp_CC,
                      CCVariable<double>  & mu) = 0;

  //__________________________________
  //
  virtual void
  computeDynViscosity(const Patch              * patch,
                      constCCVariable<double>  & temp_CC,
                      CCVariable<double>       & mu) = 0;

  //__________________________________
  //
  virtual void
  initialize (const Level * level ) = 0;


  //__________________________________
  //        Name of the model used
  void setName(std::string s)
  {
    m_modelName = s;
  }
  //__________________________________
  //
  std::string getName()
  {
    return m_modelName;
  }

  //__________________________________
  //        Is the viscosity non-zero anywhere on this patch
  void
  set_isViscosityDefined( bool in ){
    m_isViscosityDefined = in;
  }

  //__________________________________
  //
  bool
  isViscosityDefined(){
    return m_isViscosityDefined;
  }

  //__________________________________
  //    The last model in the input file takes precedence over all models.
  static
  bool  isDynViscosityDefined( std::vector<bool> trueFalse );


  //__________________________________
  //          Model can say it must be "last"
  void setCallOrder( callOrder c ){
    m_callOrder = c;
  }

  //__________________________________
  //
  callOrder getCallOrder(){
    return m_callOrder;
  }

  //__________________________________
  //    check that the order of the models is correct
  static
  bool inCorrectOrder( std::vector<Viscosity*> );

};

}

#endif /* _Viscosity_H_ */


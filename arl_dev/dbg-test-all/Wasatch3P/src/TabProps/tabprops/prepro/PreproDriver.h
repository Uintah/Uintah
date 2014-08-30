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

#ifndef PREPRO_DRIVER_h
#define PREPRO_DRIVER_h

#include <vector>
#include <string>

#include <tabprops/TabPropsConfig.h>

// forward declarations:
class ReactionModel;
class ParseGroup;
class PresumedPDFMixMdl;
class MixMdlHelper;

namespace Cantera_CXX{ class IdealGasMix; }


//====================================================================

/**
 *  @class  MixMdlParser
 *  @date   March, 2006
 *  @author James C. Sutherland
 *
 *  @brief Construct the appropriate mixing models from parsed input database.
 */
class MixMdlParser
{
public:
  MixMdlParser( const ParseGroup & );
  ~MixMdlParser();


private:
  const ParseGroup & m_parseGroup;
  PresumedPDFMixMdl * m_mixMdl;
  std::string m_convVarName;

  void set_mesh( const ParseGroup & pg,
                 MixMdlHelper & mm );
};

//====================================================================

/**
 *  @class  RxnMdlParser
 *  @date   March, 2006
 *  @author James C. Sutherland
 *
 *  @brief Construct the appropriate reaction models from parsed input database.
 */
class RxnMdlParser
{
public:
  RxnMdlParser( const ParseGroup & );
  ~RxnMdlParser();

private:
  const ParseGroup & m_parseGroup;
  ReactionModel * m_rxnMdl;
  Cantera_CXX::IdealGasMix * m_gas;

  /**
   *  Implement a simple 2-stream mixing model (unreacting, adiabatic)
   */
  void nonreact( const ParseGroup & pg );

  /**
   *  Implement the Burke-Schuman chemistry model (infinitely fast, irreversible, complete
   *  reaction).  Vary mixture fraction and (potentially) heat loss.
   */
  void fastchem( const ParseGroup & pg );

  /**
   *  Implement the equilibrium model over a range of mixture fraction and (potentially)
   *  heat loss.
   */
  void equil( const ParseGroup & pg );

  /**
   *  Import solutions from James' flamelet code.
   */
  void slfm_jcs( const ParseGroup & pg );

  void setup_cantera( const ParseGroup & pg );

  void get_comp( const ParseGroup & pg,
                 std::vector<double> & y,
                 bool & haveMassFrac,
                 Cantera_CXX::IdealGasMix & gas );

  double get_fuel_temp( const ParseGroup & pg );
  double get_oxid_temp( const ParseGroup & pg );
  double get_pressure ( const ParseGroup & pg );

  void setup_output_vars( const ParseGroup & pg );

  double set_stretch_fac( const ParseGroup & pg,
                          const double defaultFac );

};

//====================================================================

#endif // PREPRO_DRIVER_h

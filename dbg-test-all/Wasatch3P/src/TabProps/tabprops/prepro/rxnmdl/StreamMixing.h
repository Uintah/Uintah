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

#include <tabprops/prepro/rxnmdl/ReactionModel.h>

class MixtureFraction;

/**
 * \class  StreamMixing
 * \author James C. Sutherland
 * \brief  nonreacting two stream mixing
 */
class StreamMixing : public ReactionModel
{
public:

  StreamMixing( Cantera_CXX::IdealGasMix & gas,
                const std::vector<double> & y_oxid,
                const std::vector<double> & y_fuel,
                const bool haveMassFrac,
                const int order,
                const int nFpts );

  ~StreamMixing();

  void set_fuel_temperature( const double T );
  void set_oxid_temperature( const double T );
  void set_pressure( const double P );

  void implement();

private:

  Cantera_CXX::IdealGasMix & gasMix_;
  const int nFpts_;
  std::vector<double> fpts_;

  MixtureFraction * mixfr_;

  double fuelTemp_, oxidTemp_, fuelEnth_, oxidEnth_, pressure_;

  const std::vector<std::string> & indep_var_names();

};

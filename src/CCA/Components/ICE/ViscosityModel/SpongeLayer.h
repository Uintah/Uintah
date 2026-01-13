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


#ifndef ICE_VISCOSITY_SPONGELAYER_H
#define ICE_VISCOSITY_SPONGELAYER_H

#include <CCA/Components/ICE/ViscosityModel/Viscosity.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/Iterator.h>
#include <Core/Grid/Patch.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <string>

namespace Uintah {

class CellIterator;
class Box;

//______________________________________________________________________
//

class SpongeLayer: public Viscosity {
public:

  SpongeLayer(ProblemSpecP & sl_ps,
              const GridP  & grid);

  ~SpongeLayer();


  virtual void outputProblemSpec(ProblemSpecP& vModels_ps);

  //__________________________________
  //
  virtual void
  computeDynViscosity(const Patch         * patch,
                      CCVariable<double>  & temp_CC,
                      CCVariable<double>  & mu)
  {
    computeDynViscosity_impl<CCVariable<double> >( patch, temp_CC, mu);
  }

  virtual void
  computeDynViscosity(const Patch              * patch,
                      constCCVariable<double>  & temp_CC,
                      CCVariable<double>       & mu)
  {
    computeDynViscosity_impl<constCCVariable<double> >( patch, temp_CC, mu);
  }

  virtual void
  initialize (const Level * level );

protected:

  //______________________________________________________________________
  // Returns a cell iterator over Sponge layer cells on this patch
  CellIterator getCellIterator(const Patch* patch) const ;


  //______________________________________________________________________
  //  Returns the cell index nearest to the point
  IntVector findCell( const Level * level,
                      const Point & p);
  //______________________________________________________________________
  //
  std::string getExtents_string() const;

  std::string getName() const {return m_SL_name;};

  void print();

  template< class CCVar>
  void
  computeDynViscosity_impl( const Patch       * patch,
                            CCVar             & temp_CC,
                            CCVariable<double>& mu);

  IntVector m_lowIndx;                  // low index of sponge layer
  IntVector m_highIndx;                 // high index of sponge layer
  Box m_box;                            // box of the sponge layer
  double    m_maxDynViscosity;          // max dynamic viscosity
  std::string m_SL_name{"notSet"};
};
}  // end namespace Uintah
#endif

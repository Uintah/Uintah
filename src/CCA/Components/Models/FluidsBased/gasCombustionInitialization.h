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

#ifndef Packages_Uintah_CCA_Components_Models_gasCombustionInitialization_h
#define Packages_Uintah_CCA_Components_Models_gasCombustionInitialization_h

#include <CCA/Components/Models/FluidsBased/gasCombustionMechanism.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <memory>
#include <string>
#include <vector>

//______________________________________________________________________
//  Initial gas states for gasCombustion read from files.
//
//  gasCombustion::initialize owns the per-cell work every method shares:
//  skipping solid cells and geom_object regions, the closure species,
//  mixture cv and gamma, p = rho R T and the sound speed.  A method only
//  answers "what is the gas state at this point".  To add one, derive
//  from gasCombustionInit and add its ups block to
//  createGasCombustionInit.
//
//  All values are in the ups <units> system, the same as the DW.
//______________________________________________________________________

namespace Uintah {

  //__________________________________
  //  Gas state at one point
  struct gcInitState {
    double              T   {0.0};
    Vector              vel {0.0, 0.0, 0.0};
    double              rho {0.0};
    std::vector<double> Y;               // nTracked, tracked order
  };

  //__________________________________
  //  Base class
  class gasCombustionInit {
  public:
    virtual ~gasCombustionInit() = default;

    // read the ups block and the data file
    virtual void problemSetup(ProblemSpecP initPs,
                              const ReactionMech & mech) = 0;

    // gas state at a cell centre; s.Y is already sized nTracked
    virtual void sample(const Point & pt,
                        gcInitState & s) const = 0;

    // one line for the "initialization:" log entry
    virtual std::string describe() const = 0;

    // cells with gas vol_frac at or below this keep ICE's state
    double solidVolFrac() const { return m_solidVolFrac; }

  protected:
    double m_solidVolFrac {1.0e-3};
  };

  //__________________________________
  //  Returns the method named by a block inside <gasCombustion>, or
  //  nullptr if there is none.  Throws if more than one is present.
  std::unique_ptr<gasCombustionInit>
  createGasCombustionInit(const ProblemSpecP & gcPs,
                          const ReactionMech & mech);

  //______________________________________________________________________
  //  <initProfile>: 1D profile from a .dat file (e.g. Cantera, SD Toolbox).
  //  Columns: x  T  u  rho  press  then nTracked mass fractions in tracked
  //  order (mechanism order, closure species omitted).  '#' lines are
  //  ignored.  Linear interpolation onto cell centres; outside the profile
  //  the end values are used.  press is read but not used: pressure comes
  //  from the EOS.  The velocity is along <axis>.
  //
  //  Optional <perturbation> seeds cellular structure by sampling the
  //  profile at x - amplitude*sin(2*pi*xt/wavelength + phase), where xt is
  //  the coordinate along <perturbation><axis>.
  //______________________________________________________________________
  class profile1DInit : public gasCombustionInit {
  public:
    virtual void problemSetup(ProblemSpecP initPs,
                              const ReactionMech & mech);

    virtual void sample(const Point & pt,
                        gcInitState & s) const;

    virtual std::string describe() const;

  private:
    int         m_axis {0};
    std::string m_filename;

    std::vector<double>              m_x;
    std::vector<double>              m_T;
    std::vector<double>              m_u;
    std::vector<double>              m_rho;
    std::vector<double>              m_press;
    std::vector<std::vector<double>> m_Y;     // rows sized nTracked

    bool   m_perturbActive     {false};
    int    m_perturbAxis       {1};
    double m_perturbAmplitude  {0.0};
    double m_perturbWavelength {1.0};
    double m_perturbPhase      {0.0};
  };

  //______________________________________________________________________
  //  <initProfile2D>: 2D field on a plane, copied along the perpendicular
  //  axis.  <plane> is xy, xz or yz; for plane "ab" the data are indexed
  //  (a, b) and copied along the third axis n.
  //
  //  .dat   columns: a  b  T  vel_a  vel_b  rho  press  Y_1 ... Y_nTracked
  //         (tracked order).  Rows in any order; the points must form a
  //         tensor-product grid, spacing may be non-uniform.
  //  .npz   from np.savez or np.savez_compressed, float64, C order:
  //         1D arrays named after the plane axes (e.g. x, y);
  //         (na, nb) arrays T, rho, vel_<a>, vel_<b>,
  //         Y_<species> for every tracked species;
  //         optional vel_<n>, p, Y_<closure species> (not used).
  //
  //  Bilinear interpolation onto cell centres; outside the data each axis
  //  is clamped to its end values, as for the 1D profile.
  //______________________________________________________________________
  class profile2DInit : public gasCombustionInit {
  public:
    virtual void problemSetup(ProblemSpecP initPs,
                              const ReactionMech & mech);

    virtual void sample(const Point & pt,
                        gcInitState & s) const;

    virtual std::string describe() const;

  private:
    void readDat(int nTracked);
    void readNpz(const ReactionMech & mech);
    void checkData() const;

    std::string m_filename;
    std::string m_plane;
    int m_axisA {0};      // first in-plane axis
    int m_axisB {1};      // second in-plane axis
    int m_axisN {2};      // copy axis

    std::vector<double> m_a;     // na, increasing
    std::vector<double> m_b;     // nb, increasing

    // fields stored [ia*nb + ib]
    std::vector<double> m_T;
    std::vector<double> m_rho;
    std::vector<double> m_velA;
    std::vector<double> m_velB;
    std::vector<double> m_velN;              // empty if not given
    std::vector<std::vector<double>> m_Y;    // [k][ia*nb + ib]
  };

} // namespace Uintah

#endif

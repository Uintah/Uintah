/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef PUDA_H
#define PUDA_H

#include <string>

namespace Uintah {

  struct CommandLineFlags 
  {
    bool be_brief;
    bool do_AA_MMS_1;
    bool do_AA_MMS_2;
    bool do_all_ccvars;
    bool do_asci;
    bool do_cell_stresses;
    bool do_ER_MMS;
    bool do_gridstats;
    bool do_GV_MMS;
    bool do_ice_momentum;
    bool do_jacquie;
    bool do_jim1;
    bool do_jim2;
    bool do_DOP;
    bool do_listvars;
    bool do_partvar;
    bool do_PIC;
    bool do_POL;
    bool do_pressure;
    bool do_pStressHstgrm;
    bool do_timesteps;
    bool do_todd1;
    bool do_varsummary;
    bool do_verbose;
    bool use_extra_cells;

    unsigned long time_step_lower;
    unsigned long time_step_upper;
    unsigned long time_step_inc;
    bool tslow_set;
    bool tsup_set;
    int tskip;
    int matl;
    int dir;

    std::string filebase;
    std::string particleVariable;

    CommandLineFlags() {
      be_brief         = false;
      do_AA_MMS_1      = false;
      do_AA_MMS_2      = false;
      do_all_ccvars    = false;
      do_asci          = false;
      do_cell_stresses = false;
      do_ER_MMS        = false;
      do_gridstats     = false;
      do_GV_MMS        = false;
      do_ice_momentum  = false;
      do_jacquie       = false;
      do_jim1          = false;
      do_jim2          = false;
      do_DOP           = false;
      do_listvars      = false;
      do_partvar       = false;
      do_PIC           = false;
      do_POL           = false;
      do_pressure      = false;
      do_pStressHstgrm = false;
      do_timesteps     = false;
      do_todd1         = false;
      do_varsummary    = false;
      do_verbose       = false;
      use_extra_cells  = true;

      time_step_lower = 0;
      time_step_upper = 1;
      time_step_inc = 1;
      tslow_set = false;
      tsup_set = false;
      tskip = 1;
      matl  = 0;
    }
  };

} // end namespace Uintah

#endif

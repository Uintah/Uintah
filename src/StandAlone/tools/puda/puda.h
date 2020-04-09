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
    bool do_timesteps   {false};
    bool do_gridstats   {false};
    bool do_listvars    {false};
    bool do_varsummary  {false};
    bool be_brief       {false};
    bool do_pressure    {false};
    bool do_jacquie     {false};
    bool do_jim1        {false};
    bool do_jim2        {false};
    bool do_PIC         {false};
    bool do_POL         {false};
    bool do_AA_MMS_1    {false};
    bool do_AA_MMS_2    {false};
    bool do_GV_MMS      {false};
    bool do_ER_MMS      {false};
    bool do_aveParticles {false};
    bool do_partvar         {false};
    bool do_asci            {false};
    bool do_cell_stresses   {false};
    bool do_NCvar_double    {false};
    bool do_NCvar_float     {false};
    bool do_NCvar_point     {false};
    bool do_NCvar_vector    {false};
    bool do_NCvar_matrix3   {false};
    bool do_CCvar_double    {false};
    bool do_CCvar_float     {false};
    bool do_CCvar_point     {false};
    bool do_CCvar_vector    {false};
    bool do_CCvar_matrix3   {false};
    bool do_PTvar           {false};
    bool do_PTvar_all       {false};
    bool do_patch           {false};
    bool do_pStressHstgrm   {false};
    bool do_material        {false};
    bool do_verbose         {false};
    bool do_all_ccvars      {false};
    bool do_todd1           {false};
    bool do_ice_momentum    {false};

    bool use_extra_cells    {true};

    unsigned long time_step_lower  = 0;
    unsigned long time_step_upper  = 1;
    unsigned long time_step_inc    = 1;
    bool tslow_set    = false;
    bool tsup_set     = false;
    int tskip    = 1;
    int matl     = 0;
    std::string i_xd;
    std::string filebase;
    std::string particleVariable;
    std::string ccVarInput;

    CommandLineFlags() {};
  };

} // end namespace Uintah

#endif

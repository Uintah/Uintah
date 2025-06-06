/*
 * The MIT License
 *
 * Copyright (c) 2012-2025 The University of Utah
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
#ifndef test_species_density_solver_h
#define test_species_density_solver_h

#include <sci_defs/wasatch_defs.h>

#ifndef HAVE_POKITT
// kill compilation if we don't have pokitt.
#error test code for the density solver for low-Mach species transport requires PoKiTT.
#endif

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Wasatch/GraphHelperTools.h>

namespace WasatchCore{

void test_low_mach_species_density_solver( Uintah::ProblemSpecP& params,
                                           GraphCategories& gc,
                                           std::set<std::string>& persistentFields );

} // namespace WasatchCore

#endif // test_species_density_solver_h

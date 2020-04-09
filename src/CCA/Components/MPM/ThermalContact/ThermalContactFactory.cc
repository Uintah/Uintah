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

#include <CCA/Components/MPM/ThermalContact/ThermalContactFactory.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
using std::cerr;

using namespace Uintah;

ThermalContact*
ThermalContactFactory::create( const ProblemSpecP     & ps,
                                     SimulationStateP & d_sS, 
                                     MPMLabel         * lb,
                                     MPMFlags         * flag )
{
   ProblemSpecP mpm_ps = ps->findBlockWithOutAttribute( "MaterialProperties" )->findBlock( "MPM" );
   if(!mpm_ps)
     throw ProblemSetupException("Cannot find MPM material subsection.",
                                 __FILE__, __LINE__);

   ProblemSpecP thermalContact_ps = mpm_ps->findBlock("thermal_contact");
   std::string thermalContactType = "null";

   if (thermalContact_ps) {
     thermalContact_ps->getWithDefault("type", thermalContactType, "STThermalContact");
   }

   if (thermalContactType == "null")
     return (scinew NullThermalContact(thermalContact_ps, d_sS, lb, flag));

   if (thermalContactType == "STThermalContact")
     return (scinew STThermalContact(thermalContact_ps, d_sS, lb, flag));

   if (thermalContactType == "simple")
     return (scinew SimpleThermalContact(thermalContact_ps, d_sS, lb, flag));


   throw ProblemSetupException("Unknown thermal contact type ("
                               +thermalContactType+")", __FILE__, __LINE__);

}

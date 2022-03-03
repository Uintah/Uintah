/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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

#include <CCA/Components/MPM/Materials/Dissolution/CompositeDissolution.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>

using namespace std;
using namespace Uintah;

CompositeDissolution::CompositeDissolution(const ProcessorGroup* myworld,
                                           MPMLabel* Mlb)
  : Dissolution(myworld, Mlb, 0)
{
}

CompositeDissolution::~CompositeDissolution()
{
  for(list<Dissolution*>::iterator mit(d_m.begin());mit!=d_m.end();mit++)
    delete *mit;
}

void CompositeDissolution::outputProblemSpec(ProblemSpecP& ps)
{
  for (list<Dissolution*>::const_iterator it = d_m.begin();it != d_m.end();it++)
    (*it)->outputProblemSpec(ps);
}

void
CompositeDissolution::add(Dissolution * m)
{
  d_m.push_back(m);
}

void
CompositeDissolution::computeMassBurnFraction(const ProcessorGroup* pg,
                                              const PatchSubset* patches,
                                              const MaterialSubset* matls,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{
  for(list<Dissolution*>::iterator mit(d_m.begin());mit!=d_m.end();mit++) {
      (*mit)->computeMassBurnFraction(pg, patches, matls, old_dw, new_dw);
  }
}

void
CompositeDissolution::addComputesAndRequiresMassBurnFrac(SchedulerP & sched,
                                                    const PatchSet* patches,
                                                    const MaterialSet* matls) 
{
  for(list<Dissolution*>::const_iterator mit(d_m.begin());mit!=d_m.end();mit++){
      (*mit)->addComputesAndRequiresMassBurnFrac(sched, patches, matls);
  }
}

void
CompositeDissolution::setTemperature(double BHTemp)
{
  for(list<Dissolution*>::const_iterator mit(d_m.begin());mit!=d_m.end();mit++){
      (*mit)->setTemperature(BHTemp);
  }
}

void
CompositeDissolution::setPhase(std::string LCPhase)
{
  for(list<Dissolution*>::const_iterator mit(d_m.begin());mit!=d_m.end();mit++){
      (*mit)->setPhase(LCPhase);
  }
}

void
CompositeDissolution::setTimeConversionFactor(double tcf)
{
//  proc0cout << "TCF = " << tcf << endl;
  for(list<Dissolution*>::const_iterator mit(d_m.begin());mit!=d_m.end();mit++){
      (*mit)->setTimeConversionFactor(tcf);
  }
}

void CompositeDissolution::setGrowthFractionRate(const double QGVF)
{
  // Rate at which the growth vector is achieved at this load level
  for(list<Dissolution*>::const_iterator mit(d_m.begin());mit!=d_m.end();mit++){
      (*mit)->setGrowthFractionRate(QGVF);
  }
}

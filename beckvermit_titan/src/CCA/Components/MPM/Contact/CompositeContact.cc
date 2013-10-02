/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <CCA/Components/MPM/Contact/CompositeContact.h>
#include <Core/Labels/MPMLabel.h>
#include <CCA/Components/MPM/MPMFlags.h>

using namespace std;
using namespace Uintah;

CompositeContact::CompositeContact(const ProcessorGroup* myworld, MPMLabel* Mlb,MPMFlags* MFlag)
  : Contact(myworld, Mlb, MFlag, 0)
{
}

CompositeContact::~CompositeContact()
{
  for(list<Contact*>::iterator mit(d_m.begin());mit!=d_m.end();mit++)
    delete *mit;
}

void CompositeContact::outputProblemSpec(ProblemSpecP& ps)
{
  for (list<Contact*>::const_iterator it = d_m.begin(); it != d_m.end(); it++)
    (*it)->outputProblemSpec(ps);
}

void
CompositeContact::add(Contact * m)
{
  d_m.push_back(m);
}

void
CompositeContact::initFriction(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse*,
                               DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){
      NCVariable<double> frictionWork_m;
      new_dw->allocateAndPut(frictionWork_m,lb->frictionalWorkLabel, matls->get(m),
                             patch);
      frictionWork_m.initialize(0.);
    }
  }
}

void
CompositeContact::exMomInterpolated(const ProcessorGroup* pg,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  for(list<Contact*>::iterator mit(d_m.begin());mit!=d_m.end();mit++)
    {
      (*mit)->exMomInterpolated(pg, patches, matls, old_dw, new_dw);
    }
}

void
CompositeContact::exMomIntegrated(const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  for(list<Contact*>::iterator mit(d_m.begin());mit!=d_m.end();mit++)
    {
      (*mit)->exMomIntegrated(pg, patches, matls, old_dw, new_dw);
    }
}

void
CompositeContact::addComputesAndRequiresInterpolated(SchedulerP & sched,
                                                     const PatchSet* patches,
                                                     const MaterialSet* matls) 
{
  if (!flag->d_fracture) {
    Task * t = scinew Task("Contact::initFriction", 
                        this, &CompositeContact::initFriction);
    t->computes(lb->frictionalWorkLabel);
    sched->addTask(t, patches, matls);
  }
  for(list<Contact*>::const_iterator mit(d_m.begin());mit!=d_m.end();mit++)
    {
      (*mit)->addComputesAndRequiresInterpolated(sched, patches, matls);
    }
}

void
CompositeContact::addComputesAndRequiresIntegrated(SchedulerP & sched,
                                                   const PatchSet* patches,
                                                   const MaterialSet* matls) 
{
  for(list<Contact*>::const_iterator mit(d_m.begin());mit!=d_m.end();mit++)
    {
      (*mit)->addComputesAndRequiresIntegrated(sched, patches, matls);
    }
}

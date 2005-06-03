#include <Packages/Uintah/CCA/Components/MPM/Contact/CompositeContact.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

CompositeContact::CompositeContact(const ProcessorGroup* myworld, MPMLabel* Mlb,MPMFlags* MFlag)
  : Contact(myworld, Mlb, MFlag, 0)
{
}

CompositeContact::~CompositeContact()
{
  for(list<Contact*>::iterator mit(d_m.begin());mit!=d_m.end();mit++)
    delete *mit;
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
    Task * t = new Task("Contact::initFriction", 
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

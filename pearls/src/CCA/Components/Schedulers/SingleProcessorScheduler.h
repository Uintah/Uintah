/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#ifndef CCA_COMPONENTS_SCHEDULERS_SINGLEPROCESSORSCHEDULER_H
#define CCA_COMPONENTS_SCHEDULERS_SINGLEPROCESSORSCHEDULER_H

#include <CCA/Components/Schedulers/SchedulerCommon.h>
#include <CCA/Ports/DataWarehouseP.h>
 
#include <Core/Grid/Task.h>
#include <map>

namespace Uintah {

  class OnDemandDataWarehouse;
  class Task;

/**************************************

CLASS
   SingleProcessorScheduler
   

GENERAL INFORMATION

   SingleProcessorScheduler.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   Single Processor Scheduler

DESCRIPTION
   Static task ordering and deterministic execution without MPI.
   Primarily used for small runs on a single desktop or laptop.
  
  
****************************************/

class SingleProcessorScheduler : public SchedulerCommon {

  public:

    SingleProcessorScheduler( const ProcessorGroup * myworld, const Output * oport, SingleProcessorScheduler * parent = NULL );

    virtual ~SingleProcessorScheduler();

    virtual void execute(int tgnum = 0, int iteration = 0);
    
    virtual SchedulerP createSubScheduler();

  private:

    // Disable copy and assignment
    SingleProcessorScheduler( const SingleProcessorScheduler& );
    SingleProcessorScheduler& operator=(const SingleProcessorScheduler&);

    virtual void verifyChecksum();

    SingleProcessorScheduler* m_parent;
};
} // End namespace Uintah
   
#endif // End CCA_COMPONENTS_SCHEDULERS_SINGLEPROCESSORSCHEDULER_H

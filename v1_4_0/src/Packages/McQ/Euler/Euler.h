#ifndef __MCQ_EULER_H__
#define __MCQ_EULER_H__

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/Thread/Mutex.h>

namespace McQ {
using namespace SCIRun;

  class Euler: public Module {
     public:
       static Module* make(const clString& id);
       Euler(const clString& id);
       virtual void execute(void);

       void setState(GeomGroup* group, ScalarFieldHandle& pressure);

     private:
       Mutex execMutex;
       GeometryOPort* boxesOut;
       ScalarFieldOPort* pressureOut;

       GeomGroup* curBoxes;
       ScalarFieldHandle curPressure;
  };
} // End namespace McQ


#endif // __MCQ_EULER_H__

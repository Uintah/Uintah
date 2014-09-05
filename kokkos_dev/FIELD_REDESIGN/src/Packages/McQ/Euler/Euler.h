#ifndef __MCQ_EULER_H__
#define __MCQ_EULER_H__

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Datatypes/ScalarFieldRGdouble.h>
#include <SCICore/Thread/Mutex.h>

namespace McQ {
  using namespace PSECore::Dataflow;
  using namespace PSECore::Datatypes;
  using namespace SCICore::GeomSpace;
  using namespace SCICore::Thread;

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

}

#endif // __MCQ_EULER_H__


#ifndef CONTEXT_H
#define CONTEXT_H 1

namespace rtrt {

class Stats;
class Worker;
class Scene;
class PerProcessorContext;
  
struct Context {
  Stats* stats;
  Worker* worker;
  Scene* scene;
  PerProcessorContext* ppc;
  Context(Worker*, Scene* scene, Stats*);
};

} // end namespace rtrt

#endif


#ifndef CONTEXT_H
#define CONTEXT_H 1

namespace rtrt {

class Stats;
class Worker;
class Scene;
class PerProcessorContext;
  
struct Context {
  Scene* scene;
  Stats* stats;
  PerProcessorContext* ppc;
  int rendering_scene;
  int worker_num;
  Context(Scene*, Stats*, PerProcessorContext*, int rendering_scene,
          int worker_num);
};

} // end namespace rtrt

#endif

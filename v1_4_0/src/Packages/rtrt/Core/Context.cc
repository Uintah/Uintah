
#include <Packages/rtrt/Core/Context.h>

using namespace rtrt;

Context::Context(Worker* worker, Scene* scene, Stats* stats)
  : worker(worker), scene(scene), stats(stats), ppc(0)
{
}

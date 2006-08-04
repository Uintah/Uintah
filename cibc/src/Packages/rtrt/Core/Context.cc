
#include <Packages/rtrt/Core/Context.h>

using namespace rtrt;

Context::Context(Scene* scene, Stats* stats, PerProcessorContext* ppc,
                 int rendering_scene, int worker_num)
  : scene(scene), stats(stats), ppc(ppc), rendering_scene(rendering_scene),
    worker_num(worker_num)
{
}


#include <Packages/Uintah/Core/Grid/Material.h>

namespace Uintah {
  class ArchesMaterial : public Material {
  public:
    ArchesMaterial();
    virtual ~ArchesMaterial();
    virtual Burn* getBurnModel();
  };
}

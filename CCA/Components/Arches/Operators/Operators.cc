#include <CCA/Components/Arches/Operators/Operators.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Patch.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Wasatch/Operators/UpwindInterpolant.h>
#include <CCA/Components/Wasatch/Operators/FluxLimiterInterpolant.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>
#include <CCA/Components/Wasatch/Wasatch.h>

using namespace Uintah;

#define BUILD_UPWIND( VOLT )                                          \
{                                                                     \
}

#define BUILD_UPWIND_LIMITER( VOLT )                                      \
{                                                                         \
}

#define BUILD_PARTICLE_OPS( VOLT )                                                                     \
{                                                                                                       \
}

Operators&
Operators::self()
{
  static Operators s;
  return s;
}

Operators::Operators()
{}

Operators::~Operators()
{}

void
Operators::delete_patch_set()
{

  for (std::map<int, Uintah::PatchSet*>::iterator i=_patches_for_operators.begin();
      i != _patches_for_operators.end(); i++){
    delete i->second;
  }

}

template<typename FieldT>
const Point arches_get_low_position(const Uintah::Patch& patch);

template<>
const Point arches_get_low_position<SpatialOps::SVolField>(const Uintah::Patch& patch)
{
  return patch.getCellPosition(patch.getCellLowIndex());
}

template<>
const Point arches_get_low_position<SpatialOps::XVolField>(const Uintah::Patch& patch)
{
  const Uintah::Vector spacing = patch.dCell();
  Point low = patch.getCellPosition(patch.getCellLowIndex());
  low.x( low.x() - spacing[0]/2.0 );
  return low;
}

template<>
const Point arches_get_low_position<SpatialOps::YVolField>(const Uintah::Patch& patch)
{
  const Uintah::Vector spacing = patch.dCell();
  Point low = patch.getCellPosition(patch.getCellLowIndex());
  low.y( low.y() - spacing[1]/2.0 );
  return low;
}

template<>
const Point arches_get_low_position<SpatialOps::ZVolField>(const Uintah::Patch& patch)
{
  const Uintah::Vector spacing = patch.dCell();
  Point low = patch.getCellPosition(patch.getCellLowIndex());
  low.z( low.z() - spacing[2]/2.0 );
  return low;
}

void
Operators::create_patch_operators( const LevelP& level, SchedulerP& sched, const MaterialSet* matls ){

}


const Uintah::PatchSet*
Operators::get_patchset( const PatchsetSelector pss,
                       const Uintah::LevelP& level,
                       Uintah::SchedulerP& sched )
{
  return nullptr;
}

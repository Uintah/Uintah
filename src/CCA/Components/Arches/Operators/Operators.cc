#include <CCA/Components/Arches/Operators/Operators.h> 
#include <Core/Geometry/Point.h>
#include <Core/Grid/Patch.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <spatialops/structured/FVStaggered.h>
#include <spatialops/particles/ParticleFieldTypes.h>
#include <spatialops/particles/ParticleOperators.h>
#include <CCA/Components/Wasatch/Operators/UpwindInterpolant.h>
#include <CCA/Components/Wasatch/Operators/FluxLimiterInterpolant.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>
#include <CCA/Components/Wasatch/Wasatch.h>

using namespace Uintah; 

#define BUILD_UPWIND( VOLT )                                          \
{                                                                     \
  typedef UpwindInterpolant<VOLT,FaceTypes<VOLT>::XFace> OpX;         \
  typedef UpwindInterpolant<VOLT,FaceTypes<VOLT>::YFace> OpY;         \
  typedef UpwindInterpolant<VOLT,FaceTypes<VOLT>::ZFace> OpZ;         \
  pi._sodb.register_new_operator<OpX>( scinew OpX() );                \
  pi._sodb.register_new_operator<OpY>( scinew OpY() );                \
  pi._sodb.register_new_operator<OpZ>( scinew OpZ() );                \
}

#define BUILD_UPWIND_LIMITER( VOLT )                                      \
{                                                                         \
  typedef FluxLimiterInterpolant<VOLT,FaceTypes<VOLT>::XFace> OpX;        \
  typedef FluxLimiterInterpolant<VOLT,FaceTypes<VOLT>::YFace> OpY;        \
  typedef FluxLimiterInterpolant<VOLT,FaceTypes<VOLT>::ZFace> OpZ;        \
  pi._sodb.register_new_operator<OpX>( scinew OpX(dim,bcPlus,bcMinus) );  \
  pi._sodb.register_new_operator<OpY>( scinew OpY(dim,bcPlus,bcMinus) );  \
  pi._sodb.register_new_operator<OpZ>( scinew OpZ(dim,bcPlus,bcMinus) );  \
}

#define BUILD_PARTICLE_OPS( VOLT )                                                                     \
{                                                                                                       \
  typedef SpatialOps::Particle::CellToParticle<VOLT> C2P;                                               \
  typedef SpatialOps::Particle::ParticleToCell<VOLT> P2C;                                               \
  const SCIRun::Point low = arches_get_low_position<VOLT>(*patch);                                               \
  pi._sodb.register_new_operator<C2P>(scinew C2P(Dx.x(), low.x(), Dx.y(), low.y(), Dx.z(), low.z()) );\
  pi._sodb.register_new_operator<P2C>(scinew P2C(Dx.x(), low.x(), Dx.y(), low.y(), Dx.z(), low.z()) );\
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

template<typename FieldT>
const SCIRun::Point arches_get_low_position(const Uintah::Patch& patch);

template<>
const SCIRun::Point arches_get_low_position<SpatialOps::SVolField>(const Uintah::Patch& patch)
{
  return patch.getCellPosition(patch.getCellLowIndex());
}
  
template<>
const SCIRun::Point arches_get_low_position<SpatialOps::XVolField>(const Uintah::Patch& patch)
{
  const Uintah::Vector spacing = patch.dCell();
  SCIRun::Point low = patch.getCellPosition(patch.getCellLowIndex());
  low.x( low.x() - spacing[0]/2.0 );
  return low;
}

template<>
const SCIRun::Point arches_get_low_position<SpatialOps::YVolField>(const Uintah::Patch& patch)
{
  const Uintah::Vector spacing = patch.dCell();
  SCIRun::Point low = patch.getCellPosition(patch.getCellLowIndex());
  low.y( low.y() - spacing[1]/2.0 );
  return low;
}

template<>
const SCIRun::Point arches_get_low_position<SpatialOps::ZVolField>(const Uintah::Patch& patch)
{
  const Uintah::Vector spacing = patch.dCell();
  SCIRun::Point low = patch.getCellPosition(patch.getCellLowIndex());
  low.z( low.z() - spacing[2]/2.0 );
  return low;
}

void 
Operators::create_patch_operators( const LevelP& level, SchedulerP& sched, const MaterialSet* matls ){ 

  const Uintah::PatchSet* patches = get_patchset( USE_FOR_OPERATORS, level, sched );
  for( int ipss=0; ipss<patches->size(); ++ipss ){
    const Uintah::PatchSubset* pss = patches->getSubset(ipss);
    for( int ip=0; ip<pss->size(); ++ip ){

      //SpatialOps::OperatorDatabase* const opdb = scinew SpatialOps::OperatorDatabase();
      const Uintah::Patch* const patch = pss->get(ip);

      IntVector low = patch->getExtraCellLowIndex(); 
      IntVector high = patch->getExtraCellHighIndex(); 

      const SCIRun::IntVector udim = patch->getCellHighIndex() - patch->getCellLowIndex();

      std::vector<int> dim(3,1);
      for( size_t i=0; i<3; ++i ){ dim[i] = udim[i];}

      std::vector<bool> bcPlus(3,false);
      bcPlus[0] = patch->getBCType(Uintah::Patch::xplus) != Uintah::Patch::Neighbor;
      bcPlus[1] = patch->getBCType(Uintah::Patch::yplus) != Uintah::Patch::Neighbor;
      bcPlus[2] = patch->getBCType(Uintah::Patch::zplus) != Uintah::Patch::Neighbor;
      
      // check if there are any physical boundaries present on the minus side of the patch
      std::vector<bool> bcMinus(3,false);
      bcMinus[0] = patch->getBCType(Uintah::Patch::xminus) != Uintah::Patch::Neighbor;
      bcMinus[1] = patch->getBCType(Uintah::Patch::yminus) != Uintah::Patch::Neighbor;
      bcMinus[2] = patch->getBCType(Uintah::Patch::zminus) != Uintah::Patch::Neighbor;

      IntVector size = high - low; 

      Vector Dx = patch->dCell(); 
      Vector L(size[0]*Dx.x(),size[1]*Dx.y(),size[2]*Dx.z());

      Operators::PatchInfo pi;

      int pid = patch->getID(); 

      BUILD_UPWIND(SpatialOps::SVolField); 
      BUILD_UPWIND_LIMITER(SpatialOps::SVolField);
      BUILD_PARTICLE_OPS(SpatialOps::XVolField); 
      BUILD_PARTICLE_OPS(SpatialOps::YVolField); 
      BUILD_PARTICLE_OPS(SpatialOps::ZVolField); 

      SpatialOps::build_stencils( size[0], size[1], size[2],
                                  L[0],    L[1],    L[2],
                                  pi._sodb );

      this->patch_info_map.insert(std::make_pair(pid, pi)); 

    }
  }
}


const Uintah::PatchSet*
Operators::get_patchset( const PatchsetSelector pss,
                       const Uintah::LevelP& level,
                       Uintah::SchedulerP& sched )
{
  switch ( pss ) {

    case USE_FOR_TASKS:
      // return sched->getLoadBalancer()->getPerProcessorPatchSet(level);
      return level->eachPatch();
      break;

    case USE_FOR_OPERATORS: {
      const int levelID = level->getID();
      const Uintah::PatchSet* const allPatches = sched->getLoadBalancer()->getPerProcessorPatchSet(level);
      const Uintah::PatchSubset* const localPatches = allPatches->getSubset( _myworld->myrank() );

      //std::map< int, const Uintah::PatchSet* >::iterator ip = patchesForOperators_.find( levelID );

      //if( ip != patchesForOperators_.end() ) return ip->second;

      Uintah::PatchSet* patches = new Uintah::PatchSet;
      patches->addEach( localPatches->getVector() );
      //patchesForOperators_[levelID] = patches;
      return patches;
    }
  }
  return NULL;
}


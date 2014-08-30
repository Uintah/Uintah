/*
 *  \file   UintahFieldManager.h
 *  \author James C. Sutherland
 *
 * Copyright (c) 2011 The University of Utah
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

#ifndef Expr_UintahFieldManager_h
#define Expr_UintahFieldManager_h

//#define DEBUG_WRITE_FIELD_MANAGER_UPDATES
//#define DEBUG_FM_ALL

//-- standard includes --//
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cassert>
#include <iomanip>
#include <string>
#include <map>

//-- boost includes --//
#include <boost/any.hpp>
#include <boost/ref.hpp>
#include <boost/foreach.hpp>
#ifdef ENABLE_THREADS
#include <boost/thread/mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#endif

//-- expression library includes --//
#include <expression/ManagerTypes.h>
#include <expression/FieldManagerBase.h>
#include <expression/Tag.h>

//-- SpatialOps includes --//
#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/structured/MemoryTypes.h>
#include <spatialops/structured/ExternalAllocators.h>
#include <spatialops/Nebo.h>

//-- Uintah Includes --//
#include <Core/Grid/Variables/PerPatch.h>      /* single, per-patch value */
#include <Core/Grid/Variables/VarLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Parallel/ProcessorGroup.h>
# ifdef ENABLE_CUDA
#include <Core/Grid/Variables/GPUGridVariableBase.h>  /* GPU grid Variables */
#include <Core/Grid/Variables/GPUPerPatchBase.h>      /* GPU perpatch Variables */
#endif

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/FieldAdaptor.h>

namespace Expr {

//==================================================================

  /**
   * \enum FieldMode
   * \brief Enumerates the modes that a Uintah field has.
   */
  enum FieldMode
  {
    COMPUTES,  //!< COMPUTES
    MODIFIES,  //!< MODIFIES
    REQUIRES   //!< REQUIRES
  };

  template<typename T>
  inline T& operator<<( T& os, const FieldMode mode ){
    switch ( mode ) {
      case COMPUTES: os << "COMPUTES"; break;
      case MODIFIES: os << "MODIFIES"; break;
      case REQUIRES: os << "REQUIRES"; break;
    }
    return os;
  }

  /**
   * \struct UintahFieldAllocInfo
   * \brief Provides information required to obtain a Uintah field from the data warehouse.
   */
  struct UintahFieldAllocInfo
  {
    const Uintah::VarLabel* varlabel;  ///< The Uintah description of the variable
    bool useOldDataWarehouse;          ///< use old or new data warehouse?
    FieldMode mode;                    ///< What mode do we interact with the data warehouse in? Can be changed later on.
    int nghost;                        ///< How many ghost cells should this field use?
    Uintah::Ghost::GhostType ghostType;
    UintahFieldAllocInfo( const Uintah::VarLabel* vl,
                          const FieldMode m,
                          const int ng,
                          const Uintah::Ghost::GhostType gt,
                          const bool olddw = false )
    : varlabel( vl ),
      mode( m ),
      nghost( ng ),
      ghostType( gt )
    {
      useOldDataWarehouse = olddw;
    }
    UintahFieldAllocInfo()
    {
      varlabel = NULL;
      mode = REQUIRES;
      nghost = 0;
    }
  };

  typedef std::map<Tag,UintahFieldAllocInfo*> IDUintahInfoMap;

//====================================================================

  /** @struct SpatialFieldAllocator
   *  @author Devin Robison
   *
   *  Provides allocate_field method which returns a spatial field given a FieldInfo object.
   *  Note: This lets us avoid problems with partial class template specializations.
   *
   */
  template< typename FieldT >
  struct SpatialFieldAllocator
  {
    static SpatialOps::SpatFldPtr<FieldT>
    allocate_field( const Wasatch::AllocInfo* info,
                    const short int deviceIndex = CPU_INDEX )
    {
      using namespace SpatialOps;
      IntVec bcMinus, bcPlus;
      Wasatch::get_bc_logicals( info->patch, bcMinus, bcPlus );
      return SpatialFieldStore::get_from_window<FieldT>(
          Wasatch::get_memory_window_for_uintah_field<FieldT>( info->patch ),
          BoundaryCellInfo::build<FieldT>( bcPlus ),
          GhostData( Wasatch::get_n_ghost<FieldT>() ),
          deviceIndex );
    }
  };

  /** @struct SpatialFieldAllocator
   *  @author Tony Saad
   *  @date   June 23, 2014
   *  @brief Allocator specialization for particle fields.
   */
  template<>
  struct SpatialFieldAllocator<SpatialOps::Particle::ParticleField> {
    static SpatialOps::SpatFldPtr<SpatialOps::Particle::ParticleField>
    allocate_field( const Wasatch::AllocInfo* ainfo,
                   const short int deviceIndex = CPU_INDEX )
    {
      using namespace SpatialOps;

      Uintah::ParticleSubset* const pset = ainfo->newDW->haveParticleSubset(ainfo->materialIndex, ainfo->patch) ?
          ainfo->newDW->getParticleSubset(ainfo->materialIndex, ainfo->patch) :
          ainfo->oldDW->getParticleSubset(ainfo->materialIndex, ainfo->patch);
      
      const int nParticles = pset->numParticles();
      return SpatialFieldStore::get_from_window<Particle::ParticleField>( MemoryWindow( IntVec(nParticles,1,1) ),
                                                                          BoundaryCellInfo::build<Particle::ParticleField>(false,false,false),
                                                                          GhostData(0),
                                                                          deviceIndex );
    }
  };

  /** @struct UintahFieldContainer
   *  @author Devin Robison
   *
   *  Provides a way to maintain field references to Uintah fields.
   *  note: Uintah field types end up being ref counted pointers that we cannot release until
   *  the field is no longer in use. To avoid specializing the field structures, we specialize
   *  the container
   */
  template< typename FieldT >
  struct UintahFieldContainer
  {
  public:
    typedef typename Wasatch::SelectUintahFieldType<FieldT>::type       UFT;
    typedef typename Wasatch::SelectUintahFieldType<FieldT>::const_type ConstUFT;

    UFT& get(){ return uintahFieldPointer_; }
    ConstUFT& getc(){ return c_uintahFieldPointer_; }
  private:
    UFT uintahFieldPointer_;
    ConstUFT c_uintahFieldPointer_;
  };

  template< >
  struct UintahFieldContainer<SpatialOps::SingleValueField>
  {
  public:
    typedef Uintah::PerPatch<double*> UFT;
    UFT& get(){ return uintahFieldPointer_; }
  private:
    UFT uintahFieldPointer_;
  };

  /** @struct UintahFieldAllocator
   *  @author Devin Robison
   *
   *  Provides allocate_field method which returns a spatial field given FieldInfo and AllocInfo objects.
   *  Note: This lets us avoid problems with partial class template specializations.
   */
  template< typename FieldT >
  struct UintahFieldAllocator
  {
    static
    SpatialOps::SpatFldPtr<FieldT>
    allocate_field( UintahFieldContainer<FieldT>& uintahFieldContainer,
                    const Wasatch::AllocInfo* const ainfo,
                    const UintahFieldAllocInfo& finfo,
                    const short int deviceIndex = CPU_INDEX )
    {
#     ifdef DEBUG_FM_ALL
      std::cout << " -> allocating " << finfo.varlabel->getName() << " as a uintah field" <<
                   " on device Index : " << deviceIndex << std::endl;
#     endif

      // select which datawarehouse to extract this variable from
      Uintah::DataWarehouse* const dw = finfo.useOldDataWarehouse ? ainfo->oldDW : ainfo->newDW;
      const bool isGPUTask = ainfo->isGPUTask;

#     ifndef NDEBUG
      if( dw == NULL ){
        std::ostringstream msg;
        msg << "ERROR: data warehouse is NULL for variable named "
            << finfo.varlabel->getName() << "'\n\t" << std::endl
            << __FILE__ << " : " << __LINE__ << std::endl;
        throw std::runtime_error( msg.str() );
      }
      if(!IS_VALID_INDEX(deviceIndex)){
        std::ostringstream msg;
        msg << "ERROR: Invalid deviceIndex passed to UintahFieldAllocator::allocate_field() " << std::endl
            << "for variable named " << finfo.varlabel->getName() << " with device : " << deviceIndex << "'\n\t" << std::endl
            << __FILE__ << " : " << __LINE__ << std::endl;
        throw std::runtime_error( msg.str() );
      }
#     endif

      // set field mode
      switch( finfo.mode ){
        case COMPUTES:{
#         ifdef DEBUG_FM_ALL
          std::cout << "COMPUTES (allocate & put) : " << finfo.varlabel->getName() << std::endl;
#         endif
          typedef typename UintahFieldContainer<FieldT>::UFT UFT;
          UFT& uintahFieldVar = uintahFieldContainer.get();
          double* uintahDeviceFieldVar = NULL;  // Device Variable

#         ifdef ENABLE_CUDA
          // homogeneous case
          if( isGPUTask && IS_GPU_INDEX(deviceIndex) ) {
            const char* label = finfo.varlabel->getName().c_str();
	    dw->allocateTemporary( uintahFieldVar,
				   ainfo->patch,
			       	   finfo.ghostType,
				   finfo.nghost );
            Uintah::GPUGridVariable<double> myDeviceVar;
            dw->getGPUDW()->getModifiable( myDeviceVar,
                                           label,
                                           ainfo->patch->getID(),
                                           ainfo->materialIndex );
            uintahDeviceFieldVar = const_cast<double*>( myDeviceVar.getPointer() );
#         ifdef DEBUG_FM_ALL
            std::cout << "Uintah Device variable (" << finfo.varlabel->getName() << ") at address : " << uintahDeviceFieldVar <<
                ", with size : "<< myDeviceVar.getMemSize() << std::endl;
#         endif
          }
          else{ // heterogeneous case
	    dw->allocateAndPut( uintahFieldVar,
				finfo.varlabel,
				ainfo->materialIndex,
				ainfo->patch,
				finfo.ghostType,
				finfo.nghost );
	  }
#         else
          dw->allocateAndPut( uintahFieldVar,
                              finfo.varlabel,
                              ainfo->materialIndex,
                              ainfo->patch,
                              finfo.ghostType,
                              finfo.nghost );
#         endif // ENABLE_CUDA
          return Wasatch::wrap_uintah_field_as_spatialops<FieldT,UFT>( uintahFieldVar,
                                                                       *ainfo,
                                                                       SpatialOps::GhostData(finfo.nghost),
                                                                       deviceIndex,
                                                                       uintahDeviceFieldVar,
                                                                       isGPUTask );
        }

        case REQUIRES:{
#         ifdef DEBUG_FM_ALL
	  std::cout << "REQUIRES (get) : " << finfo.varlabel->getName() << std::endl;
#         endif
          typedef typename UintahFieldContainer<FieldT>::ConstUFT ConstUFT;
          ConstUFT& uintahFieldVar = uintahFieldContainer.getc();
          double* uintahDeviceFieldVar = NULL;  // Device Variable
          dw->get( uintahFieldVar,
                   finfo.varlabel,
                   ainfo->materialIndex,
                   ainfo->patch,
                   finfo.ghostType,
                   finfo.nghost );
#         ifdef ENABLE_CUDA
            const char* label = finfo.varlabel->getName().c_str();

//            // Identify the varibles that have been tagged as CPU only
//            // because of cleaving but has been moved to GPUDataWarehouse.
//            if( mtype == SpatialOps::LOCAL_RAM && dw->getGPUDW()->exist( label, ainfo->patch->getID(), ainfo->materialIndex ) ) {
//              mtype = SpatialOps::EXTERNAL_CUDA_GPU;
//            }

            // homogeneous task
            if( isGPUTask && IS_GPU_INDEX(deviceIndex) ){
              Uintah::GPUGridVariable<double> myDeviceVar;
              dw->getGPUDW()->get( myDeviceVar,
                                   label,
                                   ainfo->patch->getID(),
                                   ainfo->materialIndex );
              uintahDeviceFieldVar = const_cast<double*>( myDeviceVar.getPointer() );
#         ifdef DEBUG_FM_ALL
            std::cout << "Uintah Device variable (" << finfo.varlabel->getName() << ") at address : " << uintahDeviceFieldVar <<
                ", with size : "<< myDeviceVar.getMemSize() << std::endl;
#         endif
            }
#         endif
          return Wasatch::wrap_uintah_field_as_spatialops<FieldT,ConstUFT>( uintahFieldVar,
                                                                            *ainfo,
                                                                            SpatialOps::GhostData(finfo.nghost),
                                                                            deviceIndex,
                                                                            uintahDeviceFieldVar,
                                                                            isGPUTask );
        }

        case MODIFIES:{
#         ifdef DEBUG_FM_ALL
          std::cout << "MODIFIES (get modifiable) : " << finfo.varlabel->getName() << std::endl;
#         endif
          typedef typename UintahFieldContainer<FieldT>::UFT UFT;
          UFT& uintahFieldVar = uintahFieldContainer.get();
          double* uintahDeviceFieldVar = NULL;  // Device Variable
          dw->getModifiable( uintahFieldVar,
                             finfo.varlabel,
                             ainfo->materialIndex,
                             ainfo->patch,
                             finfo.ghostType,
                             finfo.nghost );
#         ifdef ENABLE_CUDA
          // homogeneous task
          if( isGPUTask && IS_GPU_INDEX(deviceIndex) ){
            Uintah::GPUGridVariable<double> myDeviceVar;
            const char* label = finfo.varlabel->getName().c_str();
              dw->getGPUDW()->getModifiable( myDeviceVar,
                                             label,
                                             ainfo->patch->getID(),
                                             ainfo->materialIndex );
            uintahDeviceFieldVar = const_cast<double*>( myDeviceVar.getPointer() );
#         ifdef DEBUG_FM_ALL
            std::cout << "Uintah Device variable (" << finfo.varlabel->getName() << ") at address : " << uintahDeviceFieldVar <<
                ", with size : "<< myDeviceVar.getMemSize() << std::endl;
#         endif
          }
#         endif
          return Wasatch::wrap_uintah_field_as_spatialops<FieldT,UFT>( uintahFieldVar,
                                                                       *ainfo,
                                                                       SpatialOps::GhostData(finfo.nghost),
                                                                       deviceIndex,
                                                                       uintahDeviceFieldVar,
                                                                       isGPUTask );
        }

        default:{
          std::ostringstream msg;
          msg << "ERROR: Invalid uintah field state request, legal values: ( MODIFIES, COMPUTES, REQUIRES )\n\t"
              << __FILE__ << " : " << __LINE__ << std::endl;
          throw std::runtime_error( msg.str() );
        }
      }  // end switch
    }  // end allocate_field
  };

  template< >
  struct UintahFieldAllocator<SpatialOps::Particle::ParticleField>
  {
    static
    SpatialOps::SpatFldPtr<SpatialOps::Particle::ParticleField>
    allocate_field( UintahFieldContainer<SpatialOps::Particle::ParticleField>& uintahFieldContainer,
                    const Wasatch::AllocInfo* ainfo,
                    const UintahFieldAllocInfo& finfo,
                    short int deviceIndex = CPU_INDEX )
    {
      namespace SP = SpatialOps::Particle;
      typedef SP::ParticleField FieldT;
#     ifdef DEBUG_FM_ALL
      std::cout << " -> allocating " << finfo.varlabel->getName() << " as a uintah Particle field" <<
                   " on device Index : " << deviceIndex << std::endl;
#     endif

      // select which datawarehouse to extract this variable from
      Uintah::DataWarehouse* const dw = finfo.useOldDataWarehouse ? ainfo->oldDW : ainfo->newDW;
      const bool isGPUTask = ainfo->isGPUTask;

      Uintah::ParticleSubset* const pset = ainfo->newDW->haveParticleSubset(ainfo->materialIndex, ainfo->patch) ?
      ainfo->newDW->getParticleSubset(ainfo->materialIndex, ainfo->patch) :
      ainfo->oldDW->getParticleSubset(ainfo->materialIndex, ainfo->patch);

#     ifndef NDEBUG
      if( dw == NULL ){
        std::ostringstream msg;
        msg << "ERROR: data warehouse is NULL for variable named "
            << finfo.varlabel->getName() << "'\n\t" << std::endl
            << __FILE__ << " : " << __LINE__ << std::endl;
        throw std::runtime_error( msg.str() );
      }
      if(!IS_VALID_INDEX(deviceIndex)){
        std::ostringstream msg;
        msg << "ERROR: Invalid deviceIndex passed to UintahFieldAllocator::allocate_field() " << std::endl
            << "for variable named " << finfo.varlabel->getName() << " with device : " << deviceIndex << "'\n\t" << std::endl
            << __FILE__ << " : " << __LINE__ << std::endl;
        throw std::runtime_error( msg.str() );
      }
#     endif

      // set field mode
      switch( finfo.mode ){
        case COMPUTES:{
#         ifdef DEBUG_FM_ALL
          std::cout << "COMPUTES particle (allocate & put) : " << finfo.varlabel->getName() << std::endl;
#         endif
          typedef UintahFieldContainer<SP::ParticleField>::UFT UFT;
          UFT& uintahFieldVar = uintahFieldContainer.get();
          double* uintahDeviceFieldVar = NULL;  // Device Variable

#         ifdef ENABLE_CUDA
          // homogeneous task
          if( isGPUTask && IS_GPU_INDEX(deviceIndex) ){
            const char* label = finfo.varlabel->getName().c_str();
            dw->allocateTemporary( uintahFieldVar,
                                   pset );
            Uintah::GPUGridVariable<double> myDeviceVar;
            dw->getGPUDW()->getModifiable( myDeviceVar,
                                           label,
                                           ainfo->patch->getID(),
                                           ainfo->materialIndex );
            uintahDeviceFieldVar = const_cast<double*>( myDeviceVar.getPointer() );
#           ifdef DEBUG_FM_ALL
            std::cout << "Uintah Device variable (" << finfo.varlabel->getName() << ") at address : " << uintahDeviceFieldVar <<
                ", with size : "<< myDeviceVar.getMemSize() << std::endl;
#           endif
          }
          else{ // heterogeneous case
            dw->allocateAndPut( uintahFieldVar,
                                finfo.varlabel,
                                pset );
          }
#         else
          dw->allocateAndPut( uintahFieldVar,
                              finfo.varlabel,
                              pset );
#         endif // ENABLE_CUDA
          return Wasatch::wrap_uintah_field_as_spatialops<FieldT,UFT>( uintahFieldVar,
                                                                       *ainfo,
                                                                       SpatialOps::GhostData(finfo.nghost),
                                                                       deviceIndex,
                                                                       uintahDeviceFieldVar,
                                                                       isGPUTask );
        }

        case REQUIRES:{
#         ifdef DEBUG_FM_ALL
          std::cout << "REQUIRES particle (get) : " << finfo.varlabel->getName() << std::endl;
#         endif
          typedef UintahFieldContainer<FieldT>::ConstUFT ConstUFT;
          ConstUFT& uintahFieldVar = uintahFieldContainer.getc();
          double* uintahDeviceFieldVar = NULL;  // Device Variable
          dw->get( uintahFieldVar,
                   finfo.varlabel,
                   pset );
#         ifdef ENABLE_CUDA
            const char* label = finfo.varlabel->getName().c_str();

//            // Identify the varibles that have been tagged as CPU only
//            // because of cleaving but has been moved to GPUDataWarehouse.
//            if( mtype == SpatialOps::LOCAL_RAM && dw->getGPUDW()->exist( label, ainfo->patch->getID(), ainfo->materialIndex ) ) {
//              mtype = SpatialOps::EXTERNAL_CUDA_GPU;
//            }

            // homogeneous task
            if( isGPUTask && IS_GPU_INDEX(deviceIndex) ) {
              Uintah::GPUGridVariable<double> myDeviceVar;
              dw->getGPUDW()->get( myDeviceVar,
                                   label,
                                   ainfo->patch->getID(),
                                   ainfo->materialIndex );
              uintahDeviceFieldVar = const_cast<double*>( myDeviceVar.getPointer() );
#             ifdef DEBUG_FM_ALL
              std::cout << "Uintah Device variable (" << finfo.varlabel->getName() << ") at address : " << uintahDeviceFieldVar <<
                  ", with size : "<< myDeviceVar.getMemSize() << std::endl;
#             endif
            }
#         endif
          return Wasatch::wrap_uintah_field_as_spatialops<FieldT,ConstUFT>( uintahFieldVar,
                                                                            *ainfo,
                                                                            SpatialOps::GhostData(finfo.nghost),
                                                                            deviceIndex,
                                                                            uintahDeviceFieldVar,
                                                                            isGPUTask );
        }

        case MODIFIES:{
#         ifdef DEBUG_FM_ALL
          std::cout << "MODIFIES particle (get modifiable) : " << finfo.varlabel->getName() << std::endl;
#         endif
          typedef UintahFieldContainer<FieldT>::UFT UFT;
          UFT& uintahFieldVar = uintahFieldContainer.get();
          double* uintahDeviceFieldVar = NULL;  // Device Variable
          dw->getModifiable( uintahFieldVar,
                             finfo.varlabel,
                             pset );
#         ifdef ENABLE_CUDA
          // homogeneous task
          if( isGPUTask && IS_GPU_INDEX(deviceIndex) ){
            Uintah::GPUGridVariable<double> myDeviceVar;
            const char* label = finfo.varlabel->getName().c_str();
              dw->getGPUDW()->getModifiable( myDeviceVar,
                                             label,
                                             ainfo->patch->getID(),
                                             ainfo->materialIndex );
            uintahDeviceFieldVar = const_cast<double*>( myDeviceVar.getPointer() );
#         ifdef DEBUG_FM_ALL
            std::cout << "Uintah Device variable (" << finfo.varlabel->getName() << ") at address : " << uintahDeviceFieldVar <<
                ", with size : "<< myDeviceVar.getMemSize() << std::endl;
#         endif
          }
#         endif
          return Wasatch::wrap_uintah_field_as_spatialops<FieldT,UFT>( uintahFieldVar,
                                                                       *ainfo,
                                                                       SpatialOps::GhostData(finfo.nghost),
                                                                       deviceIndex,
                                                                       uintahDeviceFieldVar,
                                                                       isGPUTask );
        }

        default:{
          std::ostringstream msg;
          msg << "ERROR: Invalid uintah field state request, legal values: ( MODIFIES, COMPUTES, REQUIRES )\n\t"
              << __FILE__ << " : " << __LINE__ << std::endl;
          throw std::runtime_error( msg.str() );
        }
      }  // end switch
    } // end allocate_field
    
  };

  template< >
  struct UintahFieldAllocator<SpatialOps::SingleValueField>
  {
    static
    SpatialOps::SpatFldPtr<SpatialOps::SingleValueField>
    allocate_field( UintahFieldContainer<SpatialOps::SingleValueField>& uintahFieldContainer,
                    const Wasatch::AllocInfo* ainfo,
                    const UintahFieldAllocInfo& finfo,
                    short int deviceIndex = CPU_INDEX )
    {
      // jcs need to get const in the requires case, right?
      using namespace SpatialOps;

#     ifdef DEBUG_FM_ALL
      std::cout << " -> allocating " << finfo.varlabel->getName() << " as a uintah SVF field" <<
                   " on device Index : " << deviceIndex << std::endl;
#     endif

#     ifdef ENABLE_CUDA
      Uintah::GPUPerPatch<double> myDeviceVar;
#     endif

      Uintah::DataWarehouse* const dw = finfo.useOldDataWarehouse ? ainfo->oldDW : ainfo->newDW;
      const bool isGPUTask = ainfo->isGPUTask;
      Uintah::PerPatch<double*>& uintahFieldVar = uintahFieldContainer.get();

#     ifndef NDEBUG
      if( dw == NULL ){
        std::ostringstream msg;
        msg << "ERROR: data warehouse is NULL for variable named "
            << finfo.varlabel->getName() << "'\n\t" << std::endl
            << __FILE__ << " : " << __LINE__ << std::endl;
        throw std::runtime_error( msg.str() );
      }
      if(!IS_VALID_INDEX(deviceIndex)){
        std::ostringstream msg;
        msg << "ERROR: Invalid deviceIndex passed to UintahFieldAllocator::allocate_field() " << std::endl
            << "for variable named " << finfo.varlabel->getName() << " with device : " << deviceIndex << "'\n\t" << std::endl
            << __FILE__ << " : " << __LINE__ << std::endl;
        throw std::runtime_error( msg.str() );
      }
#     endif

      switch( finfo.mode ){
        case COMPUTES:{
#         ifdef DEBUG_FM_ALL
          std::cout << "COMPUTES SVF (allocate & put) : " << finfo.varlabel->getName() << std::endl;
#         endif
          uintahFieldVar.setData( new double );
          dw->put( uintahFieldVar, finfo.varlabel, ainfo->materialIndex, ainfo->patch );
#         ifdef ENABLE_CUDA
          if( isGPUTask && IS_GPU_INDEX(deviceIndex) ){
            const char* label = finfo.varlabel->getName().c_str();
            dw->getGPUDW()->getModifiable( myDeviceVar, label, ainfo->patch->getID(), ainfo->materialIndex );
          }
#         endif
          break;
        }
        case REQUIRES:
#         ifdef DEBUG_FM_ALL
          std::cout << "REQUIRES SVF (allocate & put) : " << finfo.varlabel->getName() << std::endl;
#         endif

        case MODIFIES:{
          // jcs note that it appears that Uintah does not allow getModifiable() for per-patch
          // variables.  That means that this is a bit sketchy.  But it seems to work so far.
#         ifdef DEBUG_FM_ALL
          std::cout << "MODIFIES SVF (get modifiable) : " << finfo.varlabel->getName() << std::endl;
#         endif
          dw->get( uintahFieldVar, finfo.varlabel, ainfo->materialIndex, ainfo->patch );

#         ifdef ENABLE_CUDA
          // homogeneous case
          if( isGPUTask && IS_GPU_INDEX(deviceIndex) ){
            const char* label = finfo.varlabel->getName().c_str();

//            if( dw->getGPUDW()->exist( label, ainfo->patch->getID(), ainfo->materialIndex ) ) {
//              mtype = SpatialOps::EXTERNAL_CUDA_GPU;
//            }

            dw->getGPUDW()->get( myDeviceVar, label, ainfo->patch->getID(),  ainfo->materialIndex );
          }
#         endif
          break;
        }
        default:{
          std::ostringstream msg;
          msg << "ERROR: Invalid uintah field state request. Legal values: ( MODIFIES, COMPUTES, REQUIRES )\n\t"
              << __FILE__ << " : " << __LINE__ << std::endl;
          throw std::runtime_error( msg.str() );
        }
      }

      // field values based on heterogenous task or homogeneous task
      double* fieldValues_ = NULL;

      if( isGPUTask ){
        // homogeneous tasks
#       ifdef ENABLE_CUDA
#       ifndef NDEBUG
        if(!IS_GPU_INDEX(deviceIndex)){
          std::ostringstream msg;
          msg << " Error : deviceIndex passed : " << deviceIndex << " is not compatible for homogeneous GPU tasks. \n\t"
              << __FILE__ << " : " << __LINE__ << std::endl;
          throw std::runtime_error( msg.str() );
        }
#       endif
        fieldValues_ = const_cast<double*>( myDeviceVar.getPointer() );
#       endif
      }else{
        // heterogeneous case
        deviceIndex = CPU_INDEX;
        fieldValues_ = const_cast<SingleValueField::value_type*>( uintahFieldVar.get() );
      }

      return SpatialOps::SpatFldPtr<SingleValueField>(
          new SingleValueField( MemoryWindow( IntVec(1,1,1), IntVec(0,0,0), IntVec(1,1,1) ),
                                BoundaryCellInfo::build<SingleValueField>(false,false,false),  // bc doesn't matter for single value fields
                                GhostData( Wasatch::get_n_ghost<SpatialOps::SingleValueField>() ),
                                fieldValues_,
                                ExternalStorage,
                                deviceIndex ) );
    }  // End allocate_field

  };  // template <> struct UintahFieldAllocator<SingleValueField>

  //=================================================================

  /**
   * \class UintahFieldManager
   * \brief Provides field managment when interfacing ExprLib with Uintah.
   *
   * When interfacing with Uintah, some fields may be managed by Uintah while others
   * may be managed locally.  The UintahFieldManager provides the ability to work
   * with all fields, wherever they are managed, and manages either allocation &
   * deallocation or interfacing to Uintah's data warehouse to obtain and release
   * fields.
   *
   * In this context:
   *  - the \c register_field method adds a request for a field with the given Tag to be available.
   *  - the \c allocate_fields method either fetches memory from Uintah and
   *    wraps it as SpatialField objects or internally allocates it from a
   *    memory pool.
   *  - the \c field_ref method obtains a field that has previously been
   *    registered and allocated
   *  - the \c deallocate_fields method releases all fields that have been allocated.
   */
  template< typename FieldT >
  class UintahFieldManager : public FieldManagerBase
  {
  public:
    UintahFieldManager()
      : FieldManagerBase()
    {
      properties_["UintahInfo"] = boost::ref( idUintahMap_ );
      fieldsAreAllocated_ = 0;
    }

    ~UintahFieldManager();

    FieldID register_field( const Tag& );

    /**
     *  \brief Retrieve the field with the specified Tag.
     *  \param tag the Tag for the field of interest
     */
    const FieldT& field_ref( const Tag& tag ) const;

    /**
     *  \brief Retrieve the field with the specified Tag.
     *  \param tag the Tag for the field of interest
     *
     *  Note: there is not a version like this:
     *  \code
     *  FieldT& field_ref( const Tag& tag ) const;
     *  \endcode
     *  because we use the const method to control the type of access to fields.
     *  Specifically, if you have a const UintahFieldManager, then you can only
     *  obtain const FieldT from it.  This allows us to control write-access to
     *  fields more carefully.
     */
    FieldT& field_ref( const Tag& tag );

    /**
     * \brief For scratch fields, this releases their memory back to the pool.
     *        It has no effect on Uintah fields or on persistent/locked fields.
     */
    inline bool release_field( const Tag& );

    /**
     * \brief Used to lock non-persistent fields, blocking memory from being freed automatically.
     *
     * Note that this only has an effect on locally managed fields.  Fields
     * managed externally (e.g., by Uintah) will not be freed automatically
     * even if they are unlocked.
     *
     * \sa \ref unlock_field
     */
    inline bool lock_field( const Tag& tag );

    /**
     * \brief Used to unlock a field, allowing memory to be freed automatically
     *        if the field has not been marked as persistent.
     *
     * Note that this only has an effect on locally managed fields.  Fields
     * managed externally (e.g., by Uintah) will not be freed automatically
     * even if they are unlocked.
     *
     * \sa \ref lock_field
     */
    inline bool unlock_field( const Tag& tag );

    /**
     * \brief Creates a copy of the field on the target device.
     *
     * \param tag -- Field identifier, used to locate the proper FieldStruct
     * \param deviceIndex -- Index of the device type that will consume this field
     *
     * Note that this performs an asynchronous copy of the field.  To ensure
     * that the transfer has completed, you should call \ref validate_field_location
     */
    inline void
    prep_field_for_consumption( const Tag& tag,
                                const short int deviceIndex );

    /**
     * \brief Ensure that the copy of the field is available on the target device.
     *
     * \param tag -- Field identifier, used to locate the proper FieldStruct
     * \param deviceIndex -- Location of the device type that will consume this field
     */
    inline void
    validate_field_location( const Tag& tag,
                             const short int deviceIndex );

    /**
     * \brief Set the given field Location to be active.  Any subsequent Nebo
     * calls on that field will be performed on the specified field location.
     *
     * \param tag -- Field identifier, used to locate the proper FieldStruct
     * \param deviceIndex -- Active field Location to be set
     */
    inline void
    set_active_field_location( const Tag& tag,
                               const short int deviceIndex );

    /**
     * \brief Changes the field's memory manager.
     *
     * \param tag -- Field identifier, used to locate the proper FieldStruct
     * \param m -- Type indicator for the device that will consume this field
     * \param deviceIndex -- Index of the device type that will consume this field
     */
    inline void
    set_field_memory_manager( const Tag&,
                              const MemoryManager m,
                              const short int deviceIndex = CPU_INDEX );

    /**
     *  \brief allocate fields (resolve memory from Uintah or build it locally, as appropriate)
     *  \param info an AllocInfo object
     */
    inline void allocate_fields( const boost::any& info );

    /**
     * \brief releases all bound memory back to either Uintah or to the memory
     *        pool, as appropriate.
     */
    inline void deallocate_fields();

    /**
     * \brief query the existence of a field with the supplied Tag
     * \param tag the field of interest
     * \return true if the field has been registered.
     */
    inline bool has_field( const Tag& tag ) const;

    /**
     * \brief output information about what fields have been registered to the supplied stream
     * \param os the stream to dump information on
     */
    inline void dump_fields( std::ostream& os ) const;

    // this should only be called by the Expression base class for CARRY_FORWARD fields
    void copy_field_forward( const Tag& tag, FieldT& f ) const;

  private:

    struct FieldInfo
    {
      bool isAllocated;         ///< true once the field has been allocated
      bool isLocked;            ///< true if the field should be persistent
      short int deviceIndex;    ///< the device index where the field lives (CPU_INDEX, GPU_INDEX, etc)
      MemoryManager memoryMgr;  ///< the memory management strategy for this field
      UintahFieldAllocInfo uintahInfo;  ///< information required for interaction with Uintah fields
      UintahFieldContainer<FieldT>* uintahFieldContainer; ///< the Uintah version of this field
      SpatialOps::SpatFldPtr<FieldT> fieldPtr;            ///< the SpatialField version of this field

      FieldInfo()
      {
        isAllocated = false;
        isLocked = false;
        deviceIndex = CPU_INDEX;
        memoryMgr = MEM_EXTERNAL;
        uintahFieldContainer = NULL;
      }

      FieldInfo( const FieldInfo& a )
      {
        *this = a;
      }

      FieldInfo&
      operator=( const FieldInfo& a )
      {
        isAllocated = a.isAllocated;
        isLocked    = a.isLocked;
        deviceIndex = a.deviceIndex;
        memoryMgr   = a.memoryMgr;
        uintahInfo  = a.uintahInfo;
        uintahFieldContainer = a.uintahFieldContainer;
        if( isAllocated ){
          if( !fieldPtr.isnull() ) fieldPtr = a.fieldPtr;
        }
        else{
          assert( !isLocked );
          fieldPtr.detach();
        }
        return *this;
      }
    };

    typedef std::map<Tag,FieldInfo> IDFieldInfoMap;

    int fieldsAreAllocated_;
    IDFieldInfoMap idFieldInfoMap_;
    IDUintahInfoMap idUintahMap_;
    const Wasatch::AllocInfo* allocInfo_;
    bool isHomogeneousgpu_;

    UintahFieldManager( const UintahFieldManager& );  // no copying
    UintahFieldManager&
    operator=( const UintahFieldManager& );  // no assignment

    /**
     *  \class ExecMutex
     *  \brief Scoped lock.
     */
    class ExecMutex
    {
#   ifdef ENABLE_THREADS
      const boost::mutex::scoped_lock lock;
      inline boost::mutex& get_mutex() const {static boost::mutex m; return m;}
    public:
      ExecMutex() : lock( get_mutex() ) {}
      ~ExecMutex() {}
#   else
    public:
      ExecMutex(){}
      ~ExecMutex(){}
#   endif
    };

  };

//------------------------------------------------------------------

  template< typename FieldT >
  bool
  UintahFieldManager<FieldT>::has_field( const Tag& tag ) const
  {
    return (idFieldInfoMap_.find( tag ) != idFieldInfoMap_.end());
  }

//--------------------------------------------------------------------

  template< typename FieldT >
  UintahFieldManager<FieldT>::~UintahFieldManager()
  {
    ExecMutex lock;
    BOOST_FOREACH( typename IDFieldInfoMap::value_type& myPair, idFieldInfoMap_ ){
      Uintah::VarLabel::destroy( myPair.second.uintahInfo.varlabel );
    }
  }

//--------------------------------------------------------------------

  template< typename FieldT >
  inline bool
  UintahFieldManager<FieldT>::release_field( const Tag& tag )
  {
    ExecMutex lock;
    const typename IDFieldInfoMap::iterator i = idFieldInfoMap_.find( tag );

    if( i == idFieldInfoMap_.end() ){
      std::ostringstream msg;
      msg << "ERROR!  No field " << tag << " has been registered.\n"
          << "Registered fields follow.\n";
      this->dump_fields( msg );
      msg << std::endl << __FILE__ << " : " << __LINE__;
      throw std::runtime_error( msg.str() );
    }

    FieldInfo& finfo = i->second;

    if( finfo.isLocked || !finfo.isAllocated ) return false;

    // only "release" fields that are scratch fields.  That is, they
    // have not been locked and are not managed by Uintah.
    switch ( finfo.memoryMgr ) {
      case UNKNOWN:{
        std::ostringstream msg;
        msg << "ERROR!  UNKNOWN MemoryManager for " << tag
            << std::endl << __FILE__ << " : " << __LINE__ << std::endl;
        throw std::runtime_error( msg.str() );
      }
      case MEM_EXTERNAL   : return false;
      case MEM_STATIC_GPU : return false;
      case MEM_DYNAMIC_GPU:  // fall through
      case MEM_DYNAMIC    :
#       ifdef DEBUG_FM_ALL
        std::cout << "UintahFieldManager::release_field() for " << tag << std::endl;
#       endif
        finfo.isAllocated = false;
        finfo.fieldPtr.detach();
        break;
    }
    return true;
  }

//--------------------------------------------------------------------

  template< typename FieldT >
  inline bool
  UintahFieldManager<FieldT>::lock_field( const Tag& tag )
  {
    ExecMutex lock;  // thread safety
#   ifdef DEBUG_FM_ALL
    std::cout << "UintahFieldManager::lock_field() on " << tag << std::endl;
#   endif

    const typename IDFieldInfoMap::iterator i = idFieldInfoMap_.find( tag );

    if( i == idFieldInfoMap_.end() ) {
      std::ostringstream msg;
      msg << "ERROR!  No field " << tag << " has been registered.\n"
          << "Registered fields follow.\n"
          << std::endl;
      this->dump_fields( msg );
      msg << std::endl << __FILE__ << " : " << __LINE__ << std::endl << std::endl;
      throw std::runtime_error( msg.str() );
    }

    i->second.isLocked = true;
    return true;
  }

//--------------------------------------------------------------------

  template< typename FieldT >
  inline bool
  UintahFieldManager<FieldT>::unlock_field( const Tag& tag )
  {
    ExecMutex lock;
    const typename IDFieldInfoMap::iterator i = idFieldInfoMap_.find( tag );
    if( i == idFieldInfoMap_.end() ) {
      std::ostringstream msg;
      msg << "ERROR!  No field " << tag << " has been registered."
          << std::endl
          << "Registered fields follow."
          << std::endl << " at " << __FILE__ << " : " << __LINE__
          << std::endl;
      this->dump_fields( msg );
      throw std::runtime_error( msg.str() );
    }
#   ifdef DEBUG_FM_ALL
    std::cout << "UintahFieldManager::unlock_field() for " << tag << std::endl;
#   endif
    i->second.isLocked = false;
    return true;
  }

  //-----------------------------------------------------------------

  template< typename FieldT >
  void
  UintahFieldManager<FieldT>::
  prep_field_for_consumption( const Tag& tag,
                              const short int deviceIndex )
  {
    ExecMutex lock;

#   ifdef DEBUG_FM_ALL
    std::cout << "UintahFieldManager::prep_field_for_consumption() for " << tag << std::endl;
#   endif

#   ifdef ENABLE_CUDA
    try {
      typename IDFieldInfoMap::iterator ifldx = idFieldInfoMap_.find( tag );
      FieldInfo& finfo = ifldx->second;

      if( !finfo.isAllocated ) {
        std::ostringstream msg;
        msg << "Prep_field_for_consumption called on an unallocated field: "
            << tag << "\n\t"
            << __FILE__ << " : " << __LINE__ << "\n";
        throw std::runtime_error( msg.str() );
      }

      // Attempts to add the requested field to the device
      finfo.fieldPtr->add_device_async( deviceIndex );
    }
    catch( std::runtime_error& e ) {
      std::ostringstream msg;
      msg << "UintahFieldManager::prep_field_for_consumption failed.\n\t"
          << " call values => " << tag.name() << " , "
          << "deviceIndex : " << deviceIndex << "\n\t-"
          << " at " << __FILE__ << " : " << __LINE__ << std::endl
          << " More information:\n" << e.what() << std::endl;
      throw std::runtime_error( msg.str() );
    }
#   endif
  }

  //-----------------------------------------------------------------


  template< typename FieldT >
  void
  UintahFieldManager<FieldT>::
  validate_field_location( const Tag& tag,
                           const short int deviceIndex )
  {
    ExecMutex lock;
#   ifdef DEBUG_FM_ALL
    std::cout << "UintahFieldManager::validate_field_location() for " << tag
              << " & field Location : " << deviceIndex << std::endl;
#   endif
    // Note : Synchronization of a field location is only necessary in case of
    //        heterogeneous task graph, where the Uintah field location-CPU has
    //        to be synchronized with the scratch GPU field location
#   ifdef ENABLE_CUDA
    if( !isHomogeneousgpu_ ){
      try{
        typename IDFieldInfoMap::iterator ifldx = idFieldInfoMap_.find( tag );
        FieldInfo& finfo = ifldx->second;

        if( !finfo.isAllocated ){
          std::ostringstream msg;
          msg << "validate_field_location() called on an unallocated field: "
              << tag << "\n\t"
              << __FILE__ << " : " << __LINE__ << "\n";
          throw std::runtime_error( msg.str() );
        }

        // Attempts to sync the requested field to the device
        finfo.fieldPtr->validate_device( deviceIndex );
      }
      catch( std::runtime_error& e ){
        std::ostringstream msg;
        msg << "UintahFieldManager::validate_field_location() failed !\n\t"
            << " call values => " << tag.name() << " , "
            << "deviceIndex : " << deviceIndex << "\n\t-"
            << " at " << __FILE__ << " : " << __LINE__ << std::endl
            << " More information:\n" << e.what() << std::endl;
        throw std::runtime_error( msg.str() );
      }
    }
#   endif
  }

  //-----------------------------------------------------------------

  template< typename FieldT >
  void
  UintahFieldManager<FieldT>::
  set_active_field_location( const Tag& tag,
                             const short int deviceIndex )
  {
    ExecMutex lock;
#   ifdef DEBUG_FM_ALL
    std::cout << "UintahFieldManager::set_active_field_location() for " << tag
              << " & field Location : " << deviceIndex << std::endl;
#   endif
    // Note : setting a field location to be active is only required in a
    //        heterogeneous GPU Uintah task.
#   ifdef ENABLE_CUDA
    if( !isHomogeneousgpu_ ){
      try {
        typename IDFieldInfoMap::iterator ifldx = idFieldInfoMap_.find( tag );
        FieldInfo& finfo = ifldx->second;

        if( !finfo.isAllocated ) {
          std::ostringstream msg;
          msg << "set_active_field_location() called on an unallocated field: "
              << tag << "\n\t"
              << __FILE__ << " : " << __LINE__ << "\n";
          throw std::runtime_error( msg.str() );
        }

        // set the field active for the deviceIndex
        finfo.fieldPtr->set_device_as_active( deviceIndex );
      }
      catch( std::runtime_error& e ) {
        std::ostringstream msg;
        msg << "UintahFieldManager::set_active_field_location() failed !\n\t"
            << " call values => " << tag.name() << " , "
            << "deviceIndex : " << deviceIndex << "\n\t-"
            << " at " << __FILE__ << " : " << __LINE__ << std::endl
            << " More information:\n" << e.what() << std::endl;
        throw std::runtime_error( msg.str() );
      }
    }
#   endif
  }

  //-----------------------------------------------------------------

  template< typename FieldT >
  void
  UintahFieldManager<FieldT>::set_field_memory_manager( const Tag& tag,
      const MemoryManager m,
      const short int deviceIndex )
  {
    ExecMutex lock;
    const typename IDFieldInfoMap::iterator i = idFieldInfoMap_.find( tag );
    if( i == idFieldInfoMap_.end() ) {
      std::ostringstream msg;
      msg << "ERROR!  No field " << tag << " has been registered."
          << std::endl
          << "Registered fields follow."
          << std::endl << " at " << __FILE__ << " : " << __LINE__
          << std::endl;
      this->dump_fields( msg );
      throw std::runtime_error( msg.str() );
    }

    FieldInfo& finfo = i->second;

#   ifdef DEBUG_WRITE_FIELD_MANAGER_UPDATES
    std::cout << "Setting memory manager for " << tag << " -> " << finfo.memoryMgr
        << " & device Location : " << deviceIndex << std::endl;
#   endif

    const MemoryManager to = m;
    const MemoryManager from = finfo.memoryMgr;

    if( to == from ) return;  // nothing to do.

    finfo.memoryMgr = m;
    finfo.deviceIndex = deviceIndex;

#   ifdef DEBUG_FM_ALL
    std::cout << "MemoryManager swap: "
        << get_memory_manager_description( from ) << " -> "
        << get_memory_manager_description( to )
        << " " << tag << std::endl;
#   endif
  }

  //-----------------------------------------------------------------

  template< typename FieldT >
  FieldID
  UintahFieldManager<FieldT>::register_field( const Tag& tag )
  {
    ExecMutex lock;
#   ifdef DEBUG_FM_ALL
    std::cout << "UintahFieldManager::register_field() for " << tag << std::endl;
#   endif

    if( tag.context() == INVALID_CONTEXT ) {
      std::ostringstream msg;
      msg << "invalid context detected on '" << tag.name() << "'"
          << std::endl << " at " << __FILE__ << " : " << __LINE__ << std::endl;
      throw std::invalid_argument( msg.str() );
    }

    const typename IDFieldInfoMap::iterator i = idFieldInfoMap_.find( tag );
    if( i == idFieldInfoMap_.end() ) {
      UintahFieldAllocInfo uinfo( Uintah::VarLabel::create( tag.name(),
                                                            Wasatch::get_uintah_field_type_descriptor<FieldT>() ),
                                  REQUIRES,  // default - can be changed
                                  Wasatch::get_n_ghost<FieldT>(),
                                  Wasatch::get_uintah_ghost_type<FieldT>(),
                                  tag.context() == STATE_N );

      FieldInfo finfo;
      finfo.uintahInfo = uinfo;
      idFieldInfoMap_[tag] = finfo;

      // stash this in the idUintahMap_, which is available outside of this for use in Wasatch
      idUintahMap_[tag] = &idFieldInfoMap_[tag].uintahInfo;

      // CARRY_FORWARD fields must register the "old" value as well.  We call that STATE_N.
      if( tag.context() == CARRY_FORWARD ) {
        register_field( Tag( tag.name(), STATE_N ) );
      }
    }

    return tag.id();
  }

  //-----------------------------------------------------------------

  template< typename FieldT >
  void
  UintahFieldManager<FieldT>::allocate_fields( const boost::any& anyinfo )
  {
    ExecMutex lock;
#   ifdef DEBUG_FM_ALL
    std::cout << "UintahFieldManager::allocate_fields()" << std::endl;
#   endif
    const Wasatch::AllocInfo& allocInfo = boost::any_cast<
        boost::reference_wrapper<const Wasatch::AllocInfo> >( anyinfo );

    // stash this for use when we call field_ref, which is where the actual allocation is performed
    allocInfo_ = &allocInfo;
    isHomogeneousgpu_ = allocInfo_->isGPUTask;

    // extract information from the uintah data warehouse and bind it here.
    BOOST_FOREACH( typename IDFieldInfoMap::value_type& myPair, idFieldInfoMap_ ){

      FieldInfo& finfo = myPair.second;

      // allocate the field
      if( !finfo.isAllocated && (finfo.memoryMgr == MEM_EXTERNAL || finfo.memoryMgr == MEM_STATIC_GPU) ) {
        finfo.uintahFieldContainer = new UintahFieldContainer<FieldT>();
        short int deviceIndex = CPU_INDEX;
#       ifdef ENABLE_CUDA
        // TODO : Fix this for multiple GPUs
        if( finfo.memoryMgr == MEM_STATIC_GPU ) deviceIndex = GPU_INDEX;
#       endif
        finfo.fieldPtr = UintahFieldAllocator<FieldT>::allocate_field( *finfo.uintahFieldContainer, allocInfo_, finfo.uintahInfo, deviceIndex );
        finfo.isAllocated = true;
      }
    }
    fieldsAreAllocated_++;
  }

  //-----------------------------------------------------------------

  template< typename FieldT >
  void
  UintahFieldManager<FieldT>::deallocate_fields()
  {
    ExecMutex lock;

    --fieldsAreAllocated_;
    if( fieldsAreAllocated_ != 0 ) return;

    BOOST_FOREACH( typename IDFieldInfoMap::value_type& myPair, idFieldInfoMap_ ){
      FieldInfo& finfo = myPair.second;
      if( finfo.isAllocated ){
#       ifdef DEBUG_FM_ALL
        std::cout << "UintahFieldManager::deallocate_fields() for " << myPair.first << std::endl;
#       endif
        switch( finfo.memoryMgr ){
          case MEM_DYNAMIC:
            // Memory has to get back to CPU Memory pool
            finfo.fieldPtr.detach();
            break;

          case MEM_EXTERNAL:
            delete finfo.uintahFieldContainer;
            finfo.uintahFieldContainer = NULL;
            break;

#         ifdef ENABLE_CUDA
          case MEM_STATIC_GPU: {
            delete finfo.uintahFieldContainer;
            finfo.uintahFieldContainer = NULL;
            break;
          }
          // Memory has to get back to GPU Memory pool
          case MEM_DYNAMIC_GPU: {
            finfo.fieldPtr.detach();
            break;
          }
#         endif

          default :{
            std::ostringstream msg;
            msg << "Error: Trying to deallocate an unsupported memory field " << finfo.memoryMgr
            << "\n\t" << __FILE__ << " : " << __LINE__ << std::endl;
            throw std::runtime_error( msg.str() );
          }
        }
        finfo.isAllocated = false;
      }
    }
  }

  //-----------------------------------------------------------------

  template< typename FieldT >
  const FieldT&
  UintahFieldManager<FieldT>::field_ref( const Tag& tag ) const
  {
    ExecMutex lock;

#   ifdef DEBUG_FM_ALL
    std::cout << "UintahFieldManager::field_ref() const for " << tag << std::endl;
#   endif

    if( allocInfo_ == NULL ){
      std::ostringstream msg;
      msg
      << "ERROR! Must first call allocate_fields() before calling field_ref()\n\t"
      << __FILE__ << " : " << __LINE__ << std::endl;
      throw std::runtime_error( msg.str() );
    }

    const typename IDFieldInfoMap::const_iterator itag = idFieldInfoMap_.find( tag );
    if( itag == idFieldInfoMap_.end() ) {
      std::ostringstream msg;
      msg << "ERROR in call to field_ref() - no field with tag \n\t"
          << tag << "\n\thas been registered!\n\t"
          << "Note: a common reason for this is if you asked for a field\n"
          << "but used the wrong type of FieldManager.\n\n"
          << "Registered fields follow:\n";
      dump_fields( msg );
      msg << "\n\t" << __FILE__ << " : " << __LINE__ << std::endl;
      throw std::runtime_error( msg.str() );
    }

    const FieldInfo& finfo = itag->second;

    // Disallow a const reference to a field that is dynamic,
    // since it cannot possibly hold any valid values.
    if( !finfo.isAllocated ) {
      std::ostringstream msg;
      msg << "ERROR! Requesting a const reference to field\n\t" << tag
          << "\n\twhich is not persistent and has been released!\n\t"
          << __FILE__ << " : " << __LINE__ << std::endl;
      throw std::runtime_error( msg.str() );
    }
    return *finfo.fieldPtr;
  }

  //-----------------------------------------------------------------

  namespace detail {
    template< typename FieldT >
    inline void
    field_copy( const FieldT& src, FieldT& dest ){
      using namespace SpatialOps;
      dest <<= src;
    }
    template< >
    inline void
    field_copy<double>( const double& src, double& dest ){
      dest = src;
    }
  }

  template< typename FieldT >
  FieldT&
  UintahFieldManager<FieldT>::
  field_ref( const Tag& tag )
  {
    ExecMutex lock;

    const typename IDFieldInfoMap::iterator i = idFieldInfoMap_.find( tag );
    if( i == idFieldInfoMap_.end() ) {
      std::ostringstream msg;
      msg << "ERROR!  No field with tag " << tag << " has been registered!" << std::endl
          << "        Note: a common reason for this is if you asked for a field" << std::endl
          << "        but used the wrong type of FieldManager." << std::endl << std::endl
          << " registered fields follow:" << std::endl;
      dump_fields( msg );
      throw std::runtime_error( msg.str() );
    }

    FieldInfo& finfo = i->second;

#   ifdef DEBUG_FM_ALL
    std::cout << "UintahFieldManager::field_ref() for SpatialOps scratch field " << tag
              << " with deviceIndex : " << finfo.deviceIndex << std::endl;
#   endif

    if( !finfo.isAllocated ){
      switch ( finfo.memoryMgr ) {
        case MEM_DYNAMIC :
#         ifdef DEBUG_FM_ALL
          std::cout << " -> allocating " << tag << " as a SpatialOps scratch field" << std::endl;
#         endif
          finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field( allocInfo_ );
          finfo.isAllocated = true;
          break;

        case MEM_EXTERNAL :{  // should have been allocated during call to allocate_fields
          std::ostringstream msg;
          msg << "ERROR! Requesting a reference to field\n\t" << tag
              << "\nwhich is not persistent and has been released!\n\t"
              << __FILE__ << " : " << __LINE__ << std::endl;
          throw std::runtime_error( msg.str() );
        }

#       ifdef ENABLE_CUDA
        case MEM_DYNAMIC_GPU :
#         ifdef DEBUG_FM_ALL
          std::cout << " -> allocating " << tag << " as a SpatialOps GPU scratch field" << std::endl;
#         endif
#         ifndef NDEBUG
          if( !IS_GPU_INDEX(finfo.deviceIndex) ){
            std::ostringstream msg;
            msg << "ERROR! Invalid deviceIndex found while allocating for GPU scratch fields : " << finfo.deviceIndex
                << __FILE__ << " : " << __LINE__ << std::endl;
            throw std::runtime_error( msg.str() );
          }
#         endif
          finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field( allocInfo_, finfo.deviceIndex );
          finfo.isAllocated = true;
          break;

        case MEM_STATIC_GPU :{  // should have been allocated during call to allocate_fields
          std::ostringstream msg;
          msg << "ERROR! Requesting a reference to field\n\t" << tag
              << "\nwhich is not persistent and has been released!\n\t"
              << __FILE__ << " : " << __LINE__ << std::endl;
          throw std::runtime_error( msg.str() );
        }
#       endif
        default :{
          std::ostringstream msg;
          msg << "Error: Request for unsupported memory manager " << finfo.memoryMgr
              << "\n\t" << __FILE__ << " : " << __LINE__ << std::endl;
          throw std::runtime_error( msg.str() );
        }
      } // switch
    } // if

    return *finfo.fieldPtr;
  }

  //-----------------------------------------------------------------

  template< typename FieldT >
  void
  UintahFieldManager<FieldT>::
  dump_fields( std::ostream& os ) const
  {
    using namespace std;
    if( idFieldInfoMap_.empty() ) return;

    os.setf( ios::left );
    os << setw( 40 ) << "Field Name" << setw( 12 ) << "Context" << setw( 12 )
       << "Mode" << setw( 20 ) << "Field Manager Type" << endl
       << "-----------------------------------------------------------------"
       << endl;
    BOOST_FOREACH( const typename IDFieldInfoMap::value_type& myPair, idFieldInfoMap_ ){
      const FieldInfo& finfo = myPair.second;
      os << setw(40) << myPair.first.name()
         << setw(12) << myPair.first.context()
         << setw(12) << finfo.uintahInfo.mode
         << setw(20) << finfo.memoryMgr << endl;
    }
  }

  //-----------------------------------------------------------------


  template< typename FieldT >
  void
  UintahFieldManager<FieldT>::
  copy_field_forward( const Tag& tag, FieldT& f ) const
  {
    detail::field_copy( field_ref( Tag(tag.name(), STATE_N) ), f );
  }

  //-----------------------------------------------------------------

}  // namespace Expr

#endif

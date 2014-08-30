#ifndef LBMS_FieldManager_h
#define LBMS_FieldManager_h

//-- boost includes --//
#include <boost/any.hpp>
#include <boost/ref.hpp>

//-- expression library includes --//
#include <expression/DefaultFieldManager.h>
#include <expression/Tag.h>

#include <fields/Fields.h>

namespace Expr{

  struct LBMSFieldInfo
  {
    bool xbcp, ybcp, zbcp;
    const SpatialOps::IntVec nFine, nCoarse;
    LBMSFieldInfo( const bool xbc, const bool ybc, const bool zbc,
                   const SpatialOps::IntVec fineDim,
                   const SpatialOps::IntVec coarseDim )
      : xbcp( xbc ), ybcp( ybc ), zbcp( zbc ),
        nFine( fineDim ),
        nCoarse( coarseDim )
    {}
  };

  namespace detail{
    template< typename FieldT >
    inline FieldAllocInfo set_field_info( const LBMSFieldInfo& lbmsinfo ){
      return FieldAllocInfo( lbmsinfo.nFine, 0, 0, lbmsinfo.xbcp, lbmsinfo.ybcp, lbmsinfo.zbcp );
    }

    template<>
    inline FieldAllocInfo set_field_info<SpatialOps::SVolField>( const LBMSFieldInfo& lbmsinfo ){
      return FieldAllocInfo( lbmsinfo.nCoarse, 0, 0, lbmsinfo.xbcp, lbmsinfo.ybcp, lbmsinfo.zbcp );
    }
    template<>
    inline FieldAllocInfo set_field_info<SpatialOps::SSurfXField>( const LBMSFieldInfo& lbmsinfo ){
      return FieldAllocInfo( lbmsinfo.nCoarse, 0, 0, lbmsinfo.xbcp, lbmsinfo.ybcp, lbmsinfo.zbcp );
    }
    template<>
    inline FieldAllocInfo set_field_info<SpatialOps::SSurfYField>( const LBMSFieldInfo& lbmsinfo ){
      return FieldAllocInfo( lbmsinfo.nCoarse, 0, 0, lbmsinfo.xbcp, lbmsinfo.ybcp, lbmsinfo.zbcp );
    }
    template<>
    inline FieldAllocInfo set_field_info<SpatialOps::SSurfZField>( const LBMSFieldInfo& lbmsinfo ){
      return FieldAllocInfo( lbmsinfo.nCoarse, 0, 0, lbmsinfo.xbcp, lbmsinfo.ybcp, lbmsinfo.zbcp );
    }
  }
  //--------------------------------------------------------------------

  template<typename FieldT>
  class LBMSFieldManager : public DefaultFieldManager<FieldT>
  {
  public:
    LBMSFieldManager(){}
    ~LBMSFieldManager();

    inline void allocate_fields( const boost::any& );

  private:
    LBMSFieldManager( const LBMSFieldManager& ); // no copying
    LBMSFieldManager& operator=( const LBMSFieldManager& ); // no assignment
  };

  //------------------------------------------------------------------

  template<typename FieldT>
  LBMSFieldManager<FieldT>::~LBMSFieldManager()
  {
    this->deallocate_fields();
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  void
  LBMSFieldManager<FieldT>::allocate_fields( const boost::any& anyinfo )
  {
    try{
      const LBMSFieldInfo& lbmsinfo = boost::any_cast< boost::reference_wrapper<const LBMSFieldInfo> >(anyinfo);
      const FieldAllocInfo info = detail::set_field_info<FieldT>( lbmsinfo );
      DefaultFieldManager<FieldT>::allocate_fields( boost::cref(info) );
    }
    catch( const boost::bad_any_cast& err ){
      std::ostringstream msg;
      msg << std::endl
          << "Expected 'Expr::LBMSFieldInfo' object to allocate_fields()." << std::endl
          << "  Ensure that calls to TimeStepper::finalize() and LBMSFieldManager::allocate_fields()" << std::endl
          << "  supply an Expr::LBMSFieldInfo object" << std::endl
          << "  Something like allocate_fields( boost::cref( info ) );" << std::endl << std::endl
          << __FILE__ << " : " << __LINE__ << std::endl;
      throw std::runtime_error( msg.str() );
    }
  }

  //==================================================================

} // namespace Expr

#endif // LBMS_FieldManager_h

#include <CCA/Components/Arches/Task/FieldContainer.h>

using namespace Uintah; 

ArchesFieldContainer::ArchesFieldContainer( const Wasatch::AllocInfo& alloc_info, const Patch* patch )
  : _wasatch_ainfo(alloc_info), _patch(patch)
{

  _nonconst_var_map.clear(); 
  _const_var_map.clear(); 

}

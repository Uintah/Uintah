#include <CCA/Components/Arches/Task/FieldContainer.h>

using namespace Uintah;

ArchesFieldContainer::ArchesFieldContainer( const Patch* patch,
                                            const int matl_index,
                                            const VariableRegistry variable_reg,
                                            DataWarehouse* old_dw,
                                            DataWarehouse* new_dw )
  :
    _patch(patch),
    _matl_index(matl_index),
    _old_dw(old_dw),
    _new_dw(new_dw),
    _variable_reg(variable_reg)
{

  _nonconst_var_map.clear();
  _const_var_map.clear();

}

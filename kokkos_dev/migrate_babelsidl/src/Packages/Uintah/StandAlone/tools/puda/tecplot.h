
#include <string>

namespace Uintah {

  class DataArchive;

  void tecplot( DataArchive *       da,
                const bool          tslow_set, 
                const bool          tsup_set,
                unsigned long &     time_step_lower,
                unsigned long &     time_step_upper,
                bool                do_all_ccvars,
                const std::string & ccVarInput,
                const std::string & i_xd,
                int                 tskip );
}


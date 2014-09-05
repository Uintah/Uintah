
namespace Uintah {

  class DataArchive;

  void asci( DataArchive *   da,
             const bool      tslow_set,
             const bool      tsup_set,
             unsigned long & time_step_lower,
             unsigned long & time_step_upper );
  
}

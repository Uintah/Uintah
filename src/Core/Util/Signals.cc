#include <Core/Util/Signals.h>

namespace SCIRun {

void connect( Signal &s, void (*fun)(), int priority=0 )
{
  s.add ( new StaticSlot( fun, priority ) );
}

bool disconnect( Signal &s, void (*fun)() )
{
  return s.rem( StaticSlot( fun ) );
}
  
} // end namespace SCIRun


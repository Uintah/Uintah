#include "whoc.hh"
#include "govcca.hh"

extern "C" govcca::Component make_Babel_whoc()
{
  return whoc::Com::_create();
}


#include "whof77.hh"
#include "govcca.hh"

extern "C" govcca::Component make_Babel_whof77()
{
  return whof77::Com::_create();
}


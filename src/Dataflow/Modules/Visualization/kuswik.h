#ifndef _KUSWIK_H_
#define _KUSWIK_H_

#include <Core/Geom/Color.h>
using namespace SCIRun;

#define at() cerr << "@ " << __FILE__ << ":" << __LINE__ << endl

#ifndef TRUE
  #define TRUE 1
  #define FALSE 0
#endif

const Color BLACK( 0., 0., 0. );
const Color WHITE( 1., 1., 1. );

#endif /* _KUSWIK_H_ */

#ifndef _KUSWIK_H_
#define _KUSWIK_H_

#include <Geom/Color.h>

#define at() cerr << "@ " << __FILE__ << ":" << __LINE__ << endl

#ifndef TRUE
  #define TRUE 1
  #define FALSE 0
#endif

const SCICore::GeomSpace::Color BLACK( 0., 0., 0. );
const SCICore::GeomSpace::Color WHITE( 1., 1., 1. );

#endif /* _KUSWIK_H_ */

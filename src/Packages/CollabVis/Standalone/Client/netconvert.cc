#include <stdio.h>
#include <stdlib.h>
#include <iostream.h>

#define ASFMT_SHORT_NAMES
#include "Network/AppleSeeds/formatseed.h"

main( int argc, char ** argv ) {
  
  DataDescriptor d = SIMPLE_DATA( FLOAT_TYPE, 1 );
  float in, out;
  int todo = 1 << (sizeof(int) - 1);
  int done = 0;
  if ( argc > 1 )
    todo = atoi( argv[1] );
  cerr << "Doing " << todo << " conversions." << endl;
  
  while (!feof( stdin ) ) {
    if ( done >= todo ) break;
    fread( &in, sizeof(float), 1, stdin );
    if ( feof( stdin ) ) break;
    
    ConvertHostToNetwork( (void*)&out,
			  (void*)&in,
			  &d,
			  1 );
    fwrite( &out, sizeof(float), 1, stdout );
  }
}

/*
To reduce memory usage, user has the compile time option to
not store the particle size. This has some consequences:

1. The code assumes that the initial particle resolution is 2x2x2 per cell 

To turn off storing the particle size, comment out the #define below, and uncomment
the #undef
*/


#define KEEP_PSIZE
//#undef KEEP_PSIZE

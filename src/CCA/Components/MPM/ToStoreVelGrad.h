/*
To reduce memory usage, user has the compile time option to turn 
not store the velocity gradient at the particles. This has some consequences:

1.  Only UCNH material model will work.  Other Hyperelastic models can be made
to work. 
2.  Artificial viscosity will not be available.

To turn off storing the velocity gradient, comment out the #define below, and uncomment
the #undef
*/


#define KEEP_VELGRAD
//#undef KEEP_VELGRAD

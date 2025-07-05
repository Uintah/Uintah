/*
To reduce memory usage, user has the compile time option to turn off all
thermal related code.  This will only work with certain constitutive models,
namely UCNH (aka comp_neo_hook*) for now.

To turn off thermal calculations, comment out the #define below, and uncomment
the #undef
*/


#define INCLUDE_THERMAL
//#undef INCLUDE_THERMAL

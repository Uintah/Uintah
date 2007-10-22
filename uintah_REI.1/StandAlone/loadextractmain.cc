#include <Packages/Uintah/StandAlone/loadextract.h>

int main(int argc, char** argv)
{
   loadextract loadex("sqplate.k");
   //loadextract loadex("C2500pickup.k");
   loadex.extract(argc, argv);

  //__________________________________
  //  Default Values
}

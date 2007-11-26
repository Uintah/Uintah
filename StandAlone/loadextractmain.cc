#include <Packages/Uintah/StandAlone/loadextract.h>

int main(int argc, char** argv)
{
  FILE * fparam;
  char kfilename[1024];
  float pconv, lconv1, lconv2;

  fparam = fopen("loadParam.txt", "r"); //open the parameter file


  if (fparam==NULL)
    {
      printf("Can't open the parameter file! Quit!\n");
      exit(0);
    }

  fscanf(fparam, "%s\n", kfilename);
  fscanf(fparam, "%f %f %f", &pconv, &lconv1, &lconv2);
  fclose(fparam);

  loadextract loadex(kfilename, pconv, lconv1, lconv2);
   //loadextract loadex("C2500pickup.k");
   loadex.extract(argc, argv);

  //__________________________________
  //  Default Values
}

#include <Packages/Uintah/StandAlone/loadextract.h>

int main(int argc, char** argv)
{
  FILE * fparam;
  fparam = fopen("loadParam.txt", "r"); //open the parameter file


  if (fparam==NULL)
    {
      printf("Can't open the parameter file! Quit!\n");
      exit(0);
    }

  char kfilename[1024];
  fscanf(fparam, "%s\n", kfilename);
  loadextract loadex(kfilename);
  fscanf(fparam, "%lf, %lf, %lf\n",&(loadex.presconv),
	 &(loadex.lengthconv1),
	 &(loadex.lengthconv2));
  fclose(fparam);
   //loadextract loadex("C2500pickup.k");
   loadex.extract(argc, argv);

  //__________________________________
  //  Default Values
}

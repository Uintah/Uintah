/*
 *
 *
 */

#include <Classlib/String.h>
#include <Classlib/Pstreams.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Classlib/Array3.h>
#include <iostream.h>
#include <fstream.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>


#define FILE_HEADER     62

int
main( int argc, char *argv[] )
{
  char *in_name;
  unsigned char *density;
  int density_size;
  int density_fd;
  int x,y,z;

  if (argc != 5)
    {
      printf("Usage: DENtoSF <patientName> <x> <y> <z> \n");
      exit(0);
    }

  in_name = argv[1];
  x = atoi(argv[2]);
  y = atoi(argv[3]);
  z = atoi(argv[4]);

  density_size = x*y*z;
  density = new unsigned char[density_size];

  /* load the raw data */
  if ((density_fd = open(in_name, 0)) < 0) {
    perror("open");
    fprintf(stderr, "could not open %s\n", in_name);
    exit(1);
  }
  if (lseek(density_fd, FILE_HEADER, 0) < 0) {
    perror("seek");
    fprintf(stderr, "could not read data from %s\n", in_name);
    exit(1);
  }
  if (read(density_fd, density, density_size) != density_size) {
    perror("read");
    fprintf(stderr, "could not read data from %s\n", in_name);
    exit(1);
  }
  close(density_fd);

  ScalarFieldRG* sf=new ScalarFieldRG;
  sf->resize(x, y, z);
  
  int i,j,k,count;
  count = 0;
  for(k=0;k<z;k++)
    for(i=0;i<x;i++)
      for(j=0;j<y;j++)
	sf->grid(i,j,k) = density[count++];

  char outname[100];
  sprintf(outname, "%s.%03d.sfrg", in_name, i);
  TextPiostream stream(outname, Piostream::Write);
  ScalarFieldHandle sh=sf;
  Pio(stream, sh);
  

  return 0;
}


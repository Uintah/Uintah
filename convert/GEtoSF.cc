/*
 * main.c - Does everything
 *
 * David Weinstein, January 1994
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

double slice_spacing;
int num_dir;
int num_slices;
int hdr_len;
int width;
ScalarFieldRG* sf;
char *in_name;

void read_header() {
    FILE *fp;
    char fullpath[100];
    sprintf(fullpath, "/home/sci/data1/brain/%s/sci.hdr", in_name);
    if ((fp=fopen(fullpath, "rt")) == NULL) {
	printf("Couldn't open %s  Exiting...\n", fullpath);
	exit(0);
    }
    fscanf(fp, "%d", &num_dir);
    fscanf(fp, "%d", &num_slices);
    fscanf(fp, "%d", &hdr_len);
    fscanf(fp, "%d", &width);
    fclose(fp);
}

void read_files() {
    FILE *fp;
    char fullpath[100];
    int i1, i2;
    int max=0;
  
    for (int i=1; i<=num_dir; i++) {
	ScalarFieldRG* sf=new ScalarFieldRG;
	sf->resize(width, width, num_slices);
	for (int j=0; j<num_slices; j++) {
	    sprintf(fullpath,"/home/sci/data1/brain/%s/%03d/I.%03d", in_name,
		    i, j*num_dir+i);
	    if ((fp=fopen(fullpath, "rt")) == NULL) {
		printf("Couldn't open %s  Exiting...\n", fullpath);
		exit(0);
	    }
	    for (int h=0; h<hdr_len; h++) {
		if (getc(fp) == EOF) exit(0);
	    }
	
	    for (int y=width-1;y>=0;y--) {
		for (int x=0;x<width;x++) {
		    if ((i1=getc(fp)) == EOF) {
			printf("Image file wasn't long enough.\n");
			exit(0);
		    }
		}	
		i1 *= 256;
		if ((i2=getc(fp)) == EOF) {
		    printf("Image file wasn't long enough.\n");
		    exit(0);
		}	
		i1+=i2;
		if (i1>max) max=i1;
		sf->grid(x,y,j)=i1;
	    }
	}
	double scale=255./max;
	for (int x=0; x<width; x++)
	    for (int y=0; y<width; y++)
		for (int z=0; z<num_slices; z++)
		    sf->grid(x,y,z) = sf->grid(x,y,z)*scale;	
	char outname[100];
	sprintf(outname, "%s.%03d.sfrg", in_name, i);
	TextPiostream stream(outname, Piostream::Write);
	ScalarFieldHandle sh=sf;
	Pio(stream, sh);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
	printf("Usage: GEtoSF patientName\n");
	exit(0);
    }
    in_name=argv[1];
    read_header();
    read_files();

    return 0;
}

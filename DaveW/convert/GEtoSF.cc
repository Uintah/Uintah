/*
 * main.c - Does everything
 *
 * David Weinstein, April 1996
 *
 */

#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array2.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/ScalarFieldRGBase.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/ScalarFieldRGdouble.h>
#include <SCICore/Datatypes/ScalarFieldRGfloat.h>
#include <SCICore/Datatypes/ScalarFieldRGint.h>
#include <SCICore/Datatypes/ScalarFieldRGshort.h>
#include <SCICore/Datatypes/ScalarFieldRGuchar.h>
#include <SCICore/Datatypes/ScalarFieldRGchar.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Persistent/Pstreams.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>

using namespace SCICore::Containers;
using namespace SCICore::Datatypes;
using namespace SCICore::Geometry;
using namespace SCICore::Math;
using namespace SCICore::PersistentSpace;
using namespace std;

int padn;
int num_slices;
int hdr_len;
int width;
Array1<int> table;
ScalarFieldRGchar *sf_c, *sf;
char *in_name;
double zScale;
char out[100];
char baseName[100];

//  header file format:
//
//  [header length]
//  width
//  num_slices
//  zScale
//  output_filename

int read_header(FILE **fpp) {
    FILE *fp=*fpp;
    int OKflag=1;
    OKflag &= fscanf(fp, "%d", &num_slices);
    OKflag &= fscanf(fp, "%lf", &zScale);
    OKflag &= fscanf(fp, "%s", out);
    return OKflag;
}


//  segmentation file format:
//
//  [# structures]
//  default_matl_id
//  struct1_id  matl1_id
//  struct2_id  matl2_id

int read_legend(FILE **fpp) {
    FILE *fp=*fpp;
    int OKflag=1;
    int num_structures, default_matl;
    int struc, matl;

    OKflag &= fscanf(fp, "%d", &num_structures);
    table.resize(num_structures);
    OKflag &= fscanf(fp, "%d", &default_matl);
    for (int i=0; i<num_structures; i++) {
	table[i]=default_matl;
    }
    while(!feof(fp)) {
	if (fscanf(fp, "%d %d", &struc, &matl)==2)
	    table[struc]=matl;
    }
    return OKflag;
}

void read_seg_images(char *path) {
    FILE *fp;
    char image_path[100];
    int i1, i2;
  
    sf_c=new ScalarFieldRGchar;
    sf_c->resize(width+2*padn, width+2*padn, num_slices+2*padn);
    sf_c->grid.initialize('0');
    for (int j=0; j<num_slices; j++) {
	sprintf(image_path,"%s/%s.%03d", path, baseName, j+1);
	if ((fp=fopen(image_path, "rt")) == NULL) {
	    printf("Couldn't open %s  Exiting...\n", image_path);
	    exit(0);
	}
	for (int h=0; h<hdr_len; h++) {
	    if (getc(fp) == EOF) {
		cerr << "File "<<image_path<<" ended prematurely!  Exiting...\n";
		exit(0);
	    }
	}
	
	for (int y=width-1;y>=0;y--) {
	    for (int x=0;x<width;x++) {
		if ((i1=getc(fp)) == EOF) {
		    printf("Image file wasn't long enough.\n");
		    exit(0);
		}
		i1 *= 256;
		if ((i2=getc(fp)) == EOF) {
		    printf("Image file wasn't long enough.\n");
		    exit(0);
		}	
		i1+=i2;
//		if ((y==width-1) || (y==0) || (x==width-1) || (x==0)) i1=0;
		// change this so that instead of storing the data in the 
		// sf field, we are just counting how many times we see each 
		// data value
		sf_c->grid(x+padn,y+padn,j+padn)=(char)table[i1]+'0';
	    }	
	}
	fclose(fp);
    }
    sf_c->compute_minmax();
    sf_c->set_bounds(Point(0,0,0), Point(2, 2, (num_slices-1+2*padn)*2*zScale/(width-1+2*padn)));
}

void read_mri_images(char *path) {
    FILE *fp;
    char image_path[100];
    int i1, i2;
    int max=0;

//    Array3<int> vals(width, width, num_slices);
    sf=new ScalarFieldRGchar;
    sf->resize(width+2*padn, width+2*padn, num_slices+2*padn);
    sf->grid.initialize(0);
    for (int j=0; j<num_slices; j++) {
	sprintf(image_path,"%s/%s.%03d", path, baseName, num_slices-j);
	if ((fp=fopen(image_path, "rt")) == NULL) {
	    printf("Couldn't open %s  Exiting...\n", image_path);
	    exit(0);
	}
	for (int h=0; h<hdr_len; h++) {
	    if (getc(fp) == EOF) {
		cerr << "File "<<image_path<<" ended prematurely!  Exiting...\n";
		exit(0);
	    }
	}
	
	for (int y=0; y<width; y++) {
	    for (int x=0;x<width;x++) {
		if ((i1=getc(fp)) == EOF) {
		    printf("Image file wasn't long enough.\n");
		    exit(0);
		}
		i1 *= 256;
		if ((i2=getc(fp)) == EOF) {
		    printf("Image file wasn't long enough.\n");
		    exit(0);
		}	
		i1+=i2;
//		if ((y==width-1) || (y==0) || (x==width-1) || (x==0)) i1=0;
		if (i1>max) max=i1;
//		vals(x,y,j)=i1;
		sf->grid(x+padn,y+padn,j+padn)=i1;
	    }	
	}
	fclose(fp);
    }
//    double scale=255./max;
//    for (int x=0; x<width; x++)
//	for (int y=0; y<width; y++)
//	    for (int z=0; z<num_slices; z++)
//		sf->grid(x+padn,y+padn,z+padn) = vals(x,y,z)*scale;
    sf->compute_minmax();
    double MIN, MAX;
    sf->get_minmax(MIN,MAX);
    cerr << "min="<<MIN<<"  max="<<MAX<<"  (max="<<max<<")\n";
    sf->set_bounds(Point(0,0,0), Point(width-1, width-1, (num_slices-1)*zScale));
}

void write_seg_field() {
    char outname[100];
    sprintf(outname, "%s.c_sfrg", out);
    printf("Writing file: %s\n", outname);
    TextPiostream stream(outname, Piostream::Write);
    ScalarFieldHandle sh=sf_c;
    sh->set_raw(1);
    Pio(stream, sh);
}

void write_mri_field() {
    char outname[100];
    sprintf(outname, "%s.sfrg", out);
    printf("Writing file: %s\n", outname);
    TextPiostream stream(outname, Piostream::Write);
    ScalarFieldHandle sh=sf;
    sh->set_raw(1);
    Pio(stream, sh);
}

int main(int argc, char *argv[]) {
    char path[100];
    char hdr[100];
    char legend[100];
    FILE *hdr_fp;
    FILE *legend_fp;
    int mri=0;

    if (argc < 2) {
	cerr << "Usage: "<<argv[0]<<" {-seg | -mri} [path] [baseName] [padn]\n";
	exit(0);
    }

    if (strcmp("-mri", argv[1])==0) mri=1;
    else if (strcmp("-seg", argv[1])!=0) {
	cerr << "Usage: "<<argv[0]<<" {-seg | -mri} [path] [baseName\n";
	exit(0);
    }

    if (argc >= 3)
	sprintf(path, "%s", argv[2]);
    else
	sprintf(path, ".");

    if (argc >= 4)
	sprintf(baseName, "%s", argv[3]);
    else
	if (mri) {
	    sprintf(baseName, "I");
	} else {
	    sprintf(baseName, "output");
	}
    
    if (argc == 5)
	padn=atoi(argv[4]);
    else
	padn=0;
    cerr << "Padding with "<<padn<<" voxels.\n";

    sprintf(hdr, "%s/sci.hdr", path);

    if (!mri)	
	sprintf(legend, "%s/seg.legend", path);

    if ((hdr_fp=fopen(hdr, "rt")) == NULL) {
	printf("Couldn't open header frile: %s  Exiting...\n", hdr);
	exit(0);
    }

    fscanf(hdr_fp, "%d", &hdr_len);
    fscanf(hdr_fp, "%d", &width);
    if (feof(hdr_fp) || !read_header(&hdr_fp)) {
	printf("Error reading header file: %s  Exiting...\n", hdr);
	exit(0);
    }	    
    fclose(hdr_fp);

    if (!mri) {
	if ((legend_fp=fopen(legend, "rt")) == NULL) {
	    printf("Couldn't open segmentation legend file: %s  Exiting...\n", legend);
	    exit(0);
	}
	if (feof(hdr_fp) || !read_legend(&legend_fp)) {
	    printf("Error reading segmentation legend file: %s  Exiting...\n", legend);
	    exit(0);
	}
	fclose(legend_fp);
    }

    if (mri)
	read_mri_images(path);
    else
	read_seg_images(path);


    if (mri)
	write_mri_field();
    else
	write_seg_field();

    return 0;
}

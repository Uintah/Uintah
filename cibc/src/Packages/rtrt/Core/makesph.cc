
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char* argv[])
{
    int nx,ny,nz;
    char out[200];
    if(argc==1){
	nx=ny=nz=atoi(argv[1]);
	sprintf(out, "sphere.%d", nx);
    } else {
	nx=atoi(argv[1]);
	ny=atoi(argv[2]);
	nz=atoi(argv[3]);
	sprintf(out, "sphere.%d_%d_%d", nx, ny, nz);
    }
    FILE* ofile=fopen(out, "w");
    for(int i=0;i<nx;i++){
	double x=2.*i/(nx-1)-1;
	for(int j=0;j<ny;j++){
	    double y=2.*j/(ny-1)-1;
	    for(int k=0;k<nz;k++){
		double z=2.*k/(nz-1)-1;
		float mag=sqrt(x*x+y*y+z*z);
		fwrite(&mag, sizeof(float), 1, ofile);
	    }
	}
    }
    fclose(ofile);
    char buf[100];
    sprintf(buf, "%s.hdr", out);
    FILE* hdr=fopen(buf, "w");
    fprintf(hdr, "%d %d %d\n", nx, ny ,nz);
    fprintf(hdr, "-1 -1 -1\n");
    fprintf(hdr, "1 1 1\n");
    fprintf(hdr, "%g %g\n", 0.0, sqrt(3.0));
    fclose(hdr);
}

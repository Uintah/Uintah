/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#include <cstdio>
#include <cstdlib>
#include <cmath>

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

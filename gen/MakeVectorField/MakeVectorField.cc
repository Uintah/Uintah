
#include <iostream.h>
#include <stdlib.h>
#include <Classlib/String.h>

#include <VectorFieldRG.h>

main(int argc, char** argv)
{
    if(argc != 4){
	cerr << "Specify nx, ny and nz as arguments\n";
	exit(-1);
    }
    int nx=atoi(argv[1]);
    int ny=atoi(argv[2]);
    int nz=atoi(argv[3]);
    VectorFieldRG* field=new VectorFieldRG;
    field->resize(nx, ny, nz);
    for(int i=0;i<nx;i++){
	for(int j=0;j<ny;j++){
	    for(int k=0;k<nz;k++){
		double x=double(i)/double(nx-1);
		double y=double(j)/double(ny-1);
		double z=double(k)/double(nz-1);
		x=x*2-1;
		y=y*2-1;
		z=z*2-1;
		double th=atan2(y, x);
		double r=sqrt(x*x+y*y);
		Vector v(sin(th)*r, cos(th)*r, .0);
		field->grid(i,j,k)=v;
	    }
	}
    }
    field->set_minmax(Point(0,0,0), Point(1,1,1));

    TextPiostream stream("field.out", Piostream::Write);
    field->io(stream);
}

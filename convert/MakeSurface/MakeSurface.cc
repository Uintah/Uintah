
#include <Multitask/Task.h>
#include <Surface.h>
#include <iostream.h>
#include <fstream.h>

int main(int argc, char** argv)
{
    // Initialize the multithreader
    TaskManager::initialize();

    if(argc != 4 && argc != 5){
	cerr << "Usage: " << argv[0] << " infile pointfile trifile [surface name]" << endl;
	return -1;
    }
    clString name(argc==5?argv[4]:argv[2]);

    TriSurface* surf=new TriSurface;
    surf->name=name;
    ifstream pfile(argv[1]);
    int npoints;
    pfile >> npoints;
    cerr << "Reading " << npoints << " points";
    cerr.flush();
    Array1<Point> points(npoints);
    for(int i=0;i<npoints;i++){
	if(i%1000 == 0){
	    cerr << ".";
	    cerr.flush();
	}
	double x,y,z;
	pfile >> x >> y >> z;
	points[i]=Point(x,y,z);
    }
    cerr << endl;
    Array1<int> pointnum(npoints);
    for(i=0;i<npoints;i++)
	pointnum[i]=-1;
    ifstream tfile(argv[2]);
    int ntris;
    tfile >> ntris;
    int pointno=0;
    cerr << "Reading " << ntris << " triangles";
    cerr.flush();
    for(i=0;i<ntris;i++){
	if(i%1000 == 0){
	    cerr << ".";
	    cerr.flush();
	}
	int i1, i2, i3;
	tfile >> i1 >> i2 >> i3;
	if(pointnum[i1]==-1){
	    surf->add_point(points[i1]);
	    pointnum[i1]=pointno++;
	}
	if(pointnum[i2]==-1){
	    surf->add_point(points[i2]);
	    pointnum[i2]=pointno++;
	}
	if(pointnum[i3]==-1){
	    surf->add_point(points[i3]);
	    pointnum[i3]=pointno++;
	}
	surf->add_triangle(pointnum[i1], pointnum[i2], pointnum[i3]);
    }
    cerr << endl;

    // Write out the persistent representation...
    cerr << "Writing output...\n";
    TextPiostream stream(argv[3], Piostream::Write);
    surf->io(stream);
    delete surf;
    return 0;
}

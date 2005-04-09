
#include <Classlib/Timer.h>
#include <Math/fft.h>
#include <Math/MiscMath.h>
#include <iostream.h>
#include <stdlib.h>

void usage(char* progname)
{
    cerr << progname << " d n [stride] [r]\n";
    exit(-1);

}

int main(int argc, char** argv)
{
    if(argc<3){
	usage(argv[0]);
    }
    int stride=0;
    if(argc>3){
	stride=atoi(argv[3]);
    }
    int r=1;
    if(argc>4){
	r=atoi(argv[4]);
    }
    int n=atoi(argv[2]);
    int d=atoi(argv[1]);
    CPUTimer timer;
    int s=stride==0?1:stride;
    int size=s*n;
    if(d==2){
	size*=n;
	stride=0;
    }
    float* data=new float[2*size];
    for(int x=0;x<size;x++){
	data[x*2]=1;
	data[x*2+1]=0;
    }
    unsigned long flops=0;
    unsigned long memrefs=0;
    timer.start();
    if(d==0){
	if(stride==0){
	    for(int x=0;x<size;x++){
		data[x*2]=x;
		data[x*2+1]=0;
	    }
	    for(x=0;x<size;x++){
		cout << data[x*2] << ' ' << data[x*2+1] << ' ';
	    }
	    cout << '\n';
	    for(int ii=0;ii<r;ii++){
		fft1d_float(data, n, 1, &flops, &memrefs);
	    for(x=0;x<size;x++){
		cout << data[x*2] << ' ' << data[x*2+1] << ' ';
	    }
	    cout << '\n';
		fft1d_float(data, n, -1, &flops, &memrefs);
	    }
	    for(x=0;x<size;x++){
		data[x*2]/=n;
		if(Abs(data[x*2]) < 1.e-5)
		    data[x*2]=0;
		data[x*2+1]/=n;
		if(Abs(data[x*2+1]) < 1.e-5)
		    data[x*2+1]=0;
	    }
	    for(x=0;x<size;x++){
		cout << data[x*2] << ' ' << data[x*2+1] << ' ';
	    }
	    cout << '\n';
	} else {
	    for(int x=0;x<size;x++){
		data[x*2]=x;
		data[x*2+1]=0;
	    }
	    for(x=0;x<size;x+=stride){
		cout << data[x*2] << ' ' << data[x*2+1] << ' ';
	    }
	    cout << '\n';
	    for(int ii=0;ii<r;ii++){
		fft1d_stride_float(data, n, stride, 1, &flops, &memrefs);
	    for(x=0;x<size;x+=stride){
		cout << data[x*2] << ' ' << data[x*2+1] << ' ';
	    }
	    cout << '\n';
		fft1d_stride_float(data, n, stride, -1, &flops, &memrefs);
	    }
	    for(x=0;x<size;x+=stride){
		data[x*2]/=n;
		if(Abs(data[x*2]) < 1.e-4)
		    data[x*2]=0;
		data[x*2+1]/=n;
		if(Abs(data[x*2+1]) < 1.e-4)
		    data[x*2+1]=0;
	    }
	    for(x=0;x<size;x+=stride){
		cout << data[x*2] << ' ' << data[x*2+1] << ' ';
	    }
	    cout << '\n';
	}
    } else if(d==1){
	if(stride==0){
	    for(int ii=0;ii<r;ii++){
		fft1d_float(data, n, 1, &flops, &memrefs);
		fft1d_float(data, n, -1, &flops, &memrefs);
	    }
	} else {
	    for(int ii=0;ii<r;ii++){
		fft1d_stride_float(data, n, stride, 1, &flops, &memrefs);
		fft1d_stride_float(data, n, stride, -1, &flops, &memrefs);
	    }
	}
    } else {
#if 0
	float* p=data;
	for(int y=0;y<n;y++){
	    for(int x=0;x<n;x++){
		*p++=x+y;
		*p++=0;//x-y;
	    }
	}
	p=data;
	for(y=0;y<n;y++){
	    for(int x=0;x<n;x++){
		cout << *p++ << ' ' << *p++ << ' ';
	    }
	    cout << '\n';
	}
	cout << '\n';
#endif
	for(int ii=0;ii<r;ii++){
	    fft2d_float(data, n, 1, &flops, &memrefs);
#if 0
	p=data;
	for(y=0;y<n;y++){
	    for(int x=0;x<n;x++){
		cout << *p++ << ' ' << *p++ << ' ';
	    }
	    cout << '\n';
	}
	cout << '\n';
#endif
	    fft2d_float(data, n, -1, &flops, &memrefs);
	}
#if 0
	p=data;
	float nn=n*n;
	for(y=0;y<n;y++){
	    for(int x=0;x<n;x++){
		if(Abs(*p) < 1.e-10)
		    *p=0;
		p++;
		if(Abs(*p) < 1.e-10)
		    *p=0;
		p++;
	    }
	}
	p=data;
	for(y=0;y<n;y++){
	    for(int x=0;x<n;x++){
		cout << *p++/nn << ' ' << *p++/nn << ' ';
	    }
	    cout << '\n';
	}
#endif
    }
    timer.stop();
    cout << flops << " floating point operations in " << timer.time()<< " seconds\n";
    cout << timer.time()*1000/r << " ms/iteration (fft/ifft pair)\n";
    cout << (double)flops*1.e-6/timer.time() << " MFLOPS\n";
    cout << (double)size*2*sizeof(float)/1024/1024 << " Megabytes in dataset\n";
    cout << (double)memrefs/1024/1024 << " Megabytes accessed\n";
    cout << (double)memrefs/1024/1024/timer.time() << " MB/sec\n";
}

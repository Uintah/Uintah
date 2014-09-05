/*
 *  BrickArray3.cc: Implementation of dynamic bricked 3D array
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1998
 *
 *  Copyright (C) 1994 SCI Group
 */

template<class T>
BrickArray3<T>::BrickArray3()
{
    objs=0;
    data=0;
    dm1=dm2=dm3=0;
    totaldm1=totaldm2=totaldm3=0;
    idx1=idx2=idx3=0;
    L1=L2=0;
}

template<class T>
void BrickArray3<T>::allocate()
{
    if(dm1==0 || dm2==0 || dm3==0){
	objs=0;
	data=0;
	refcnt=0;
	dm1=dm2=dm3=0;
	totaldm1=totaldm2=totaldm3=0;
	idx1=idx2=idx3=0;
	L1=L2=0;
	return;
    }
    
    // 128 byte cache line
    L1=(int)(pow(128./sizeof(T), 1./3.)+.1);
    // 16K page size
    L2=(int)(pow(16384./double(sizeof(T)*L1*L1*L1), 1./3.)+.1);
    cerr << "sizeof(T)=" << sizeof(T) << '\n';
    cerr << "L1=" << L1 << ", L2=" << L2 << '\n';
    int totalx=(dm1+L2*L1-1)/(L2*L1);
    int totaly=(dm2+L2*L1-1)/(L2*L1);
    int totalz=(dm3+L2*L1-1)/(L2*L1);

    idx1=new int[dm1];

    for(int x=0;x<dm1;x++){
	int m1x=x%L1;
	int xx=x/L1;
	int m2x=xx%L2;
	int m3x=xx/L2;
	idx1[x]=m3x*totaly*totalz*L2*L2*L2*L1*L1*L1+m2x*L2*L2*L1*L1*L1+m1x*L1*L1;
    }
    idx2=new int[dm2];
    for(int y=0;y<dm2;y++){
	int m1y=y%L1;
	int yy=y/L1;
	int m2y=yy%L2;
	int m3y=yy/L2;
	idx2[y]=m3y*totalz*L2*L2*L2*L1*L1*L1+m2y*L2*L1*L1*L1+m1y*L1;
    }
    idx3=new int[dm3];
    for(int z=0;z<dm3;z++){
	int m1z=z%L1;
	int zz=z/L1;
	int m2z=zz%L2;
	int m3z=zz/L2;
	idx3[z]=m3z*L2*L2*L2*L1*L1*L1+m2z*L1*L1*L1+m1z;
    }

    totaldm1=totalx*L2*L1;
    totaldm2=totaly*L2*L1;
    totaldm3=totalz*L2*L1;
    int totalsize=totaldm1*totaldm2*totaldm3;
    cerr << "totalsize=" << totalsize*sizeof(T) << '\n';
    //objs=new T[totalsize];
    data=new char[totalsize*sizeof(T)+128+4096];
    unsigned long off=(unsigned long)data%128;
    if(off)
	objs=(T*)(data+128-off);
    else
	objs=(T*)data;
    refcnt=new int;
    *refcnt=1;
}

template<class T>
void BrickArray3<T>::resize(int d1, int d2, int d3)
{
    if(objs && dm1==d2 && dm2==d2 && dm3==d3)return;
    dm1=d1;
    dm2=d2;
    dm3=d3;
    if(objs){
	(*refcnt--);
	if(*refcnt == 0){
	    delete[] data;
	    delete[] idx1;
	    delete[] idx2;
	    delete[] idx3;
	    delete refcnt;
	}
    }
    allocate();
}

template<class T>
BrickArray3<T>::BrickArray3(int dm1, int dm2, int dm3)
: dm1(dm1), dm2(dm2),dm3(dm3)
{
    allocate();
}

template<class T>
BrickArray3<T>::~BrickArray3()
{
    if(objs){
	if(*refcnt == 0){
	    delete[] data;
	    delete[] idx1;
	    delete[] idx2;
	    delete[] idx3;
	    delete refcnt;
	}
    }
}

template<class T>
void BrickArray3<T>::initialize(const T& t)
{
    int n=totaldm1*totaldm2*totaldm3;
    for(int i=0;i<n;i++)
	objs[i]=t;
}

template<class T>
void BrickArray3<T>::share(const BrickArray3<T>& copy)
{
    if(objs){
	(*refcnt--);
	if(*refcnt == 0){
	    delete[] data;
	    delete[] idx1;
	    delete[] idx2;
	    delete[] idx3;
	}
    }
    objs=copy.objs;
    data=copy.data;
    refcnt=copy.refcnt;
    dm1=copy.dm1;
    dm2=copy.dm2;
    dm3=copy.dm3;
    totaldm1=copy.totaldm1;
    totaldm2=copy.totaldm2;
    totaldm3=copy.totaldm3;
    idx1=copy.idx1;
    idx2=copy.idx2;
    idx3=copy.idx3;
    L1=copy.L1;
    L2=copy.L2;
    (*refcnt)++;
}

/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  Array3.h: Interface to dynamic 3D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Datatypes_Array3_h
#define SCI_Datatypes_Array3_h 1

#include <Core/Util/Assert.h>
#include <Core/Datatypes/Datatype.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

template<class T> class LockArray3;
template<class T> void Pio(Piostream& stream, LockArray3<T>& data);

template<class T>
class LockArray3:public Datatype {
    T*** objs;
    int dm1;
    int dm2;
    int dm3;
    void allocate();
public:
    LockArray3();
    LockArray3(const LockArray3&);
    LockArray3(int, int, int);
    LockArray3<T>& operator=(const LockArray3&);
    virtual ~LockArray3();
    inline T& operator()(int d1, int d2, int d3) const
	{
	    ASSERTL3(d1>=0 && d1<dm1);
	    ASSERTL3(d2>=0 && d2<dm2);
	    ASSERTL3(d3>=0 && d3<dm3);
	    return objs[d1][d2][d3];
	}
    inline int dim1() const {return dm1;}
    inline int dim2() const {return dm2;}
    inline int dim3() const {return dm3;}
    void resize(int, int, int);
    void initialize(const T&);

    T* get_onedim();
    void get_onedim_byte( unsigned char *v );

    inline T*** get_dataptr() {return objs;}

    friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, LockArray3<T>&);
    friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, LockArray3<T>*&);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace SCIRun

////////////////////////////////////////////////////////////
// Start of included LockArray3.cc

#include <Core/Datatypes/LockArray3.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Geometry/Point.h>

namespace SCIRun {

template<class T>
LockArray3<T>::LockArray3()
{
    objs=0;
}

template<class T>
void LockArray3<T>::allocate()
{
    objs=scinew T**[dm1];
    T** p=scinew T*[dm1*dm2];
    T* pp=scinew T[dm1*dm2*dm3];
    for(int i=0;i<dm1;i++){
	objs[i]=p;
	p+=dm2;
	for(int j=0;j<dm2;j++){
	    objs[i][j]=pp;
	    pp+=dm3;
	}
    }
}

template<class T>
void LockArray3<T>::resize(int d1, int d2, int d3)
{
    if(objs && dm1==d2 && dm2==d2 && dm3==d3)return;
    dm1=d1;
    dm2=d2;
    dm3=d3;
    if(objs){
	delete[] objs[0][0];
	delete[] objs[0];
	delete[] objs;
    }
    allocate();
}

template<class T>
LockArray3<T>::LockArray3(const LockArray3<T>& a)
: dm1(a.dm1), dm2(a.dm2), dm3(a.dm3)
{
    allocate();
}

template<class T>
LockArray3<T>::LockArray3(int dm1, int dm2, int dm3)
: dm1(dm1), dm2(dm2),dm3(dm3)
{
    allocate();
}

template<class T>
LockArray3<T>::~LockArray3()
{
    if(objs){
	delete[] objs[0][0];
	delete[] objs[0];
	delete[] objs;
    }
}

template<class T>
void LockArray3<T>::initialize(const T& t)
{
    ASSERT(objs != 0);
    for(int i=0;i<dm1;i++){
	for(int j=0;j<dm2;j++){
	    for(int k=0;k<dm3;k++){
		objs[i][j][k]=t;
	    }
	}
    }
}

template<class T>
T* LockArray3<T>::get_onedim()
{
  int i,j,k, index;
  T* a = scinew T[dm1*dm2*dm3];
  
  for( i=0; i<dm1; i++)
    for( j=0; j<dm2; j++ )
      for( k=0; k<dm3; k++ )
	a[index++] = objs[i][j][k];
}

template<class T>
void
LockArray3<T>::get_onedim_byte( unsigned char *v )
{
  int i,j,k, index;
  index = 0;
  
  for( k=0; k<dm3; k++ )
    for( j=0; j<dm2; j++ )
      for( i=0; i<dm1; i++)
	v[index++] = objs[i][j][k];
}

template<class T>
void LockArray3<T>::io(Piostream&)
{
    std::cerr << "Error - not implemented!\n";
}

// Put this in a specialization file... Dd
//PersistentTypeID LockArray3<Point>::type_id("LockArray3", "Datatype", 0);

#define LockArray3_VERSION 1

template<class T>
void Pio(Piostream& stream, LockArray3<T>& data)
{
    /*int version=*/stream.begin_class("LockArray3", LockArray3_VERSION);
    if(stream.reading()){
	// Allocate the array...
	int d1, d2, d3;
	Pio(stream, d1);
	Pio(stream, d2);
	Pio(stream, d3);
	data.resize(d1, d2, d3);
    } else {
	Pio(stream, data.dm1);
	Pio(stream, data.dm2);
	Pio(stream, data.dm3);
    }
    for(int i=0;i<data.dm1;i++){
	for(int j=0;j<data.dm2;j++){
	    for(int k=0;k<data.dm3;k++){
		Pio(stream, data.objs[i][j][k]);
	    }
	}
    }
    stream.end_class();
}

template<class T>
void Pio(Piostream& stream, LockArray3<T>*& data) {
    if (stream.reading()) {
	data=scinew LockArray3<T>;
    }
    Pio(stream, *data);
}

} // End namespace SCIRun


#endif

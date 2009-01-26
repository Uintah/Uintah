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


/*
 *  Array3.cc: Implementation of dynamic 3D array class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Util/Assert.h>

namespace rtrt {
  
template<class T>
Array3<T>::Array3()
{
    objs=0;
    dm1=dm2=dm3=0;
}

template<class T>
void Array3<T>::allocate()
{
    if(dm1==0 || dm2==0 || dm3==0){
	objs=0;
	refcnt=0;
	return;
    }
    objs=new T**[dm1];
    T** p=new T*[dm1*dm2];
    T* pp=new T[dm1*dm2*dm3];
    for(int i=0;i<dm1;i++){
        objs[i]=p;
        p+=dm2;
        for(int j=0;j<dm2;j++){
            objs[i][j]=pp;
            pp+=dm3;
        }
    }
    refcnt=new int;
    *refcnt=1;
}

template<class T>
void Array3<T>::resize(int d1, int d2, int d3)
{
    if(objs && dm1==d1 && dm2==d2 && dm3==d3)return;
    dm1=d1;
    dm2=d2;
    dm3=d3;
    if(objs){
	(*refcnt)--;
	if(*refcnt == 0){
	    delete[] objs[0][0];
	    delete[] objs[0];
	    delete[] objs;
	    delete refcnt;
	}
    }
    allocate();
}

template<class T>
Array3<T>::Array3(int dm1, int dm2, int dm3)
: dm1(dm1), dm2(dm2),dm3(dm3)
{
    allocate();
}

template<class T>
Array3<T>::~Array3()
{
    if(objs){
	(*refcnt)--;
	if(*refcnt == 0){
	    delete[] objs[0][0];
	    delete[] objs[0];
	    delete[] objs;
	    delete refcnt;
	}
    }
}

template<class T>
void Array3<T>::initialize(const T& t)
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
void Array3<T>::share(const Array3<T>& copy)
{
    if(objs){
	(*refcnt)--;
	if(*refcnt == 0){
	    delete[] objs[0][0];
	    delete[] objs[0];
	    delete[] objs;
	    delete refcnt;
	}
    }
    objs=copy.objs;
    refcnt=copy.refcnt;
    dm1=copy.dm1;
    dm2=copy.dm2;
    dm3=copy.dm3;
    (*refcnt)++;
}

} // end namespace rtrt

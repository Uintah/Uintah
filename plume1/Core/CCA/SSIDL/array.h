/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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
 *  array.h: SSIDL array classes
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Component_SIDL_array_h
#define Component_SIDL_array_h

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SSIDL {
    template<class T> class array1 : public std::vector<T> {
    public:
	array1() : std::vector<T>(0) {}
	array1(const array1<T>& copy) : std::vector<T>(copy) {}
	array1(const std::vector<T>& copy) : std::vector<T>(copy) {}
	array1(unsigned long s) : std::vector<T>(s) {}
	const T* buffer() const{return &(*this)[0];}
    };
    template<class T> class array2 {
    public:
	typedef T* pointer;
	typedef T* const_pointer;
	typedef T* iterator;
	typedef const T* const_iterator;
	typedef size_t size_type;

	array2() {}
	~array2() {}
	array2(size_type size1, size_type size2) : s1(size1), s2(size2), data(size1*size2), rows(size1) {
	    alloc();
	}
	array2(const array2<T>& copy) : s1(copy.s1), s2(copy.s2), data(copy.data) {
	    rows.resize(s1);
	    int s=0;
	    for(unsigned int i=0;i<s1;i++){
		rows[i]=&data[s];
		s+=s2;
	    }
	}
	array2<T>& operator=(const array2<T>& copy) {
	    resize(copy.s1, copy.s2);
	    data=copy.data;
	    int s=0;
	    for(unsigned int i=0;i<s1;i++){
		rows[i]=&data[s];
		s+=s2;
	    }
	    return *this;
	}

	size_type size1() const {
	    return s1;
	}
	size_type size2() const {
	    return s2;
	}

	void resize(size_type ns1, size_type ns2) {
	    s1=ns1;
	    s2=ns2;
	    alloc();
	}

	pointer operator[](int row) {
	  return rows[row];
	}
	const_pointer operator[](int row) const {
	  return rows[row];
	}

	iterator begin() { return data.begin(); }
	const_iterator begin() const { return data.begin(); }

	iterator end() { return data.end(); }
	const_iterator end() const { return data.end(); }
	T* buffer(){return &data[0];}
    private:
	size_type s1;
	size_type s2;
	std::vector<T> data;
	std::vector<T*> rows;

	void alloc() {
	    int s=0;
	    if(data.size() != s1*s2)
		data.resize(s1*s2);
	    if(rows.size() != s1)
		rows.resize(s1);
	    for(unsigned int i=0;i<s1;i++){
		rows[i]=&data[s];
		s+=s2;
	    }
	}
    };
} // End namespace SSIDL 
#endif





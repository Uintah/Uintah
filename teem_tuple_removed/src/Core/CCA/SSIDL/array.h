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





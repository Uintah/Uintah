
#ifndef UINTAH_HOMEBREW_Array3Index_H
#define UINTAH_HOMEBREW_Array3Index_H

#include <iosfwd>

class Array3Index {
public:
    inline Array3Index(int i, int j, int k)
	: ii(i), jj(j), kk(k)
    {
    }
    inline Array3Index()
    {
    }
    inline ~Array3Index()
    {
    }
    inline Array3Index(const Array3Index& copy)
	: ii(copy.ii), jj(copy.jj), kk(copy.kk)
    {
    }
    inline Array3Index& operator=(const Array3Index& copy)
    {
	ii=copy.ii;
	jj=copy.jj;
	kk=copy.kk;
	return *this;
    }
    inline int i() const {
	return ii;
    }
    inline int j() const {
	return jj;
    }
    inline int k() const {
	return kk;
    }

private:
    int ii, jj, kk;
};

std::ostream& operator<<(std::ostream&, const Array3Index&);

#endif

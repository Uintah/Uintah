
#ifndef UV_H
#define UV_H 1

namespace rtrt {

class UV {
    double _u, _v;
public:
    inline UV() {}
    inline UV(double u, double v) : _u(u), _v(v) {}
    inline UV(const UV& copy) : _u(copy._u), _v(copy._v) {}
    inline UV& operator=(const UV& copy) {
	_u=copy._u;
	_v=copy._v;
	return *this;
    }
    inline ~UV() {};
    inline double u() {return _u;}
    inline double v() {return _v;}
    inline void set(double u, double v) {
	_u=u;
	_v=v;
    }
};

} // end namespace rtrt

#endif


#ifndef Math_Noise_h
#define Math_Noise_h 1

namespace rtrt {

class Point;
class Vector;

class Noise {
protected:
    int tablesize;
    int bitmask;
    double* noise_tab;
    int* scramble_tab;
    int get_index(int,int,int);
    double lattice(int,int,int);
public:
    Noise(int=0, int=4096);
    Noise(const Noise&);
    virtual ~Noise();
    double operator()(const Vector&);
    double operator()(double);
    Vector dnoise(const Vector&);
    Vector dnoise(const Vector&, double&);
};

} // end namespace rtrt

#endif

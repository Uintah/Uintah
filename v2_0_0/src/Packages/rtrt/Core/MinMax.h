
#ifndef MINMAX_H
#define MINMAX_H 1

namespace rtrt {

inline double Min(double x, double y)
{
    return x<y?x:y;
}

inline double Max(double x, double y)
{
    return x>y?x:y;
}

inline double Min(double x, double y, double z)
{
    return x<y?x<z?x:z:y<z?y:z;
}

inline double Max(double x, double y, double z)
{
    return x>y?x>z?x:z:y>z?y:z;
}

inline float Min(float x, float y)
{
    return x<y?x:y;
}

inline float Max(float x, float y)
{
    return x>y?x:y;
}

inline short Min(short x, short y)
{
    return x<y?x:y;
}

inline short Max(short x, short y)
{
    return x>y?x:y;
}

inline int Min(int x, int y)
{
    return x<y?x:y;
}

inline int Max(int x, int y)
{
    return x>y?x:y;
}

} // end namespace rtrt

#endif

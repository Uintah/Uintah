#pragma once

#include"Typedefs.h"
#include"Utils.h"
//bounding box class
//assuming here that bounding box is defined by 
//the "bottom-left" and "top-right" vectors given
//as from and to correspondingly
class BoundingBox
{
public:
    BoundingBox(){};
    BoundingBox(const Vec2D& a, const Vec2D& b);
    //copy constructor
    BoundingBox(const BoundingBox& b);
    //assignment operator
    const BoundingBox& operator=(const BoundingBox& b);
    bool IsValid();
    bool Contains(const Vec2D& v)const;
    bool Contains(const BoundingBox& b) const;
    bool IntersectsWith(const BoundingBox& b);
    bool StrictlyContains(const Vec2D& v);
    Vec2D& SetFrom(){return from;}
    Vec2D& SetTo(){return to;}
    const Vec2D& GetFrom() const {return from;}
    const Vec2D& GetTo() const {return to;}
    ostream& operator<<(ostream& os);
private:
    Vec2D from;
    Vec2D to;
};

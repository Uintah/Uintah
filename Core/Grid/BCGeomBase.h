#ifndef UINTAH_GRID_BCGeomBase_H
#define UINTAH_GRID_BCGeomBase_H

#include <Packages/Uintah/Core/Grid/BCData.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <typeinfo>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  using SCIRun::IntVector;
  using SCIRun::Point;
   
/**************************************

CLASS
   BCGeomBase
   
   
GENERAL INFORMATION

   BCGeomBase.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   BCGeomBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class BCGeomBase {
  public:
    BCGeomBase();
    virtual ~BCGeomBase();    
    virtual BCGeomBase* clone() = 0;
    virtual void getBCData(BCData& bc) const = 0;
    virtual void addBCData(BCData& bc)  = 0;
    virtual void addBC(BoundCondBase* bc)  = 0;
    void setBoundaryIterator(std::vector<IntVector>& b);
    void setNBoundaryIterator(std::vector<IntVector>& b);
    void setInteriorIterator(std::vector<IntVector>& i);
    void setSFCXIterator(std::vector<IntVector>& i);
    void setSFCYIterator(std::vector<IntVector>& i);
    void setSFCZIterator(std::vector<IntVector>& i);
    void getBoundaryIterator(std::vector<IntVector>& b) const;
    void getNBoundaryIterator(std::vector<IntVector>& b) const;
    void getInteriorIterator(std::vector<IntVector>& i) const;
    void getSFCXIterator(std::vector<IntVector>& i) const;
    void getSFCYIterator(std::vector<IntVector>& i) const;
    void getSFCZIterator(std::vector<IntVector>& i) const;
    virtual bool inside(const Point& p) const = 0;

  protected:
    std::vector<IntVector> boundary,interior,sfcx,sfcy,sfcz,nboundary;
  };

  bool cmp_type(BCGeomBase* p);
  template<class T> bool cmp_type(BCGeomBase* p) 
    {
      return (typeid(T) == typeid(*p));
    };
  
} // End namespace Uintah

#endif

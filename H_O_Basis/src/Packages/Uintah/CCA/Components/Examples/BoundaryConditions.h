
#ifndef Packages_Uintah_CCA_Components_Examples_BoundaryConditions_h
#define Packages_Uintah_CCA_Components_Examples_BoundaryConditions_h

#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <sgi_stl_warnings_off.h>
#include <map>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424
#endif

namespace Uintah {
  class GeometryPiece;
  class RegionDB;
  using namespace SCIRun;
  struct BC {
    enum Type {
      FreeFlow, FixedRate, FixedValue, FixedFlux, CoarseGrid, Exterior
    };
    BC() {} // Shutup warning about all member function being private...
    private:
    BC(const BC&);
    BC& operator=(const BC&);
  };

  class BCRegionBase {
  public:
  protected:
    BCRegionBase(const IntVector& offset, const GeometryPiece* piece,
		 BC::Type type)
      : piece(piece), offset(offset), type(type)
      {
      }
    virtual ~BCRegionBase();
    friend class ConditionBase;

    const GeometryPiece* piece;
    IntVector offset;
    int idx;
    BC::Type type;

    void set(NCVariable<int>& bctype, const Patch*, int mask,
	     const IntVector& topoffset);
    void getRange(const Patch* patch, IntVector& l, IntVector& h,
		  const IntVector& topoffset);

  private:
    BCRegionBase(const BCRegionBase&);
    BCRegionBase& operator=(const BCRegionBase&);
  };

  template<class T>
    class Condition;
  template<class T>
    class BCRegion : public BCRegionBase {
    public:
      T getValue() const {
	return value;
      }
      BC::Type getType() const {
	return type;
      }
    private:
      T value;
      int idx;
      BCRegion(const IntVector& offset, BC::Type type)
	: BCRegionBase(offset, 0, type), value(0)
	{
	  ASSERT(type == BC::FreeFlow || type == BC::CoarseGrid || type == BC::Exterior);
	}
      BCRegion(const IntVector& offset, const GeometryPiece* piece,
	       BC::Type type, T value)
	: BCRegionBase(offset, piece, type), value(value)
	{
	}

      friend class Condition<T>;
      void apply(Array3<T>& field, const Patch* patch) {
	IntVector l,h;
	getRange(patch, l, h, offset);
	if(piece){
	  const Level* level = patch->getLevel();
	  Vector celloffset((Vector(0.5, 0.5, 0.5)-offset.asVector()*0.5)*patch->dCell());
	  for(CellIterator iter(l, h); !iter.done(); iter++){
	    Point p(level->getNodePosition(*iter)+celloffset);
	    if(piece->inside(p))
	      field[*iter] = value;
	  }
	} else {
	  for(CellIterator iter(l, h); !iter.done(); iter++){
	    field[*iter] = value;
	  }
	}
      }
	
      BCRegion(const BCRegion<T>&);
      BCRegion<T>& operator=(const BCRegion<T>&);
  };

  class ConditionBase {
  public:
    bool getHasGlobalValue() const {
      return hasGlobalValue;
    }
  protected:
    ConditionBase(const IntVector& offset);
    virtual ~ConditionBase();
    friend class InitialConditions;
    friend class BoundaryConditions;
    IntVector offset;
    bool hasGlobalValue;
    int mask, shift;
    void setupMasks(int& shift);
    void set(NCVariable<int>& bctype, const Patch* patch,
	     const IntVector& topoffset);
    virtual int numRegions() = 0;
    virtual BCRegionBase* getRegion(int region) = 0;
    virtual void parseGlobal(ProblemSpecP& node) = 0;
    virtual void parseCondition(ProblemSpecP&, const GeometryPiece*,
				BC::Type bctype, bool valueRequired) = 0;
    virtual void merge(int& bc1, int bc2, bool pressure) = 0;

  private:
    ConditionBase(const ConditionBase&);
    ConditionBase& operator=(const ConditionBase&);
  };
  
  template<class T>
  class Condition : public ConditionBase {
  public:
    BCRegion<T>* get(int bctype) {
      return regions[(bctype&mask)>>shift];
    }
    void apply(Array3<T>& field, const Patch* patch) {
      for(typename std::vector<BCRegion<T>*>::iterator iter = regions.begin();
	  iter != regions.end(); iter++)
	(*iter)->apply(field, patch);
    }

    virtual void parseGlobal(ProblemSpecP& node) {
      // Find the value
      node->require("value", regions[0]->value);
      hasGlobalValue=true;
    }
    virtual void parseCondition(ProblemSpecP& node, const GeometryPiece* piece,
				BC::Type bctype, bool valueRequired) {
      T value;
      if(valueRequired) {
	node->require("value", value);
      } else {
	node->get("value", value);
      }
      addRegion(new BCRegion<T>(offset, piece, bctype, value));
    }
    virtual void merge(int& bc1, int bc2, bool pressure) {
      if(!mask)
	return;
      int b1=(bc1&mask)>>shift;
      int b2=(bc2&mask)>>shift;
      BCRegion<T>* r1=regions[b1];
      BCRegion<T>* r2=regions[b2];
      if(pressure){
	if(r1->type == BC::CoarseGrid){
	  if(r2->type == BC::Exterior){
	    ASSERTFAIL("Bad BC combo 1");
	  } else if(r2->type != BC::CoarseGrid) {
	    bc1 = (bc1&~mask)|(bc2&mask);
	  }
	} else if(r1->type == BC::Exterior){
	  if(r2->type == BC::CoarseGrid) {
	    ASSERTFAIL("Bad BC combo 2");
	  } else if(r2->type != BC::Exterior) {
	    bc1 = (bc1&~mask)|(bc2&mask);
	  }
	} else if(r1->type == BC::FreeFlow) {
	  if(r2->type == BC::CoarseGrid || r2->type == BC::Exterior) {
	    bc1 = (bc1&~mask)|(bc2&mask);
	  } else if(r2->type != BC::FreeFlow){
	    ASSERTFAIL("Bad BC combo 3");
	  }
	}  else if(r2->type == BC::CoarseGrid && r1->type != BC::FreeFlow){
	  // Leave bc1...
	} else if(r2->type == BC::Exterior && r1->type != BC::FreeFlow){
	  // Leave bc1...
	} else {
	  if(r1->type != BC::FreeFlow && r2->type != BC::FreeFlow){
	    ASSERT(r1 == r2);
	  }
	}
      } else {
	if(r1->type == BC::CoarseGrid){
	  if(r2->type == BC::Exterior){
	    ASSERTFAIL("Bad BC combo 1");
	  } else if(r2->type != BC::CoarseGrid) {
	    bc1 = (bc1&~mask)|(bc2&mask);
	  }
	} else if(r1->type == BC::Exterior){
	  if(r2->type == BC::CoarseGrid) {
	    ASSERTFAIL("Bad BC combo 2");
	  } else if(r2->type != BC::Exterior) {
	    bc1 = (bc1&~mask)|(bc2&mask);
	  }
	}  else if(r2->type == BC::CoarseGrid){
	  // Leave bc1...
	} else if(r2->type == BC::Exterior){
	  // Leave bc1...
	} else {
	  if(r1->type != BC::FreeFlow && r2->type != BC::FreeFlow){
	    ASSERT(r1 == r2);
	  }
	}
      }
    }
  protected:
    Condition(const IntVector& offset, bool addBoundaries)
      : ConditionBase(offset) {
      // Global is always region #0
      addRegion(new BCRegion<T>(offset, BC::FreeFlow));
      if(addBoundaries){
	addRegion(new BCRegion<T>(offset, BC::Exterior));
	addRegion(new BCRegion<T>(offset, BC::CoarseGrid));
      }
    }
    virtual ~Condition() {
      for(typename vector<BCRegion<T>*>::iterator iter = regions.begin(); iter != regions.end(); iter++)
	delete *iter;
    }

    friend class InitialConditions;
    friend class BoundaryConditions;
    void setGlobalValue(T newval) {
      ASSERT(!hasGlobalValue);
      regions[0]->value = newval;
      hasGlobalValue=true;
    }
    void addRegion(BCRegion<T>* newreg) {
      regions.push_back(newreg);
    }

    int numRegions() {
      return (int)regions.size();
    }
    BCRegionBase* getRegion(int region){
      return regions[region];
    }
  private:
    std::vector<BCRegion<T>*> regions;
    Condition(const Condition<T>&);
    Condition<T>& operator=(const Condition<T>&);
  };

  class InitialConditions {
  public:
    template<class T>
      Condition<T>* getCondition(const std::string& name) {
      return dynamic_cast<Condition<T>*>(conditions.find(name)->second);
    }
    template<class T>
      void setupCondition(const std::string& name, const IntVector& offset) {
      setupCondition(name, new Condition<T>(offset, false));
    }

    InitialConditions();
    ~InitialConditions();
    void problemSetup(ProblemSpecP&, const RegionDB& regiondb);
    bool allHaveGlobalValues(string& missing) const;

  private:
    void setupCondition(const std::string& name, ConditionBase* cond);
    typedef map<string, ConditionBase*> MapType;
    MapType conditions;

    InitialConditions(const InitialConditions&);
    InitialConditions& operator=(const InitialConditions&);
  };

  class BoundaryConditions {
  public:
    template<class T>
      Condition<T>* getCondition(const std::string& name,
				 Patch::VariableBasis basis) {
      return dynamic_cast<Condition<T>*>(conditions.find(make_pair(name, basis))->second);
    }
    template<class T>
      void setupCondition(const std::string& name, const IntVector& offset) {
      setupCondition(name, new Condition<T>(offset, offset==IntVector(0,0,0)),
		     Patch::CellBased);
      setupCondition(name, new Condition<T>(offset, offset==IntVector(1,1,1)),
		     Patch::NodeBased);
      setupCondition(name, new Condition<T>(offset, offset==IntVector(1,0,0)),
		     Patch::XFaceBased);
      setupCondition(name, new Condition<T>(offset, offset==IntVector(0,1,0)),
		     Patch::YFaceBased);
      setupCondition(name, new Condition<T>(offset, offset==IntVector(0,0,1)),
		     Patch::ZFaceBased);
    }
    BoundaryConditions();
    ~BoundaryConditions();
    void problemSetup(ProblemSpecP&, const RegionDB& regiondb);
    void set(NCVariable<int>& bctype, const Patch* patch);
    void merge(int& bc1, int bc2);
  private:
    void setupCondition(const std::string& name, ConditionBase* cond,
			Patch::VariableBasis basis);
    typedef map<pair<string, Patch::VariableBasis>, ConditionBase*> MapType;
    MapType conditions;

    void setupMasks();

    BoundaryConditions(const BoundaryConditions&);
    BoundaryConditions& operator=(const BoundaryConditions&);
  };
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#endif

#endif

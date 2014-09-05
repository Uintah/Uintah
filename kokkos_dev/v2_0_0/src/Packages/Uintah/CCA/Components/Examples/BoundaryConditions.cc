
#include <Packages/Uintah/CCA/Components/Examples/BoundaryConditions.h>
#include <Packages/Uintah/CCA/Components/Examples/RegionDB.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Math/MiscMath.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

BoundaryConditions::BoundaryConditions()
{
}

BoundaryConditions::~BoundaryConditions()
{
  for(MapType::iterator iter = conditions.begin(); iter != conditions.end(); iter++)
    delete iter->second;
}

void BoundaryConditions::problemSetup(ProblemSpecP& ps, const RegionDB& regiondb)
{
  ProblemSpecP conds = ps->findBlock("BoundaryConditions");
  if(!conds)
    throw ProblemSetupException("BoundaryConditions not found");
  for(ProblemSpecP child = conds->findBlock("Condition"); child != 0;
      child = child->findNextBlock("Condition")){
    string var;
    if(!child->getAttribute("var", var)){
      if(!child->getAttribute("variable", var)){
	throw ProblemSetupException("variable attribute not found for boundary condition");
      }
    }
    Patch::VariableBasis begin_basis, end_basis;
    string atname;
    child->require("applyAt", atname);
    if(atname == "cell" || atname == "Cell"){
      begin_basis = end_basis = Patch::CellBased;
    } else if(atname == "node" || atname == "Node"){
      begin_basis = end_basis = Patch::NodeBased;
    } else if(atname == "xface" || atname == "XFace"){
      begin_basis = end_basis = Patch::XFaceBased;
    } else if(atname == "yface" || atname == "YFace"){
      begin_basis = end_basis = Patch::YFaceBased;
    } else if(atname == "zface" || atname == "ZFace"){
      begin_basis = end_basis = Patch::ZFaceBased;
    } else if(atname == "allfaces" || atname == "AllFaces"){
      begin_basis = Patch::XFaceBased;
      end_basis = Patch::ZFaceBased;
    } else {
      throw ProblemSetupException("Boundary condition cannot be applied at: "+atname);
    }

    for(Patch::VariableBasis basis = begin_basis; basis <= end_basis;
	basis = Patch::VariableBasis(int(basis)+1)){
      MapType::iterator iter = conditions.find(make_pair(var, basis));
      if(iter == conditions.end())
	throw ProblemSetupException("Boundary condition set for unknown variable: "+var);
    }
    
    ProblemSpecP c = child->findBlock("region");
    if(c == 0){
      // This is a global value...
      throw ProblemSetupException("Boundary condition requires a region");
    } else {
      for(; c != 0; c = c->findNextBlock("region")){
	string name;
	if(!c->getAttribute("name", name))
	  throw ProblemSetupException("Region does not have a name");
	const GeometryPiece* piece = regiondb.getObject(name);
	if(!piece)
	  throw ProblemSetupException("Unknown piece: "+name);

	string type;
	child->require("type", type);
	BC::Type bctype;
	bool valueRequired=true;
	if(type == "FixedValue" || type == "fixedvalue"){
	  bctype=BC::FixedValue;
	} else if(type == "FixedRate" || type == "fixedrate"){
	  bctype=BC::FixedRate;
	} else if(type == "FixedFlux" || type == "fixedflux"){
	  bctype=BC::FixedFlux;
	} else if(type == "FreeFlow" || type == "freeflow"){
	  bctype=BC::FreeFlow;
	  valueRequired=false;
	} else {
	  throw ProblemSetupException("Unknown BC type: "+type);
	}

	for(Patch::VariableBasis basis = begin_basis; basis <= end_basis;
	    basis = Patch::VariableBasis(int(basis)+1)){
	  MapType::iterator iter = conditions.find(make_pair(var, basis));
	  iter->second->parseCondition(child, piece, bctype, valueRequired);
	}
      }
    }
  }
  setupMasks();
}

void BoundaryConditions::set(NCVariable<int>& bctype, const Patch* patch)
{
  bctype.initialize(0);
  for(MapType::iterator iter = conditions.begin(); iter != conditions.end(); iter++){
    IntVector centeroffset;
    switch(iter->first.second){
    case Patch::CellBased:
      centeroffset=IntVector(0,0,0);
      break;
    case Patch::NodeBased:
      centeroffset=IntVector(1,1,1);
      break;
    case Patch::XFaceBased:
      centeroffset=IntVector(1,0,0);
      break;
    case Patch::YFaceBased:
      centeroffset=IntVector(0,1,0);
      break;
    case Patch::ZFaceBased:
      centeroffset=IntVector(0,0,1);
      break;
    default:
      throw InternalError("Unknown BC basis");
    }
#if 0
    cerr << "\n\ncondition: " << iter->first.first << '\n';
#endif
    iter->second->set(bctype, patch, centeroffset);
  }
}

void BoundaryConditions::setupMasks()
{
  int shift=0;
  for(MapType::iterator iter = conditions.begin(); iter != conditions.end(); iter++){
    iter->second->setupMasks(shift);
  }
  cerr << "total bits: " << shift << '\n';
  ASSERT(shift < 32);  // Need to use a bigger int!
}

InitialConditions::InitialConditions()
{
}

InitialConditions::~InitialConditions()
{
  for(MapType::iterator iter = conditions.begin(); iter != conditions.end(); iter++)
    delete iter->second;
}

void InitialConditions::problemSetup(ProblemSpecP& ps, const RegionDB& regiondb)
{
  ProblemSpecP conds = ps->findBlock("InitialConditions");
  if(!conds)
    throw ProblemSetupException("InitialConditions not found");
  for(ProblemSpecP child = conds->findBlock("Condition"); child != 0;
      child = child->findNextBlock("Condition")){
    string var;
    if(!child->getAttribute("var", var)){
      if(!child->getAttribute("variable", var)){
	throw ProblemSetupException("variable attribute not found for initial condition");
      }
    }
    MapType::iterator iter = conditions.find(var);
    if(iter == conditions.end())
      throw ProblemSetupException("Initial condition set for unknown variable: "+var);
    
    ProblemSpecP c = child->findBlock("region");
    if(c == 0){
      // This is a global value...
      iter->second->parseGlobal(child);
    } else {
      for(; c != 0; c = c->findNextBlock("region")){
	string name;
	if(!c->getAttribute("name", name))
	  throw ProblemSetupException("Region does not have a name");
	const GeometryPiece* piece = regiondb.getObject(name);
	if(!piece)
	  throw ProblemSetupException("Unknown piece: "+name);
	iter->second->parseCondition(child, piece, BC::FreeFlow, true);
      }
    }
  }
}

bool InitialConditions::allHaveGlobalValues(string& missing) const
{
  for(MapType::const_iterator iter = conditions.begin(); iter != conditions.end(); iter++)
    if(!iter->second->getHasGlobalValue()){
      missing=iter->first;
      return false;
    }
  return true;
}

void InitialConditions::setupCondition(const std::string& name,
				       ConditionBase* cond)
{
  if(conditions.find(name) != conditions.end()){
    // Already set up with this name!
    throw ProblemSetupException("Redundant initial condition setup: "+name);
  }
  conditions[name]=cond;
}

void BoundaryConditions::setupCondition(const std::string& name,
					ConditionBase* cond,
					Patch::VariableBasis basis)
{
  if(conditions.find(make_pair(name, basis)) != conditions.end()){
    // Already set up with this name!
    throw ProblemSetupException("Redundant boundary condition setup: "+name);
  }
  conditions[make_pair(name, basis)]=cond;
}

ConditionBase::ConditionBase(const IntVector& offset)
  : offset(offset)
{
  hasGlobalValue = false;
}

ConditionBase::~ConditionBase()
{
}

#if 0
static void printhex(const Array3<int>& d, int mask,
		     const char* name, int pre=0)
{
  IntVector l(d.getLowIndex());
  IntVector h(d.getHighIndex());
  cerr << name << ": " << l << "-" << h << "\n";
  pre+=l.x()*13;
  for(int k=h.z()-1;k>=l.z();k--){
    cerr << "k=" << k << '\n';
    for(int j=h.y()-1;j>=l.y();j--){
      for(int i=0;i<pre;i++)
	cerr << ' ';
      for(int i=l.x();i<h.x();i++){
	cerr << setw(13) << setbase(16) << (d[IntVector(i,j,k)]&mask);
      }
      cerr << '\n';
    }
  }
  cerr << setbase(10);
}
#endif

void ConditionBase::set(NCVariable<int>& bctype, const Patch* patch,
			const IntVector& centeroffset)
{
  IntVector totaloffset(centeroffset.x()|offset.x(),
			centeroffset.y()|offset.y(),
			centeroffset.z()|offset.z());
  int n=numRegions();
  for(int i=1;i<n;i++) // Skip global region
    getRegion(i)->set(bctype, patch, mask, totaloffset);
#if 0
  if(mask != 0){
    cerr << "\n\noffset=" << offset << ", centeroffset=" << centeroffset << '\n';
    printhex(bctype, mask, "bctype");
  }
#endif
}

void ConditionBase::setupMasks(int& shift)
{
  this->shift=shift;
  int n=numRegions()-1;
  int nbits=0;
  while(n){
    n>>=1;
    nbits++;
  }
  mask = (1<<nbits)-1;
  mask <<= shift;
  shift += nbits;
  n=numRegions();
  for(int i=0;i<n;i++)
    getRegion(i)->idx = i<<this->shift;
}

BCRegionBase::~BCRegionBase()
{
}

void BCRegionBase::getRange(const Patch* patch, IntVector& l, IntVector& h,
			    const IntVector& topoffset)
{
  IntVector low(patch->getCellLowIndex());
  IntVector high(patch->getCellHighIndex()+topoffset);
  high = Min(high, patch->getNodeHighIndex());
  if(!piece){
    l=low;
    h=high;
  } else {
    Box b(piece->getBoundingBox());
    Vector off(topoffset.asVector()*0.5*patch->dCell());
    Vector delta = (b.lower()-b.upper())*1.e-6;
    const Level* level = patch->getLevel();
    IntVector ll = level->getCellIndex(b.lower()-delta+off);
    IntVector hh = level->getCellIndex(b.upper()+delta+off);
    hh += topoffset+IntVector(1,1,1);
    l = Max(low, ll);
    h = Min(high, hh);
  }
}

void BCRegionBase::set(NCVariable<int>& bctype,
		       const Patch* patch, int mask,
		       const IntVector& topoffset)
{
  IntVector l,h;
  getRange(patch, l, h, topoffset);
  if(piece){
    const Level* level = patch->getLevel();
    Vector celloffset((Vector(0.5, 0.5, 0.5)-topoffset.asVector()*0.5)*patch->dCell());
    for(CellIterator iter(l, h); !iter.done(); iter++){
      Point p(level->getNodePosition(*iter)+celloffset);
      if(piece->inside(p))
	bctype[*iter] = (bctype[*iter]&~mask)|idx;
    }
  } else {
    if(type == BC::CoarseGrid){
      for(Patch::FaceType face = Patch::startFace;
	  face <= Patch::endFace; face=Patch::nextFace(face)){
	if(patch->getBCType(face) != Patch::Coarse)
	  continue;

	IntVector l,h;
	patch->getFace(face, IntVector(0,0,0), IntVector(1,1,1), l, h);
	if(face == Patch::xplus){
	  l.x(l.x()+topoffset.x());
	  h.x(h.x()+topoffset.x());
	} else if(face != Patch::xminus){
	  h.x(h.x()+topoffset.x());
	}
	if(face == Patch::yplus){
	  l.y(l.y()+topoffset.y());
	  h.y(h.y()+topoffset.y());
	} else if(face != Patch::yminus){
	  h.y(h.y()+topoffset.y());
	}
	if(face == Patch::zplus){
	  l.z(l.z()+topoffset.z());
	  h.z(h.z()+topoffset.z());
	} else if(face != Patch::zminus){
	  h.z(h.z()+topoffset.z());
	}
	switch(face){
	case Patch::xminus:
	case Patch::xplus:
	  l-=IntVector(0,1,1);
	  h+=IntVector(0,1,1);
	  break;
	case Patch::yminus:
	case Patch::yplus:
	  l-=IntVector(1,0,1);
	  h+=IntVector(1,0,1);
	  break;
	case Patch::zminus:
	case Patch::zplus:
	  l-=IntVector(1,1,0);
	  h+=IntVector(1,1,0);
	  break;
	default:
	  break;
	}
#if 0
	if(face != Patch::xminus && face != Patch::xplus && patch->getBCType(Patch::xplus) == Patch::Neighbor){
	  h-=IntVector(1,0,0);
	}
	if(face != Patch::yminus && face != Patch::yplus && patch->getBCType(Patch::yplus) == Patch::Neighbor){
	  h-=IntVector(0,1,0);
	}
	if(face != Patch::zminus && face != Patch::zplus && patch->getBCType(Patch::zplus) == Patch::Neighbor){
	  h-=IntVector(0,0,1);
	}
#endif
	l = Max(l, bctype.getLowIndex());
	h = Min(h, bctype.getHighIndex());
	for(CellIterator iter(l, h); !iter.done(); iter++)
	  bctype[*iter] = (bctype[*iter]&~mask)|idx;
      }
    } else if(type == BC::Exterior){
      for(Patch::FaceType face = Patch::startFace;
	  face <= Patch::endFace; face=Patch::nextFace(face)){
	if(patch->getBCType(face) != Patch::None)
	  continue;

	IntVector l,h;
	patch->getFace(face, IntVector(0,0,0), IntVector(1,1,1), l, h);
	if(face == Patch::xplus){
	  l.x(l.x()+topoffset.x());
	  h.x(h.x()+topoffset.x());
	} else if(face != Patch::xminus){
	  h.x(h.x()+topoffset.x());
	}
	if(face == Patch::yplus){
	  l.y(l.y()+topoffset.y());
	  h.y(h.y()+topoffset.y());
	} else if(face != Patch::yminus){
	  h.y(h.y()+topoffset.y());
	}
	if(face == Patch::zplus){
	  l.z(l.z()+topoffset.z());
	  h.z(h.z()+topoffset.z());
	} else if(face != Patch::zminus){
	  h.z(h.z()+topoffset.z());
	}
	switch(face){
	case Patch::xminus:
	case Patch::xplus:
	  l-=IntVector(0,1,1);
	  h+=IntVector(0,1,1);
	  break;
	case Patch::yminus:
	case Patch::yplus:
	  l-=IntVector(1,0,1);
	  h+=IntVector(1,0,1);
	  break;
	case Patch::zminus:
	case Patch::zplus:
	  l-=IntVector(1,1,0);
	  h+=IntVector(1,1,0);
	  break;
	default:
	  break;
	}
	l = Max(l, bctype.getLowIndex());
	h = Min(h, bctype.getHighIndex());
	for(CellIterator iter(l, h); !iter.done(); iter++)
	  bctype[*iter] = (bctype[*iter]&~mask)|idx;
      }
    } else {
      for(CellIterator iter(l, h); !iter.done(); iter++){
	bctype[*iter] = (bctype[*iter]&~mask)|idx;
      }
    }
  }
}

void BoundaryConditions::merge(int& bc1, int bc2)
{
  for(MapType::iterator iter = conditions.begin(); iter != conditions.end(); iter++){
    iter->second->merge(bc1, bc2, iter->first.second == Patch::CellBased);
  }
}  

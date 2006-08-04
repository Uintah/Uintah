#include "IR.h"

/*
 * This file contains all the methods implementing the static checks and
 * binding the maps in the IR. The checks focus around checking and binding 
 * command specified maps and creating self referencing maps for ports that 
 * are specified as both 'in' and 'out'. Arguments are checked and special 
 * IrMethodMaps are created for matching methods.
 */


void IR::staticCheck() {

  for(unsigned int i=0; i<maps.size(); i++) {
    ///////
    // Binding a map to in and out port
    IrPort* inPort = NULL;
    IrPort* outPort = NULL;
    IrPort* tempPort = NULL;

    for(unsigned int j=0; j<inPorts.size(); j++) {
      tempPort = inPorts[j]->findPort(maps[i]->inSymbol,NULL);
      if(tempPort != NULL) {
	if(inPort == NULL) {
	  inPort = tempPort;
	}
	else {
	  errHandler("Trying to map a port that exists multiple times as 'in' or 'out' port");
	}
      }      
    }
    
    for(unsigned int j=0; j<outPorts.size(); j++) {
      tempPort = outPorts[j]->findPort(maps[i]->outSymbol,NULL);
      if(tempPort != NULL) {
	if(outPort == NULL) {
	  outPort = tempPort;
	}
	else {
	  errHandler("Trying to map a port that exists multiple times as 'in' or 'out' port");
	}
      }
    }

    if((inPort==NULL)||(outPort==NULL)) {
      errHandler("Unable to find defined ports to satisfy map " + maps[i]->inSymbol + 
		 " --> " + maps[i]->outSymbol); 
    }

    maps[i]->inPort = inPort;
    maps[i]->outPort = outPort;

    maps[i]->staticCheck();
  }

  ////////
  //Create maps for inoutPorts
  for(unsigned int i=0; i<inoutPorts.size(); i++) {
    inoutPorts[i]->selfMap(this,NULL);
  }

}

void IrMap::staticCheck() {
  inPort->staticCheck(outPort,this);
}

void IrPort:: staticCheck(IrPort* p1, IrMap* map) {
  return IrMethodL->staticCheck(p1->IrMethodL, map);
}

void IrMethodList::staticCheck(IrMethodList* ml1, IrMap* map) {
  for(unsigned int i=0; i<IrMethods.size(); i++) {
    for(unsigned int j=0; j<ml1->IrMethods.size(); j++) { 
      
      std::string nameOne = IrMethods[i]->name;
      std::string nameTwo = ml1->IrMethods[j]->name;
      IrNameMap* matchmap = NULL;

      if(nameOne == nameTwo) {
	if(IrMethods[i]->compare(ml1->IrMethods[j],NULL,map)) {
	  IrMethodMap* mmap = new IrMethodMap(IrMethods[i],ml1->IrMethods[j]);
	  map->addMethodMap(mmap);
	  if(j+1 < ml1->IrMethods.size()) i = i+1;
	}
      }
      else if (map->existsNameMap(nameOne,nameTwo,&matchmap)) {  		
	if(IrMethods[i]->compare(ml1->IrMethods[j],matchmap,map)) {
	  IrMethodMap* mmap = new IrMethodMap(IrMethods[i],ml1->IrMethods[j]);
	  map->addMethodMap(mmap);
	  if(j+1 < ml1->IrMethods.size()) i = i+1;
	}
      }
    }
  } 
}


///////
//findPort methods:
IrPort* IrDefList::findPort(std::string portname, IrPackage* pkg) {
  IrPort* port = NULL; 
  IrPort* tempPort = NULL;

  for(unsigned int i=0; i<irDefs.size(); i++) {
    if((tempPort = irDefs[i]->findPort(portname,pkg)) != NULL)
      if(port == NULL)
	port = tempPort;
      else
	errHandler("Trying to map a port that exists multiple times as 'in' or 'out' port");
  }
  return port;
}

IrPort* IrPackage::findPort(std::string portname, IrPackage* pkg) {
  return IrPortL->findPort(portname,this);
}

IrPort* IrPort::findPort(std::string portname, IrPackage* pkg) {
  package = pkg;
  if(portname == name) 
    return this;
  else 
    return NULL;
}

/////////
//selfMap methods:
void IrPackage::selfMap(IR* ir, IrPackage* pkg) {
  IrPortL->selfMap(ir,this);
}

void IrPort::selfMap(IR* ir, IrPackage* pkg) {
  package = pkg;
  if(!ir->existsOmit(name)) {
    IrMap* map = new IrMap(name,name,new IrNameMapList(),NULL);
    map->inPort = this;
    map->outPort = this;
    for(unsigned int i=0; i<IrMethodL->IrMethods.size(); i++) {
      IrMethodMap* mmap = new IrMethodMap(IrMethodL->IrMethods[i],IrMethodL->IrMethods[i]);
      (map->methodMaps).push_back(mmap);
    }
    ir->addMap(map);
  }
}

void IrDefList::selfMap(IR* ir, IrPackage* pkg) {
  for(unsigned int i=0; i<irDefs.size(); i++) {
    irDefs[i]->selfMap(ir,pkg);
  }
}

////////
//compare ports:
bool IrMethod::compare(IrMethod* m1, IrNameMap* nmap, IrMap* map) {
  return ((retType == m1->retType)&&(irArgL->compare(m1->irArgL,nmap,map))); 
}

bool IrArgumentList::compare(IrArgumentList* al1, IrNameMap* nmap, IrMap* map) {
  bool allEqual = true;

  if(irArgs.size() != al1->irArgs.size()) return false;
  for(unsigned int i=0; i<irArgs.size(); i++) {
    allEqual = (allEqual)&&(irArgs[i]->compare(al1->irArgs[i],nmap,map));
  }
  return allEqual;
}

bool IrArgument::compare(IrArgument* a1, IrNameMap* nmap, IrMap* map) {
  if(mode == a1->mode) {
    if(type == a1->type)
      return true;
    
    //Check top level maps
    if(map->existsNameMap(type,a1->type,NULL)) {
      //Mapping... so exchange mapped type information
      a1->mappedType = type;
      mappedType = a1->type;
      return true;      
    }

    //If we have a specific map to check sublevel on, then do that
    if((nmap!=NULL)&&(nmap->existsNameMap(type,a1->type))) {
      //Mapping... so exchange mapped type information
      a1->mappedType = type;
      mappedType = a1->type;
      return true;
    }
  }
  return false;
} 

//////
// existsNameMap:
int IrMap::existsNameMap(std::string one, std::string two, IrNameMap** matchmap) {
  return nameMapL->existsNameMap(one,two,matchmap);
}

int IrNameMapList::existsNameMap(std::string one, std::string two, IrNameMap** matchmap) {
  for(int i=0; i<nameMaps.size(); i++) {
    if((one == nameMaps[i]->nameOne)&&(two == nameMaps[i]->nameTwo)) {
      if(matchmap != NULL) (*matchmap) = nameMaps[i];
      return 1;
    }
  }
  //Check the general forall map in IR
  if((parentmap->irptr != NULL)&&(parentmap->irptr->getForAllMap())) {
    return parentmap->irptr->getForAllMap()->existsNameMap(one,two,matchmap);
  }
  return 0;
}

int IrNameMap::existsNameMap(std::string one, std::string two) {
  return subList->existsNameMap(one,two,NULL);
}

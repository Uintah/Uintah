// Copyright 2006  Reaction Engineering International
// by Yang

#include "KInpReader.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <set>
#include <cmath>

#ifdef _WIN32
#  include <windows.h>
#  include <Mmsystem.h> // windows time functions, link against Winmm.lib
#else
#  include <sys/time.h>
#endif
#include <errno.h>


#define BUFFER_MAX 5000 

static bool start_time_initialized = false;

#ifdef _WIN32
static DWORD start_time; // measured in milliseconds

static double
currentSeconds()
{
  if(!start_time_initialized) {
    start_time_initialized=true;
    timeBeginPeriod(1); // give the timer millisecond accuracy
    start_time = timeGetTime();
  }
  DWORD now_time = timeGetTime();

  return ((double)(now_time - start_time))*1.e-3;
}

#else
static struct timeval start_time;

static double
currentSeconds()
{
  struct timeval now_time;
  if(gettimeofday(&now_time, 0) != 0)
    throw std::string("gettimeofday failed: ")+strerror(errno);
  if (!start_time_initialized) {
    start_time_initialized = true;
    start_time = now_time;
  }

  return (now_time.tv_sec-start_time.tv_sec)+(now_time.tv_usec-start_time.tv_usec)*1.e-6;
}
#endif

Part::~Part()
{
}



k_inp_reader::k_inp_reader()
{
	errcnt = 0;
}

k_inp_reader::~k_inp_reader()
{
	size_t i;

	for (i=0; i<secBeams.size(); i++)
		if (secBeams[i]!=NULL)
			delete secBeams[i];

	for (i=0; i<secShells.size(); i++)
		if (secShells[i]!=NULL)
			delete secShells[i];

	for (i=0; i<secSolids.size(); i++)
		if (secSolids[i]!=NULL)
			delete secSolids[i];

	for (i=0; i<elemBeams.size(); i++)
		if (elemBeams[i]!=NULL)
			delete elemBeams[i];

	for (i=0; i<elemShells.size(); i++)
		if (elemShells[i]!=NULL)
			delete elemShells[i];

	for (i=0; i<elemSolids.size(); i++)
		if (elemSolids[i]!=NULL)
			delete elemSolids[i];

	for (i=0; i<nodes.size(); i++)
      if (nodes[i]!=NULL)
			delete nodes[i];

	for (i=0; i<parts.size(); i++)
		if (parts[i]!=NULL)
			delete parts[i];

   for (i=0; i<hourglass.size(); i++)
      if (hourglass[i]!=NULL)
         delete hourglass[i];
         
   for(i=0; i<nodeLists.size(); i++)
      if (nodeLists[i]!=NULL)
         delete nodeLists[i];

   for(i=0; i<loadBodys.size(); i++)
      if (loadBodys[i]!=NULL)
         delete loadBodys[i];

   for(i=0; i<curves.size(); i++)
      if (curves[i]!=NULL)
         delete curves[i];

   for(i=0; i<rigidWallPlanars.size(); i++)
      if (rigidWallPlanars[i]!=NULL)
         delete rigidWallPlanars[i];

   for(i=0; i<initailVelocityGenerations.size(); i++)
      if (initailVelocityGenerations[i]!=NULL)
         delete initailVelocityGenerations[i];

   for(i=0; i<contactOption1Titles.size(); i++)
      if (contactOption1Titles[i]!=NULL)
         delete contactOption1Titles[i];

   for(i=0; i<setSegments.size(); i++)
      if (setSegments[i]!=NULL)
         delete setSegments[i];

   for(i=0; i<sDOrientations.size(); i++)
      if (sDOrientations[i]!=NULL)
         delete sDOrientations[i];

   for(i=0; i<matPiecewiseLinearPlasticitys.size(); i++)
      if (matPiecewiseLinearPlasticitys[i]!=NULL)
         delete matPiecewiseLinearPlasticitys[i];

   for(i=0; i<matElastics.size(); i++)
      if (matElastics[i]!=NULL)
         delete matElastics[i];

   for(i=0; i<sectionDiscretes.size(); i++)
      if (sectionDiscretes[i]!=NULL)
         delete sectionDiscretes[i];

   for(i=0; i<matSpringNonlinearElastics.size(); i++)
      if (matSpringNonlinearElastics[i]!=NULL)
         delete matSpringNonlinearElastics[i];

   for(i=0; i<matRigids.size(); i++)
      if (matRigids[i]!=NULL)
         delete matRigids[i];

   for(i=0; i<matPlasticKinematics.size(); i++)
      if (matPlasticKinematics[i]!=NULL)
         delete matPlasticKinematics[i];

   for(i=0; i<airbagSimpleAirbagModels.size(); i++)
      if (airbagSimpleAirbagModels[i]!=NULL)
         delete airbagSimpleAirbagModels[i];

   for(i=0; i<constrainedExtraNodes.size(); i++)
      if (constrainedExtraNodes[i]!=NULL)
         delete constrainedExtraNodes[i];

   for(i=0; i<constrainedJoints.size(); i++)
      if (constrainedJoints[i]!=NULL)
         delete constrainedJoints[i];

   for(i=0; i<constrainedNodalRigidBodies.size(); i++)
      if (constrainedNodalRigidBodies[i]!=NULL)
         delete constrainedNodalRigidBodies[i];

   for(i=0; i<constrainedRigidBodies.size(); i++)
      if (constrainedRigidBodies[i]!=NULL)
         delete constrainedRigidBodies[i];

   for(i=0; i<constrainedSpotwelds.size(); i++)
      if (constrainedSpotwelds[i]!=NULL)
         delete constrainedSpotwelds[i];

   for(i=0; i<elementDiscretes.size(); i++)
      if (elementDiscretes[i]!=NULL)
         delete elementDiscretes[i];

   for(i=0; i<elementMass.size(); i++)
      if (elementMass[i]!=NULL)
         delete elementMass[i];

   for(i=0; i<elementSeatbeltAccelerometers.size(); i++)
      if (elementSeatbeltAccelerometers[i]!=NULL)
         delete elementSeatbeltAccelerometers[i];

   for(i=0; i<matBlatzKORubbers.size(); i++)
      if (matBlatzKORubbers[i]!=NULL)
         delete matBlatzKORubbers[i];

   for(i=0; i<matDamperViscous.size(); i++)
      if (matDamperViscous[i]!=NULL)
         delete matDamperViscous[i];

   for(i=0; i<matHoneycombs.size(); i++)
      if (matHoneycombs[i]!=NULL)
         delete matHoneycombs[i];

   for(i=0; i<matSpringElastics.size(); i++)
      if (matSpringElastics[i]!=NULL)
         delete matSpringElastics[i];


}

int k_inp_reader::is_inp_keyw(std::string oword)
{ 
  const size_t temp_size = 256;
  char temp[temp_size];
  size_t i;
  std::string word;
  
  strcpy(temp,oword.c_str());
  for (i=0; i<strlen(temp); i++)
    temp[i]=(char)toupper(temp[i]);
  temp[i]='\0';
  word = std::string(temp);
  
  if (match(word, "*KEYWORD"))
    return 1; 
  else if (match(word, "*TITLE"))
    return 2; 
  else if (match(word, "*INCLUDE"))
    return 3; 
  else if (match(word, "*END"))
    return 4; 
  else if (match(word, "*SET_NODE_LIST"))
    return 5; 
  else if (match(word, "*LOAD_BODY_Z"))
    return 6; 
  else if (match(word, "*DEFINE_CURVE"))
	return 7;
  else if (match(word, "*RIGIDWALL_PLANAR"))
	return 8;
  else if (match(word, "*INITIAL_VELOCITY_GENERATION"))
	return 9;
  else if (match(word, "*CONTACT_AUTOMATIC_SINGLE_SURFACE_TITLE"))
	return 10;
  else if (match(word, "*CONTACT_SINGLE_EDGE_TITLE"))
	return 11;
  else if (match(word, "*CONTACT_TIED_NODES_TO_SURFACE_TITLE"))
	return 12;
  else if (match(word, "*SET_SEGMENT"))
	return 13;
  //else if (match(word, "*DEFINE_CURVE"))
  //return 14;
  else if (match(word, "*DEFINE_SD_ORIENTATION"))
	return 15;
  else if (match(word, "*PART"))
	return 16;
  else if (match(word, "*SECTION_SHELL"))
	return 17;
  else if (match(word, "*MAT_PIECEWISE_LINEAR_PLASTICITY"))
	return 18;
  else if (match(word, "*SECTION_SOLID"))
	return 19;
  else if (match(word, "*MAT_ELASTIC"))
	return 20;
  else if (match(word, "*SECTION_DISCRETE"))
	return 21;
  else if (match(word, "*MAT_SPRING_NONLINEAR_ELASTIC"))
	return 22;
  else if (match(word, "*MAT_RIGID"))
	return 23;
  else if (match(word, "*SECTION_BEAM"))
	return 24;
  else if (match(word, "*ELEMENT_SOLID"))			
    return 25;
  else if (match(word, "*ELEMENT_SHELL"))			
  {
		if (match(word,"*ELEMENT_SHELL_BETA"))
			return 261;
		else if (match(word,"*ELEMENT_SHELL_THICKNESS"))
			return 262;
		else
			return 26;
  }
  else if (match(word, "*ELEMENT_BEAM"))   
    return 27;                       
  else if (match(word, "*NODE"))        
    return 28;       
  else if (match(word, "*MAT_PLASTIC_KINEMATIC"))
	  return 29;
  else if (match(word, "*HOURGLASS"))
	  return 30;
  else if (match(word, "*AIRBAG_SIMPLE_AIRBAG_MODEL"))
	  return 31;
  else if (match(word, "*CONSTRAINED_EXTRA_NODES_NODE"))
	  return 32;
  else if (match(word, "*CONSTRAINED_EXTRA_NODES_SET"))
	  return 33;
  else if (match(word, "*CONSTRAINED_JOINT_REVOLUTE"))
	  return 34;
  else if (match(word, "*CONSTRAINED_JOINT_SPHERICAL"))
	  return 35;
  else if (match(word, "*CONSTRAINED_JOINT_UNIVERSAL"))
	  return 36;
  else if (match(word, "*CONSTRAINED_NODAL_RIGID_BODY"))
	  return 37;
  else if (match(word, "*CONSTRAINED_RIGID_BODIES"))
	  return 38;
  else if (match(word, "*CONSTRAINED_SPOTWELD"))
	  return 39;
  else if (match(word, "*CONTROL_ACCURACY"))
	  return 40;
  else if (match(word, "*CONTROL_CONTACT"))
	  return 41;
  else if (match(word, "*CONTROL_CPU"))
	  return 42;
  else if (match(word, "*CONTROL_ENERGY"))
	  return 43;
  else if (match(word, "*CONTROL_OUTPUT"))
	  return 44;
  else if (match(word, "*CONTROL_SHELL"))
	  return 45;
  else if (match(word, "*CONTROL_SOLID"))
	  return 46;
  else if (match(word, "*CONTROL_TERMINATION"))
	  return 47;
  else if (match(word, "*CONTROL_TIMESTEP"))
	  return 48;
  else if (match(word, "*DATABASE_ABSTAT"))
	  return 49;
  else if (match(word, "*DATABASE_BINARY_D3PLOT"))
	  return 50;
  else if (match(word, "*DATABASE_BINARY_D3THDT"))
	  return 51;
  else if (match(word, "*DATABASE_BINARY_INTFOR"))
	  return 52;
  else if (match(word, "*DATABASE_BINARY_RUNRSF"))
	  return 53;
  else if (match(word, "*DATABASE_DEFORC"))
	  return 54;
  else if (match(word, "*DATABASE_EXTENT_BINARY"))
	  return 55;
  else if (match(word, "*DATABASE_GLSTAT"))
	  return 56;
  else if (match(word, "*DATABASE_HISTORY_NODE"))
	  return 57;
  else if (match(word, "*DATABASE_JNTFORC"))
	  return 58;
  else if (match(word, "*DATABASE_MATSUM"))
	  return 59;
  else if (match(word, "*DATABASE_NODOUT"))
	  return 60;
  else if (match(word, "*DATABASE_RCFORC"))
	  return 61;
  else if (match(word, "*DATABASE_RWFORC"))
	  return 62;
  else if (match(word, "*DATABASE_SLEOUT"))
	  return 63;
  else if (match(word, "*ELEMENT_DISCRETE"))
	  return 64;
  else if (match(word, "*ELEMENT_MASS"))
	  return 65;
  else if (match(word, "*ELEMENT_SEATBELT_ACCELEROMETER"))
	  return 66;
  else if (match(word, "*MAT_BLATZ-KO_RUBBER"))
	  return 67;
  else if (match(word, "*MAT_DAMPER_VISCOUS"))
	  return 68;
  else if (match(word, "*MAT_HONEYCOMB"))
	  return 69;
  else if (match(word, "*MAT_SPRING_ELASTIC"))
	  return 70;
  else if (match(word, "*SET_PART_LIST"))
	  return 71;
  else if (match(word,"*")) //it is still a keyword, which will mark the end of other keyword section
	  return 0; //not a key word
  else
    return -1;
}

int k_inp_reader::parse_inp(const char * filename)
{
  char buffer[BUFFER_MAX+1];
  char* token;
  std::string tok;
  int keywordSection;
  bool end_of_file;
  int cardNum;

  NodeList* cnodelist;
  SecBeam* csecbeam;
  SecShell* csecshell;
  SecSolid* csecsolid;

  ElemSolid *celemsolid;
  ElemShell *celemshell;
  ElemBeam *celembeam;
  Node *cnode;
  Part *cpart;

  LoadBody *cloadbody;
  Curve * ccurve;
  RigidWallPlanar* crigidwallplanar;
  InitailVelocityGeneration* cinitailvelocitygeneration;
  ContactOption1Title* contactoption1title;
  SetSegment* csetsegment;
  SDOrientation* csdorientation;
  SectionDiscrete* csectiondiscrete;
  AirbagSimpleAirbagModel * cairbagsimpleairbagmodel;
  ConstrainedExtraNodes *constrainedextranodes;
  ElementDiscrete *celementdiscrete;
  ElementMass *celementmass;
  ElementSeatbeltAccelerometer *celementseatbeltaccelerometer;
  ConstrainedJoint *constrainedjoint;
  ConstrainedNodalRigidBody *constrainedNodalRigidBody;
  ConstrainedRigidBody *constrainedRigidBody;
  ConstrainedSpotweld *constrainedSpotweld;

  Hourglass *chourglass;
  MatElastic *cmatelastic;
  MatPlasticKinematic *cmatplastickinematic;
  MatRigid *cmatrigid;
  MatSpringNonlinearElastic *cmatspringnonlinearelastic;
  MatPiecewiseLinearPlasticity *cmatpiecewiselinearplasticity;
  MatBlatzKORubber *cmatblatzkorubber;
  MatDamperViscous *cmatdamperviscous ;
  MatSpringElastic *cmatspringelastic;
  MatHoneycomb *cmathoneycomb;
  
  int i;
  std::string tempop;
  double tempd1, tempd2;
  SetSegmentCard2 tempSetSegmentCard2;	
  int offset;

  std::ifstream inp(filename);
  std::set<std::string> unrecognized;
  do 
	{
		end_of_file = (inp.getline(buffer, BUFFER_MAX, '\n')).eof();
		
		if (isComments(buffer)) //skip the comments line
			continue;
		
		if (buffer[0]=='*') //this is a keyword line
		{
			token = strtok(buffer," \n");
         //std::cout<<token<<std::endl;
			keywordSection = is_inp_keyw(token);

			if (keywordSection==0)
				unrecognized.insert(token);
			cardNum = 0;
			continue;
		}

		//a data line
		tempop="";
		char* found = NULL;
		char path_buffer[100] = "Dyna3DWriter\\";
		switch(keywordSection)
		{
		case 0: //not recognized
			//end of last keyword, not recognizable word
			break;
		case 1: //KEYWORD
			//"KEYWORD didn't do anything
			break;
		case 2: //TITLE
			break;
		case 3: //INCLUDE
			token = strtok(buffer, "\n");
#ifdef _WIN32
			found = strchr(token, ':');
			// if ":" not found assume path is relative
			// and prepend directory name defined above
			if (!found)
				token = strcat(path_buffer, token);
#endif
			parse_inp(token);
			break;
		case 4: //END
			inp.close(); //end of everything;deltT = atof(toks[1].c_str());
			//return 0; //normal ending
			break;
		case 5: //SET_NODE_LIST
			
			if (cardNum==0) //card 1
			{
					cnodelist = new NodeList;
					nodeLists.push_back(cnodelist);
					tok= getTok(buffer,0,10);
					cnodelist->SID=atoi(tok.c_str());
               nodeLists_m[cnodelist->SID] = cnodelist;
					tok= getTok(buffer,10,10);
					if (tok!="")
						cnodelist->DA1=atof(tok.c_str());
					else 
						cnodelist->DA1 =0;
					tok= getTok(buffer,20,10);
					if (tok!="")
						cnodelist->DA2=atof(tok.c_str());
					else 
						cnodelist->DA2 =0;
					tok= getTok(buffer,30,10);
					if (tok!="")
						cnodelist->DA3=atof(tok.c_str());
					else 
						cnodelist->DA3 =0;

			}
			else
			{
				for (i=0;i<8;i++)
				{
					tok= getTok(buffer,i*10,10);
					if (tok!="")
						(cnodelist->NID).push_back(atoi(tok.c_str()));
				}
			}
			
			break;
		case 6: //LOAD_BODY_Z
				tempop="Z";

				cloadbody = new LoadBody;
				cloadbody->op = tempop;
				loadBodys.push_back(cloadbody);
				tok= getTok(buffer,0,10);
				cloadbody->LCID = atoi(tok.c_str());
				tok= getTok(buffer,10,10);
				if (tok!="")
					cloadbody->SF = atof(tok.c_str());
				else
					cloadbody->SF = 1;
				tok= getTok(buffer,20,10);
				if (tok!="")
					cloadbody->LCIDDR = atoi(tok.c_str());
				else
					cloadbody->LCIDDR = 0;
				tok= getTok(buffer,30,10);
				if (tok!="")
					cloadbody->XC = atof(tok.c_str());
				else
					cloadbody->XC = 0;
				tok= getTok(buffer,40,10);
				if (tok!="")
					cloadbody->YC = atof(tok.c_str());
				else
					cloadbody->YC = 0;
				tok= getTok(buffer,50,10);
				if (tok!="")
					cloadbody->ZC = atof(tok.c_str());
				else
					cloadbody->ZC = 0;
			
			break;
		case 7: //DEFINE_CURVE
			
			if (cardNum==0)
			{
				ccurve = new Curve;
				curves.push_back(ccurve);
				tok= getTok(buffer,0,10);
				ccurve->LCID = atoi(tok.c_str());
				tok= getTok(buffer,10,10);
				if (tok!="")
					ccurve->SIDR = atoi(tok.c_str());
				else
					ccurve->SIDR = 0;
				tok= getTok(buffer,20,10);
				if (tok!="")
					ccurve->SFA = atof(tok.c_str());
				else
					ccurve->SFA = 1;
				tok= getTok(buffer,30,10);
				if (tok!="")
					ccurve->SFO = atof(tok.c_str());
				else
					ccurve->SFO = 1;
				tok= getTok(buffer,40,10);
				if (tok!="")
					ccurve->OFFA = atof(tok.c_str());
				else
					ccurve->OFFA = 1;
				tok= getTok(buffer,50,10);
				if (tok!="")
					ccurve->OFFO = atof(tok.c_str());
				else
					ccurve->OFFO = 1;
				tok= getTok(buffer,60,10);
				if (tok!="")
					ccurve->DATTYP = atoi(tok.c_str());
				else
					ccurve->DATTYP = 1;
			}
			else
			{
				tok= getTok(buffer,0,20);
				tempd1=atof(tok.c_str());
				tok= getTok(buffer,20,20);
				tempd2=atof(tok.c_str());
				ccurve->AO.push_back(std::pair<double, double>(tempd1, tempd2));
			}
			break;
		case 8: //RIGIDWALL_PLANAR
			
			if (cardNum==0)
			{
				crigidwallplanar = new RigidWallPlanar;
				rigidWallPlanars.push_back(crigidwallplanar);
				tok= getTok(buffer,0,10);
				crigidwallplanar->NSID = atoi(tok.c_str());
				tok= getTok(buffer,10,10);
				if (tok!="")
					crigidwallplanar->NSIDEX = atoi(tok.c_str());
				else
					crigidwallplanar->NSIDEX = 0;
				tok= getTok(buffer,20,10);
				if (tok!="")
					crigidwallplanar->BOXID = atoi(tok.c_str());
				else
					crigidwallplanar->BOXID = 0;
				tok= getTok(buffer,30,10);
				if (tok!="")
					crigidwallplanar->OFFSET = atof(tok.c_str());
				else
					crigidwallplanar->OFFSET = 0;
			}
			else //card2
			{
				tok= getTok(buffer,0,10);
				if (tok!="")
					crigidwallplanar->XT = atof(tok.c_str());
				else
					crigidwallplanar->XT = 0;
				tok= getTok(buffer,10,10);
				if (tok!="")
					crigidwallplanar->YT = atof(tok.c_str());
				else
					crigidwallplanar->YT = 0;
				tok= getTok(buffer,20,10);
				if (tok!="")
					crigidwallplanar->ZT = atof(tok.c_str());
				else
					crigidwallplanar->ZT = 0;
				tok= getTok(buffer,30,10);
				if (tok!="")
					crigidwallplanar->XH = atof(tok.c_str());
				else
					crigidwallplanar->XH = 0;
				tok= getTok(buffer,40,10);
				if (tok!="")
					crigidwallplanar->YH = atof(tok.c_str());
				else
					crigidwallplanar->YH = 0;
				tok= getTok(buffer,50,10);
				if (tok!="")
					crigidwallplanar->ZH = atof(tok.c_str());
				else
					crigidwallplanar->ZH = 0;
				tok= getTok(buffer,60,10);
				if (tok!="")
					crigidwallplanar->FRIC = atof(tok.c_str());
				else
					crigidwallplanar->FRIC = 0;
				tok= getTok(buffer,70,10);
				if (tok!="")
					crigidwallplanar->WVEL = atof(tok.c_str());
				else
					crigidwallplanar->WVEL = 0;
			}
			break;
		case 9: //INITIAL_VELOCITY_GENERATION
			if (cardNum==0)
			{
				cinitailvelocitygeneration = new InitailVelocityGeneration;
				initailVelocityGenerations.push_back(cinitailvelocitygeneration);
				tok= getTok(buffer,0,10);
				cinitailvelocitygeneration->ID = atoi(tok.c_str());
				tok= getTok(buffer,10,10);
				cinitailvelocitygeneration->STYP = atoi(tok.c_str());
				tok= getTok(buffer,20,10);
				if (tok!="")
					cinitailvelocitygeneration->OMEGA = atof(tok.c_str());
				else
					cinitailvelocitygeneration->OMEGA = 0;
				tok= getTok(buffer,30,10);
				if (tok!="")
					cinitailvelocitygeneration->VX = atof(tok.c_str());
				else
					cinitailvelocitygeneration->VX = 0;
				tok= getTok(buffer,40,10);
				if (tok!="")
					cinitailvelocitygeneration->VY = atof(tok.c_str());
				else
					cinitailvelocitygeneration->VY = 0;
				tok= getTok(buffer,50,10);
				if (tok!="")
					cinitailvelocitygeneration->VZ = atof(tok.c_str());
				else
					cinitailvelocitygeneration->VZ = 0;
			}
			else if (cardNum==1)
			{
				tok= getTok(buffer,0,10);
				if (tok!="")
					cinitailvelocitygeneration->XC = atof(tok.c_str());
				else
					cinitailvelocitygeneration->XC = 0;
				tok= getTok(buffer,10,10);
				if (tok!="")
					cinitailvelocitygeneration->YC = atof(tok.c_str());
				else
					cinitailvelocitygeneration->YC = 0;
				tok= getTok(buffer,20,10);
				if (tok!="")
					cinitailvelocitygeneration->ZC = atof(tok.c_str());
				else
					cinitailvelocitygeneration->ZC = 0;
				tok= getTok(buffer,30,10);
				if (tok!="")
					cinitailvelocitygeneration->NX = atof(tok.c_str());
				else
					cinitailvelocitygeneration->NX = 0;
				tok= getTok(buffer,40,10);
				if (tok!="")
					cinitailvelocitygeneration->NY = atof(tok.c_str());
				else
					cinitailvelocitygeneration->NY = 0;
				tok= getTok(buffer,50,10);
				if (tok!="")
					cinitailvelocitygeneration->NZ = atof(tok.c_str());
				else
					cinitailvelocitygeneration->NZ = 0;
				tok= getTok(buffer,60,10);
				if (tok!="")
					cinitailvelocitygeneration->PHASE = atoi(tok.c_str());
				else
					cinitailvelocitygeneration->PHASE = 0;
			}
			break;
	  	case 10: //CONTACT_AUTOMATIC_SINGLE_SURFACE_TITLE
			if (tempop=="")
				tempop = "AUTOMATIC_SINGLE_SURFACE";
		case 11: //CONTACT_SINGLE_EDGE_TITLE
			if (tempop=="")
				tempop = "SINGLE_EDGE";
		case 12: //CONTACT_TIED_NODES_TO_SURFACE_TITLE
			if (tempop=="")
				tempop = "TIED_NODES_TO_SURFACE";
			if (cardNum==0)
			{
				contactoption1title=new ContactOption1Title;
				contactOption1Titles.push_back(contactoption1title);
				tok = getTok(buffer, 0, 10);
				contactoption1title->CID=atoi(tok.c_str());
				contactoption1title->option1=tempop;
			}
			else if (cardNum==1)
			{
				tok = getTok(buffer, 0, 10);
				contactoption1title->SSID=atoi(tok.c_str());
				tok = getTok(buffer, 10, 10);
				contactoption1title->MSID=atoi(tok.c_str());
				tok = getTok(buffer, 20, 10);
				contactoption1title->SSTYP=atoi(tok.c_str());
				tok = getTok(buffer, 30, 10);
				contactoption1title->MSTYP=atoi(tok.c_str());
				tok= getTok(buffer,40,10);
				if (tok!="")
					contactoption1title->SBOXID = atoi(tok.c_str());
				else
					contactoption1title->SBOXID = -1;
				tok= getTok(buffer,50,10);
				if (tok!="")
					contactoption1title->MBOXID = atoi(tok.c_str());
				else
					contactoption1title->MBOXID = -1;
				tok= getTok(buffer,60,10);
				if (tok!="")
					contactoption1title->SPR = atoi(tok.c_str());
				else
					contactoption1title->SPR = 0;
				tok= getTok(buffer,70,10);
				if (tok!="")
					contactoption1title->MPR = atoi(tok.c_str());
				else
					contactoption1title->MPR = 0;
			}
			else if (cardNum==2)
			{
				tok= getTok(buffer,0,10);
				if (tok!="")
					contactoption1title->FS = atof(tok.c_str());
				else
					contactoption1title->FS = 0;
				tok= getTok(buffer,10,10);
				if (tok!="")
					contactoption1title->FD = atof(tok.c_str());
				else
					contactoption1title->FD = 0;
				tok= getTok(buffer,20,10);
				if (tok!="")
					contactoption1title->DC = atof(tok.c_str());
				else
					contactoption1title->DC = 0;
				tok= getTok(buffer,30,10);
				if (tok!="")
					contactoption1title->VC = atof(tok.c_str());
				else
					contactoption1title->VC = 0;
				tok= getTok(buffer,40,10);
				if (tok!="")
					contactoption1title->VDC = atof(tok.c_str());
				else
					contactoption1title->VDC = 0;
				tok= getTok(buffer,50,10);
				if (tok!="")
					contactoption1title->PENCHK = atoi(tok.c_str());
				else
					contactoption1title->PENCHK = 0;
				tok= getTok(buffer,60,10);
				if (tok!="")
					contactoption1title->BT = atof(tok.c_str());
				else
					contactoption1title->BT = 0;
				tok= getTok(buffer,70,10);
				if (tok!="")
					contactoption1title->DT = atof(tok.c_str());
				else
					contactoption1title->DT = 1.0E20;
				
			}
			else if (cardNum==3)
			{
				tok= getTok(buffer,0,10);
				if (tok!="")
					contactoption1title->SFS = atof(tok.c_str());
				else
					contactoption1title->SFS = 1;
				tok= getTok(buffer,10,10);
				if (tok!="")
					contactoption1title->SFM = atof(tok.c_str());
				else
					contactoption1title->SFM = 1;
				tok= getTok(buffer,20,10);
				if (tok!="")
					contactoption1title->SST = atof(tok.c_str());
				else
					contactoption1title->SST = -1000; //bogus value, 
				tok= getTok(buffer,30,10);
				if (tok!="")
					contactoption1title->MST = atof(tok.c_str());
				else
					contactoption1title->MST = -1000; //bogus value
				tok= getTok(buffer,40,10);
				if (tok!="")
					contactoption1title->SFST = atof(tok.c_str());
				else
					contactoption1title->SFST = 1;
				tok= getTok(buffer,50,10);
				if (tok!="")
					contactoption1title->SFMT = atof(tok.c_str());
				else
					contactoption1title->SFMT = 1;
				tok= getTok(buffer,60,10);
				if (tok!="")
					contactoption1title->FSF = atof(tok.c_str());
				else
					contactoption1title->FSF = 1;
				tok= getTok(buffer,70,10);
				if (tok!="")
					contactoption1title->VSF = atof(tok.c_str());
				else
					contactoption1title->VSF = 1;
			}
			break;
		case 13: //SET_SEGMENT
			if (cardNum==0)
			{
				csetsegment = new SetSegment;
				setSegments.push_back(csetsegment);
				tok = getTok(buffer,0, 10);
				csetsegment->SID = atoi(tok.c_str());
				tok = getTok(buffer,10, 10);
				if (tok!="")
					csetsegment->DA[0] = atof(tok.c_str());
				else
					csetsegment->DA[0] = 0; //bogus value
				tok = getTok(buffer,20, 10);
				if (tok!="")
					csetsegment->DA[1] = atof(tok.c_str());
				else
					csetsegment->DA[1] = 0; //bogus value
				tok = getTok(buffer,30, 10);
				if (tok!="")
					csetsegment->DA[2] = atof(tok.c_str());
				else
					csetsegment->DA[2] = 0; //bogus value
				tok = getTok(buffer,40, 10);
				if (tok!="")
					csetsegment->DA[3] = atof(tok.c_str());
				else
					csetsegment->DA[3] = 0; //bogus value
			}
			else
			{
				tok = getTok(buffer,0, 10);
				tempSetSegmentCard2.N[0]=atoi(tok.c_str());
				tok = getTok(buffer,10, 10);
				tempSetSegmentCard2.N[1]=atoi(tok.c_str());
				tok = getTok(buffer,20, 10);
				tempSetSegmentCard2.N[2]=atoi(tok.c_str());
				tok = getTok(buffer,30, 10);
				tempSetSegmentCard2.N[3]=atoi(tok.c_str());
				tok = getTok(buffer,40, 10);
				tempSetSegmentCard2.A[0]=atoi(tok.c_str());
				tok = getTok(buffer,50, 10);
				tempSetSegmentCard2.A[1]=atoi(tok.c_str());
				tok = getTok(buffer,60, 10);
				tempSetSegmentCard2.A[2]=atoi(tok.c_str());
				tok = getTok(buffer,70, 10);
				tempSetSegmentCard2.A[3]=atoi(tok.c_str());
				(csetsegment->card2).push_back(tempSetSegmentCard2);
			}
			break;
		case 14: //nothing to do here, an duplicate
			break;
		case 15: //DEFINE_SD_ORIENTATION
			csdorientation = new SDOrientation;
         sDOrientations.push_back(csdorientation);
			tok = getTok(buffer, 0, 10);
			csdorientation->VID = atoi(tok.c_str());
			tok = getTok(buffer, 10, 10);
			if (tok!="")
				csdorientation->IOP = atoi(tok.c_str());
			else
				csdorientation->IOP = 0;
			tok = getTok(buffer, 20, 10);
			if (tok!="")
				csdorientation->XT = atof(tok.c_str());
			else
				csdorientation->XT = 0;
			tok = getTok(buffer, 30, 10);
			if (tok!="")
				csdorientation->YT = atof(tok.c_str());
			else
				csdorientation->YT = 0;
			tok = getTok(buffer, 40, 10);
			if (tok!="")
				csdorientation->ZT = atof(tok.c_str());
			else
				csdorientation->ZT = 0;
			tok = getTok(buffer, 50, 10);
			if (tok!="")
				csdorientation->NID1 = atoi(tok.c_str());
			else
				csdorientation->NID1 = 0;
			tok = getTok(buffer, 60, 10);
			if (tok!="")
				csdorientation->NID2 = atoi(tok.c_str());
			else
				csdorientation->NID2 = 0;
			break;
		case 16: //PART
			if (cardNum==0)
			{
				cpart = new Part;
				parts.push_back(cpart);
				cpart->title = std::string(buffer);
			}
			else if (cardNum==1)
			{
				tok= getTok(buffer,0,10);
				cpart->pid = atoi(tok.c_str());
				tok= getTok(buffer,10,10);
				cpart->secid = atoi(tok.c_str());
				tok= getTok(buffer,20,10);
				cpart->mid = atoi(tok.c_str());
				tok= getTok(buffer,30,10);
				if (tok!="")
					cpart->eosid = atoi(tok.c_str());
				else
					cpart->eosid = 0;
				tok= getTok(buffer,40,10);
				if (tok!="")
					cpart->hgid = atoi(tok.c_str());
				else
					cpart->hgid = 0;
				tok= getTok(buffer,50,10);
				if (tok!="")
					cpart->grav = atoi(tok.c_str());
				else
					cpart->grav = 0;
				tok= getTok(buffer,60,10);
				if (tok!="")
					cpart->adpopt = atoi(tok.c_str());
				else
					cpart->adpopt = 0;
				tok= getTok(buffer,70,10);
				if (tok!="")
					cpart->tmid = atoi(tok.c_str());
				else
					cpart->tmid = 0;
			}
			break;
		case 17: //section shell SECTION_SHELL
			if (cardNum==0)
			{
				csecshell = new SecShell;
				secShells.push_back(csecshell);

				tok= getTok(buffer,0,10);
				csecshell->secid = atoi(tok.c_str());
				tok= getTok(buffer,10,10);
				csecshell->elform = atoi(tok.c_str());
				tok= getTok(buffer,20,10);
				csecshell->shrf = atof(tok.c_str());
				tok= getTok(buffer,30,10);
				csecshell->nip = atof(tok.c_str());
				tok= getTok(buffer,40,10);
				csecshell->propt = atof(tok.c_str());
				tok= getTok(buffer,50,10);
				csecshell->or_irid = atof(tok.c_str());
				tok= getTok(buffer,60,10);
				csecshell->icomp = atoi(tok.c_str());
				tok= getTok(buffer,70,10);
				csecshell->setyp = atoi(tok.c_str());
			}
			else if (cardNum==1) //card 2
			{
				tok= getTok(buffer,0,10);
				csecshell->t[0] = atof(tok.c_str());
				tok= getTok(buffer,10,10);
				csecshell->t[1] = atof(tok.c_str());
				tok= getTok(buffer,20,10);
				csecshell->t[2] = atof(tok.c_str());
				tok= getTok(buffer,30,10);
				csecshell->t[3] = atof(tok.c_str());
				tok= getTok(buffer,40,10);
				csecshell->nloc = atof(tok.c_str());
				tok= getTok(buffer,50,10);
				csecshell->marea = atof(tok.c_str());
			}
			break;
		case 18: //MAT_PIECEWISE_LINEAR_PLASTICITY
			//card 2
			if (cardNum==0)
			{
				cmatpiecewiselinearplasticity = new MatPiecewiseLinearPlasticity;
				matPiecewiseLinearPlasticitys.push_back(cmatpiecewiselinearplasticity);
				tok= getTok(buffer,0,10);
				cmatpiecewiselinearplasticity->MID = atoi(tok.c_str());
				tok= getTok(buffer,10,10);
				cmatpiecewiselinearplasticity->RO = atof(tok.c_str());
				tok= getTok(buffer,20,10);
				cmatpiecewiselinearplasticity->E = atof(tok.c_str());
				tok= getTok(buffer,30,10);
				cmatpiecewiselinearplasticity->PR = atof(tok.c_str());
				tok= getTok(buffer,40,10);
				cmatpiecewiselinearplasticity->SIGY = atof(tok.c_str());
				tok= getTok(buffer,50,10);
				if (tok!="")
					cmatpiecewiselinearplasticity->ETAN = atof(tok.c_str());
				else
					cmatpiecewiselinearplasticity->ETAN = 0.0;
				tok= getTok(buffer,60,10);
				if (tok!="")
					cmatpiecewiselinearplasticity->FAIL = atof(tok.c_str());
				else
					cmatpiecewiselinearplasticity->FAIL = 10.0E+20;
				tok= getTok(buffer,70,10);
				if (tok!="")
					cmatpiecewiselinearplasticity->TDEL = atof(tok.c_str());
				else
					cmatpiecewiselinearplasticity->TDEL = 0.0;
			}
			else if (cardNum==1)
			{
				tok= getTok(buffer,0,10);
				if (tok!="")
					cmatpiecewiselinearplasticity->C = atof(tok.c_str());
				else
					cmatpiecewiselinearplasticity->C = 0.0;
				tok= getTok(buffer,10,10);
				if (tok!="")
					cmatpiecewiselinearplasticity->P = atof(tok.c_str());
				else
					cmatpiecewiselinearplasticity->P = 0.0;
				tok= getTok(buffer,20,10);
				if (tok!="")
					cmatpiecewiselinearplasticity->LCSS = atoi(tok.c_str());
				else
					cmatpiecewiselinearplasticity->LCSS = 0;
				tok= getTok(buffer,30,10);
				if (tok!="")
					cmatpiecewiselinearplasticity->LCSR = atoi(tok.c_str());
				else
					cmatpiecewiselinearplasticity->LCSR = 0;
				tok= getTok(buffer,40,10);
				if (tok!="")
					cmatpiecewiselinearplasticity->VP = atof(tok.c_str());
				else
					cmatpiecewiselinearplasticity->VP = 0.0;
			}
			else if (cardNum==2)
			{
				for (i=0; i<8; i++)
				{
					tok= getTok(buffer,i*10,10);
					if (tok!="")
						cmatpiecewiselinearplasticity->EPS[i] = atof(tok.c_str());
					else
						cmatpiecewiselinearplasticity->EPS[i] = 0.0;
				}
			}
			else if (cardNum==3)
			{
				for (i=0; i<8; i++)
				{
					tok= getTok(buffer,i*10,10);
					if (tok!="")
						cmatpiecewiselinearplasticity->ES[i] = atof(tok.c_str());
					else
						cmatpiecewiselinearplasticity->ES[i] = 0.0;
				}
			}
			break;
		case 19: //section solid SECTION_SOLID
			csecsolid = new SecSolid;
			secSolids.push_back(csecsolid);

			tok= getTok(buffer,0,10);
			csecsolid->secid = atoi(tok.c_str());
			tok= getTok(buffer,10,10);
			csecsolid->elform = atoi(tok.c_str());
			tok= getTok(buffer,20,10);
			csecsolid->aet = atoi(tok.c_str());
			break;
		case 20: // MAT_ELASTIC
			cmatelastic = new MatElastic;
			matElastics.push_back(cmatelastic);
			tok= getTok(buffer,0,10);
			cmatelastic->MID = atoi(tok.c_str());
			tok= getTok(buffer,10,10);
			cmatelastic->RO = atof(tok.c_str());
			tok= getTok(buffer,20,10);
			cmatelastic->E = atof(tok.c_str());
			tok= getTok(buffer,30,10);
			cmatelastic->PR = atof(tok.c_str());
			tok= getTok(buffer,40,10);
			if (tok!="")
				cmatelastic->DA = atof(tok.c_str());
			else
				cmatelastic->DA = 0.0;
			tok= getTok(buffer,50,10);
			if (tok!="")
				cmatelastic->DB = atof(tok.c_str());
			else
				cmatelastic->DB = 0.0;
			tok= getTok(buffer,60,10);
			if (tok!="")
				cmatelastic->K = atof(tok.c_str());
			else
				cmatelastic->K = 0.0;
			break;
		case 21: //SECTION_DISCRETE
			if (cardNum==0)
			{
				csectiondiscrete = new SectionDiscrete;
				sectionDiscretes.push_back(csectiondiscrete);
				tok = getTok(buffer, 0, 10);
				csectiondiscrete->SECID = atoi(tok.c_str());
				tok = getTok(buffer, 10, 10);
				csectiondiscrete->DRO = atoi(tok.c_str());
				tok = getTok(buffer, 20, 10);
				csectiondiscrete->KD = atof(tok.c_str());
				tok = getTok(buffer, 30, 10);
				csectiondiscrete->V0 = atof(tok.c_str());
				tok = getTok(buffer, 40, 10);
				csectiondiscrete->CL = atof(tok.c_str());
				tok = getTok(buffer, 50, 10);
				csectiondiscrete->FD = atof(tok.c_str());
			}	
			else 
			{
				tok = getTok(buffer, 0, 10);
				csectiondiscrete->CDL = atof(tok.c_str());
				tok = getTok(buffer, 10, 10);
				csectiondiscrete->TDL = atof(tok.c_str());
			}
			break;
		case 22://MAT_SPRING_NONLINEAR_ELASTIC
			cmatspringnonlinearelastic = new MatSpringNonlinearElastic;
			matSpringNonlinearElastics.push_back(cmatspringnonlinearelastic);
			tok= getTok(buffer,0,10);
			cmatspringnonlinearelastic->MID = atoi(tok.c_str());
			tok= getTok(buffer,10,10);
			cmatspringnonlinearelastic->LCD = atoi(tok.c_str());
			tok= getTok(buffer,20,10);
			cmatspringnonlinearelastic->LCR = atoi(tok.c_str());
			break;
		case 23://MAT_RIGID
			if (cardNum==0)
			{
				cmatrigid = new MatRigid;
				matRigids.push_back(cmatrigid);

				tok= getTok(buffer,0,10);
				cmatrigid->MID = atoi(tok.c_str());
				midLeaveOut.insert(cmatrigid->MID);
				tok= getTok(buffer,10,10);
				cmatrigid->RO = atof(tok.c_str());
				tok= getTok(buffer,20,10);
				cmatrigid->E = atof(tok.c_str());
				tok= getTok(buffer,30,10);
				cmatrigid->PR = atof(tok.c_str());
				tok= getTok(buffer,40,10);
				cmatrigid->N = atof(tok.c_str());
				tok= getTok(buffer,50,10);
				cmatrigid->COUPLE = atof(tok.c_str());
				tok= getTok(buffer,60,10);
				if (tok!="")
					cmatrigid->M = atof(tok.c_str());
				else
					cmatrigid->M = 0.0;
				tok= getTok(buffer,70,10);
				if (tok!="")
					cmatrigid->ALIASRE = atof(tok.c_str());
				else
					cmatrigid->ALIASRE = 0.0;
				
			}
			else if (cardNum==1) //card 2
			{
				tok= getTok(buffer,0,10);
				cmatrigid->CMO = atof(tok.c_str());
				tok= getTok(buffer,10,10);
				cmatrigid->CON1 = atof(tok.c_str());
				tok= getTok(buffer,20,10);
				cmatrigid->CON2 = atof(tok.c_str());
			}
			else if (cardNum==2) //card 3
			{
				tok= getTok(buffer,0,10);
				cmatrigid->LCOA1 = atof(tok.c_str());
				tok= getTok(buffer,10,10);
				cmatrigid->A2 = atof(tok.c_str());
				tok= getTok(buffer,20,10);
				cmatrigid->A3 = atof(tok.c_str());
				tok= getTok(buffer,30,10);
				cmatrigid->V1 = atof(tok.c_str());
				tok= getTok(buffer,40,10);
				cmatrigid->V2 = atof(tok.c_str());
				tok= getTok(buffer,50,10);
				cmatrigid->V3 = atof(tok.c_str());
			}
			break;
		case 24: //section_beam SECTION_BEAM
			if (cardNum==0)
			{
				csecbeam = new SecBeam;
				secBeams.push_back(csecbeam);

				tok= getTok(buffer,0,10);
				csecbeam->secid = atoi(tok.c_str());
				tok= getTok(buffer,10,10);
				csecbeam->elform = atoi(tok.c_str());
				tok= getTok(buffer,20,10);
				csecbeam->shrf = atof(tok.c_str());
				tok= getTok(buffer,30,10);
				csecbeam->qr_irid = atof(tok.c_str());
				tok= getTok(buffer,40,10);
				csecbeam->cst = atof(tok.c_str());
				tok= getTok(buffer,50,10);
				csecbeam->scoor = atof(tok.c_str());
				tok= getTok(buffer,60,10);
				csecbeam->nsm = atof(tok.c_str());
			}
			else if (cardNum==1) //card 2
			{
				for (i=0; i<8; i++)
				{
					tok= getTok(buffer,i*10, 10);
					csecbeam->f[i] = atoi(tok.c_str());
				}
			}
			break;
		case 25: //element_solid ELEMENT_SOLID
				
			celemsolid = new ElemSolid;

			tok= getTok(buffer,0,8);
			celemsolid->eid = atoi(tok.c_str());
			tok= getTok(buffer,8,8);
			celemsolid->pid = atoi(tok.c_str());
			for (i=0;i<8;i++)
			{
				tok= getTok(buffer,i*8+16,8);
				celemsolid->n[i]=atoi(tok.c_str());
			}

			elemSolids.push_back(celemsolid);
			break;
		case 26: //element_shell ELEMENT_SHELL
		case 261: //ELEMENT_SHELL_BETA
		case 262: //ELEMENT_SHELL_THICKNESS
			if (cardNum==0)
			{
				celemshell = new ElemShell;
				celemshell->t[0]=celemshell->t[1]=celemshell->t[2]=celemshell->t[3]=0.0; //initialize them to be 0

				tok= getTok(buffer,0,8);
				celemshell->eid = atoi(tok.c_str());
				tok= getTok(buffer,8,8);
				celemshell->pid = atoi(tok.c_str());
				for (i=0;i<4;i++)
				{
					tok= getTok(buffer,i*8+16,8);
					celemshell->n[i]=atoi(tok.c_str());
				}

				elemShells.push_back(celemshell);
				if (keywordSection==26)
					cardNum=-1; //so the carNum++ at the end will reset it to be 0
			}
			else if ((keywordSection==261||keywordSection==262) && cardNum==1)
			{
				for (i=0;i<4;i++)
				{
					tok= getTok(buffer,i*16,16);
					celemshell->t[i]=atof(tok.c_str());
				}

                                // following is not used
				// tok= getTok(buffer,64,16);
				// celemshell->psi = atof(tok.c_str());

				cardNum=-1; //so the carNum++ at the end will reset it to be 0
				
			}
			break;
		case 27: //element_beam ELEMENT_BEAM
			celembeam = new ElemBeam;
			tok= getTok(buffer,0,8);
			celembeam->eid = atoi(tok.c_str());
			tok= getTok(buffer,8,8);
			celembeam->pid = atoi(tok.c_str());
			tok= getTok(buffer,16,8);
			celembeam->n1 = atoi(tok.c_str());
			tok= getTok(buffer,24,8);
			celembeam->n2 = atoi(tok.c_str());
			tok= getTok(buffer,32,8);
			celembeam->n3 = atoi(tok.c_str());
			tok= getTok(buffer,40,8);
			celembeam->rt1 = atoi(tok.c_str());
			tok= getTok(buffer,48,8);
			celembeam->rr1 = atoi(tok.c_str());
			tok= getTok(buffer,56,8);
			celembeam->rt2 = atoi(tok.c_str());
			tok= getTok(buffer,64,8);
			celembeam->rr2 = atoi(tok.c_str());
			tok= getTok(buffer,72,8);
			celembeam->local = atoi(tok.c_str());
				
			elemBeams.push_back(celembeam);
			break;
		case 28: // node NODE
			cnode = new Node;
			tok= getTok(buffer,0,8);
			cnode->nid = atoi(tok.c_str());
			tok= getTok(buffer,8,16);
			cnode->x[0] = atof(tok.c_str())*convert_length + translate[0];
			tok= getTok(buffer,24,16);
			cnode->x[1] = atof(tok.c_str())*convert_length + translate[1];
			tok= getTok(buffer,40,16);
			cnode->x[2] = atof(tok.c_str())*convert_length + translate[2];
			tok= getTok(buffer,56,8);
			cnode->tc = atof(tok.c_str());
			tok= getTok(buffer,72,8);
			cnode->rc = atof(tok.c_str());
			nodes.push_back(cnode);

			break;
		case 29: //MAT_PLASTIC_KINEMATIC
			if (cardNum==0)
			{
				cmatplastickinematic = new MatPlasticKinematic;
				matPlasticKinematics.push_back(cmatplastickinematic);

				tok= getTok(buffer,0,10);
				cmatplastickinematic->MID = atoi(tok.c_str());
				tok= getTok(buffer,10,10);
				cmatplastickinematic->RO = atof(tok.c_str());
				tok= getTok(buffer,20,10);
				cmatplastickinematic->E = atof(tok.c_str());
				tok= getTok(buffer,30,10);
				cmatplastickinematic->PR = atof(tok.c_str());
				tok= getTok(buffer,40,10);
				cmatplastickinematic->SIGY = atof(tok.c_str());
				tok= getTok(buffer,50,10);
				if (tok!="")
					cmatplastickinematic->ETAN = atof(tok.c_str());
				else
					cmatplastickinematic->ETAN = 0.0;
				tok= getTok(buffer,60,10);
				if (tok!="")
					cmatplastickinematic->BETA = atof(tok.c_str());
				else
					cmatplastickinematic->BETA = 0.0;
				
			}
			else if (cardNum==1) //card 2
			{
				tok= getTok(buffer,0,10);
				cmatplastickinematic->SRC = atof(tok.c_str());
				tok= getTok(buffer,10,10);
				cmatplastickinematic->SRP = atof(tok.c_str());
				tok= getTok(buffer,20,10);
				cmatplastickinematic->FS = atof(tok.c_str());
				tok= getTok(buffer,30,10);
				if (tok!="")
					cmatplastickinematic->VP = atof(tok.c_str());
				else
					cmatplastickinematic->VP = 0.0;
			}
			break;
		case 30: //HOURGLASS
			chourglass = new Hourglass;
			hourglass.push_back(chourglass);
			tok= getTok(buffer,0,10);
			chourglass->HGID = atoi(tok.c_str());
			tok= getTok(buffer,10,10);
			chourglass->IHQ = atoi(tok.c_str());
			tok= getTok(buffer,20,10);
			chourglass->QM = atof(tok.c_str());
			tok= getTok(buffer,30,10);
			chourglass->IBQ = atoi(tok.c_str());
			tok= getTok(buffer,40,10);
			if (tok!="")
				chourglass->Q[0] = atof(tok.c_str());
			else
				chourglass->Q[0] = 1.5;
			tok= getTok(buffer,50,10);
			if (tok!="")
				chourglass->Q[1] = atof(tok.c_str());
			else
				chourglass->Q[1] = 0.06;
			tok= getTok(buffer,60,10);
			if (tok!="")
				chourglass->QB = atof(tok.c_str());
			else
				chourglass->QB = chourglass->QM;
			tok= getTok(buffer,70,10);
			if (tok!="")
				chourglass->QW = atof(tok.c_str());
			else
				chourglass->QW = chourglass->QM;
			break;
		case 31: //AIRBAG_SIMPLE_AIRBAG_MODEL
			if (cardNum ==0)
			{
				cairbagsimpleairbagmodel=new AirbagSimpleAirbagModel;
				airbagSimpleAirbagModels.push_back(cairbagsimpleairbagmodel);
				tok= getTok(buffer,0,10);
				cairbagsimpleairbagmodel->SID = atoi(tok.c_str());
				tok= getTok(buffer,10,10);
				if (tok!="")
					cairbagsimpleairbagmodel->SIDTYP = atoi(tok.c_str());
				else
					cairbagsimpleairbagmodel->SIDTYP = 0;
				tok= getTok(buffer,20,10);
				if (tok!="")
					cairbagsimpleairbagmodel->RBID = atoi(tok.c_str());
				else
					cairbagsimpleairbagmodel->RBID = 0;
				tok= getTok(buffer,30,10);
				if (tok!="")
					cairbagsimpleairbagmodel->VSCA = atof(tok.c_str());
				else
					cairbagsimpleairbagmodel->VSCA = 1;
				tok= getTok(buffer,40,10);
				if (tok!="")
					cairbagsimpleairbagmodel->PSCA = atof(tok.c_str());
				else
					cairbagsimpleairbagmodel->PSCA = 1;
				tok= getTok(buffer,50,10);
				if (tok!="")
					cairbagsimpleairbagmodel->VINI = atof(tok.c_str());
				else
					cairbagsimpleairbagmodel->VINI = 0;
				tok= getTok(buffer,60,10);
				if (tok!="")
					cairbagsimpleairbagmodel->MWD = atof(tok.c_str());
				else
					cairbagsimpleairbagmodel->MWD = 0;
				tok= getTok(buffer,70,10);
				if (tok!="")
					cairbagsimpleairbagmodel->SPSF = atof(tok.c_str());
				else
					cairbagsimpleairbagmodel->SPSF = 0;
			}
			else if (cardNum ==1 )
			{
				tok= getTok(buffer,0,10);
				cairbagsimpleairbagmodel->CV = atof(tok.c_str());
				tok= getTok(buffer,10,10);
				cairbagsimpleairbagmodel->CP = atof(tok.c_str());
				tok= getTok(buffer,20,10);
				cairbagsimpleairbagmodel->T = atof(tok.c_str());
				tok= getTok(buffer,30,10);
				cairbagsimpleairbagmodel->LCID = atoi(tok.c_str());
				tok= getTok(buffer,40,10);
				cairbagsimpleairbagmodel->MU = atof(tok.c_str());
				tok= getTok(buffer,50,10);
				cairbagsimpleairbagmodel->A = atof(tok.c_str());
				tok= getTok(buffer,60,10);
				cairbagsimpleairbagmodel->PE = atof(tok.c_str());
				tok= getTok(buffer,70,10);
				cairbagsimpleairbagmodel->RO = atof(tok.c_str());
			}
			else if (cardNum ==2)
			{
				tok= getTok(buffer,0,10);
				if (tok!="")
					cairbagsimpleairbagmodel->LOU = atoi(tok.c_str());
				else
					cairbagsimpleairbagmodel->LOU = 0;
				tok= getTok(buffer,10,10);
				if (tok!="")
					cairbagsimpleairbagmodel->TEXT = atof(tok.c_str());
				else
					cairbagsimpleairbagmodel->TEXT = 0;
				tok= getTok(buffer,20,10);
				if (tok!="")
					cairbagsimpleairbagmodel->Acard2 = atof(tok.c_str());
				else
					cairbagsimpleairbagmodel->Acard2 = 0;
				tok= getTok(buffer,30,10);
				if (tok!="")
					cairbagsimpleairbagmodel->B = atof(tok.c_str());
				else
					cairbagsimpleairbagmodel->B = 1;
				tok= getTok(buffer,40,10);
				if (tok!="")
					cairbagsimpleairbagmodel->MW = atof(tok.c_str());
				else
					cairbagsimpleairbagmodel->MW = 0;
				tok= getTok(buffer,50,10);
				if (tok!="")
					cairbagsimpleairbagmodel->GASC = atof(tok.c_str());
				else
					cairbagsimpleairbagmodel->GASC = 0;
			}
			break;
		case 32: //CONSTRAINED_EXTRA_NODES_NODE
			tempop="NODE";
		case 33: //CONSTRAINED_EXTRA_NODES_SET
			if (tempop=="")
				tempop="SET";
			constrainedextranodes = new ConstrainedExtraNodes;
			constrainedExtraNodes.push_back(constrainedextranodes);
			tok = getTok(buffer, 0, 10);
			constrainedextranodes->PID = atoi(tok.c_str());
			tok = getTok(buffer, 10, 10);
			constrainedextranodes->NIDNSID = atoi(tok.c_str());
         constrainedextranodes->option = tempop;
			break;
		case 34: //CONSTRAINED_JOINT_REVOLUTE
			tempop = "REVOLUTE";
		case 35: //CONSTRAINED_JOINT_SPHERICAL
			if(tempop=="")
				tempop = "SPHERICAL";
		case 36: //CONSTRAINED_JOINT_UNIVERSAL
			if(tempop=="")
				tempop = "UNIVERSAL";
			constrainedjoint = new ConstrainedJoint;
			constrainedJoints.push_back(constrainedjoint);

			offset = 0;
			for( i=0; i < JOINT_SIZE; i++ ) // iterate through N1 to N6
			{
				tok = getTok(buffer, offset, 10);
				constrainedjoint->N[i] = atoi(tok.c_str());
				offset += 10;
			}

			tok = getTok(buffer, offset + 10, 10);
			constrainedjoint->RPS = atof(tok.c_str());
			tok = getTok(buffer, offset + 20, 10);
			constrainedjoint->DAMP = atof(tok.c_str());
         constrainedjoint->option = tempop;
			break;
		case 37: //CONSTRAINED_NODAL_RIGID_BODY
			constrainedNodalRigidBody = new ConstrainedNodalRigidBody;
			constrainedNodalRigidBodies.push_back(constrainedNodalRigidBody);
			tok = getTok(buffer, 0, 10);
			constrainedNodalRigidBody->PID = atoi(tok.c_str());
			tok = getTok(buffer, 10, 10);
			constrainedNodalRigidBody->CID = atoi(tok.c_str());
			tok = getTok(buffer, 20, 10);
			constrainedNodalRigidBody->NSID = atoi(tok.c_str());
			break;
		case 38: //CONSTRAINED_RIGID_BODIES
			constrainedRigidBody = new ConstrainedRigidBody;
			constrainedRigidBodies.push_back(constrainedRigidBody);
			tok = getTok(buffer, 0, 10);
			constrainedRigidBody->PIDM = atoi(tok.c_str());
			tok = getTok(buffer, 10, 10);
			constrainedRigidBody->PIDS = atoi(tok.c_str());
			break;
		case 39: //CONSTRAINED_SPOTWELD
			constrainedSpotweld = new ConstrainedSpotweld;
			constrainedSpotwelds.push_back(constrainedSpotweld);
			tok = getTok(buffer, 0, 10);
			constrainedSpotweld->N1 = atoi(tok.c_str());
			tok = getTok(buffer, 10, 10);
			constrainedSpotweld->N2 = atoi(tok.c_str());
			tok = getTok(buffer, 20, 10);
			constrainedSpotweld->SN = atof(tok.c_str());
			tok = getTok(buffer, 30, 10);
			constrainedSpotweld->SS = atof(tok.c_str());
			tok = getTok(buffer, 40, 10);
			constrainedSpotweld->N = atof(tok.c_str());
			tok = getTok(buffer, 50, 10);
			constrainedSpotweld->M = atof(tok.c_str());
			tok = getTok(buffer, 60, 10);
			constrainedSpotweld->TF = atof(tok.c_str());
			tok = getTok(buffer, 70, 10);
			constrainedSpotweld->EP = atof(tok.c_str());
			break;
		case 40: //CONTROL_ACCURACY
			break;
		case 41: //CONTROL_CONTACT
			break;
		case 42: //CONTROL_CPU
			break;
		case 43: //CONTROL_ENERGY
			break;
		case 44: //CONTROL_OUTPUT
			break;
		case 45: //CONTROL_SHELL
			break;
		case 46: //CONTROL_SOLID
			break;
		case 47: //CONTROL_TERMINATION
			tok = getTok(buffer, 0, 10);
			endTime = atof(tok.c_str());
			break;
		case 48: //CONTROL_TIMESTEP
			break;
		case 49: //DATABASE_ABSTAT  //We don't need to deal with any Database option for now since they are output options
			break;
		case 50: //DATABASE_BINARY_D3PLOT  //output
			break;
		case 51: //DATABASE_BINARY_D3THDT  //output
			break;
		case 52: //DATABASE_BINARY_INTFOR  //output
			break;
		case 53: //DATABASE_BINARY_RUNRSF  //outpu
			break; 
		case 54: //DATABASE_DEFORC  //output
			break;
		case 55: //DATABASE_EXTENT_BINARY  //output
			break;
		case 56: //DATABASE_GLSTAT  //output
			break;
		case 57: //DATABASE_HISTORY_NODE  //output
			break;
		case 58: //DATABASE_JNTFORC  //output
			break;
		case 59: //DATABASE_MATSUM  //output
			break;
		case 60: //DATABASE_NODOUT  //output
			break;
		case 61: //DATABASE_RCFORC  //output
			break;
		case 62: //DATABASE_RWFORC  //output
			break;
		case 63: //DATABASE_SLEOUT  //output
			break;
		case 64: //ELEMENT_DISCRETE  //output
			celementdiscrete= new ElementDiscrete;
			elementDiscretes.push_back(celementdiscrete);
			tok= getTok(buffer,0,8);
			celementdiscrete->EID=atoi(tok.c_str());
			tok= getTok(buffer,8,8);
			celementdiscrete->PID=atoi(tok.c_str());
			tok= getTok(buffer,16,8);
			celementdiscrete->N1=atoi(tok.c_str());
			tok= getTok(buffer,24,8);
			celementdiscrete->N2=atoi(tok.c_str());
			tok= getTok(buffer,32,8);
			if (tok!="")
				celementdiscrete->VID=atoi(tok.c_str());
			else
				celementdiscrete->VID=0;
			tok= getTok(buffer,40,16);
			if (tok!="")
				celementdiscrete->S=atof(tok.c_str());
			else
				celementdiscrete->S=0;
			tok= getTok(buffer,56,8);
			if (tok!="")
				celementdiscrete->PF=atoi(tok.c_str());
			else
				celementdiscrete->PF=0;
			tok= getTok(buffer,64,16);
			if (tok!="")
				celementdiscrete->OFFSET=atof(tok.c_str());
			else
				celementdiscrete->OFFSET=0;
			break;
		case 65: //ELEMENT_MASS
			celementmass = new ElementMass;
			elementMass.push_back(celementmass);
			tok= getTok(buffer,0,8);
			celementmass->EID = atoi(tok.c_str());
			tok= getTok(buffer,8,8);
			celementmass->NID = atoi(tok.c_str());
			tok= getTok(buffer,16,16);
			if (tok!="")
				celementmass->MASS = atof(tok.c_str());
			else
				celementmass->MASS=0;
			tok= getTok(buffer,32,10);
			celementmass->PID = atoi(tok.c_str());
			break;
		case 66: //ELEMENT_SEATBELT_ACCELEROMETER
			celementseatbeltaccelerometer = new ElementSeatbeltAccelerometer;
			elementSeatbeltAccelerometers.push_back(celementseatbeltaccelerometer);
			tok = getTok(buffer, 0, 10);
			if (tok!="")
				celementseatbeltaccelerometer->SBACID=atoi(tok.c_str());
			else
				celementseatbeltaccelerometer->SBACID=0;
			tok = getTok(buffer, 10, 10);
			if (tok!="")
				celementseatbeltaccelerometer->NID1=atoi(tok.c_str());
			else
				celementseatbeltaccelerometer->NID1=0;
			tok = getTok(buffer, 20, 10);
			if (tok!="")
				celementseatbeltaccelerometer->NID2=atoi(tok.c_str());
			else
				celementseatbeltaccelerometer->NID2=0;
			tok = getTok(buffer, 30, 10);
			if (tok!="")
				celementseatbeltaccelerometer->NID3=atoi(tok.c_str());
			else
				celementseatbeltaccelerometer->NID3=0;
			tok = getTok(buffer, 40, 10);
			if (tok!="")
				celementseatbeltaccelerometer->IGRAV=atoi(tok.c_str());
			else
				celementseatbeltaccelerometer->IGRAV=0;
			tok = getTok(buffer, 50, 10);
			if (tok!="")
				celementseatbeltaccelerometer->INTOPT=atoi(tok.c_str());
			else
				celementseatbeltaccelerometer->INTOPT=0;
			
			break;
		case 67: //MAT_BLATZ-KO_RUBBER
			cmatblatzkorubber = new MatBlatzKORubber;
			matBlatzKORubbers.push_back(cmatblatzkorubber);
			tok= getTok(buffer,0,10);
			cmatblatzkorubber->MID = atoi(tok.c_str());
			tok= getTok(buffer,10,10);
			cmatblatzkorubber->RO = atof(tok.c_str());
			tok= getTok(buffer,20,10);
			cmatblatzkorubber->G = atof(tok.c_str());
			tok= getTok(buffer,30,10);
			cmatblatzkorubber->REF = atof(tok.c_str());
			midLeaveOut.insert(cmatblatzkorubber->MID);
			break;
		case 68: //MAT_DAMPER_VISCOUS
			cmatdamperviscous= new MatDamperViscous;
			matDamperViscous.push_back(cmatdamperviscous);
			tok= getTok(buffer,0,10);
			cmatdamperviscous->MID = atoi(tok.c_str());
			tok= getTok(buffer,10,10);
			cmatdamperviscous->DC = atof(tok.c_str());
			break;
		case 69: //MAT_HONEYCOMB
			if (cardNum==0)
			{
				cmathoneycomb=new MatHoneycomb;
				matHoneycombs.push_back(cmathoneycomb);
				tok= getTok(buffer,0,10);
				cmathoneycomb->MID = atoi(tok.c_str());
				tok= getTok(buffer,10,10);
				cmathoneycomb->RO = atof(tok.c_str());
				tok= getTok(buffer,20,10);
				cmathoneycomb->E = atof(tok.c_str());
				tok= getTok(buffer,30,10);
				cmathoneycomb->PR = atof(tok.c_str());
				tok= getTok(buffer,40,10);
				cmathoneycomb->SIGY = atof(tok.c_str());
				tok= getTok(buffer,50,10);
				cmathoneycomb->VF = atof(tok.c_str());
				tok= getTok(buffer,60,10);
				if (tok!="")
					cmathoneycomb->MU = atof(tok.c_str());
				else
					cmathoneycomb->MU = 0.05;
				tok= getTok(buffer,70,10);
				if (tok!="")
					cmathoneycomb->BULK = atof(tok.c_str());
				else
					cmathoneycomb->BULK = 0;
			}
			else if (cardNum==1)
			{
				tok= getTok(buffer,0,10);
				cmathoneycomb->LCA = atoi(tok.c_str());
				tok= getTok(buffer,10,10);
				if (tok!="")
					cmathoneycomb->LCB = atoi(tok.c_str());
				else
					cmathoneycomb->LCB = cmathoneycomb->LCA;
				tok= getTok(buffer,20,10);
				if (tok!="")
					cmathoneycomb->LCC = atoi(tok.c_str());
				else
					cmathoneycomb->LCC = cmathoneycomb->LCA;
				tok= getTok(buffer,30,10);
				if (tok!="")
					cmathoneycomb->LCS = atoi(tok.c_str());
				else
					cmathoneycomb->LCS = cmathoneycomb->LCA;
				tok= getTok(buffer,40,10);
				if (tok!="")
					cmathoneycomb->LCAB = atoi(tok.c_str());
				else
					cmathoneycomb->LCAB = cmathoneycomb->LCS;
				tok= getTok(buffer,50,10);
				if (tok!="")
					cmathoneycomb->LCBC = atoi(tok.c_str());
				else
					cmathoneycomb->LCBC = cmathoneycomb->LCS;
				tok= getTok(buffer,60,10);
				if (tok!="")
					cmathoneycomb->LCCA = atoi(tok.c_str());
				else
					cmathoneycomb->LCCA = cmathoneycomb->LCS;
				tok= getTok(buffer,70,10);
				if (tok!="")
					cmathoneycomb->LCSR = atoi(tok.c_str());
				else
					cmathoneycomb->LCSR = 0;
			}
			else if (cardNum==2)
			{
				tok= getTok(buffer,0,10);
				cmathoneycomb->EAAU = atof(tok.c_str());
				tok= getTok(buffer,10,10);
				cmathoneycomb->EBBU = atof(tok.c_str());
				tok= getTok(buffer,20,10);
				cmathoneycomb->ECCU = atof(tok.c_str());
				tok= getTok(buffer,30,10);
				cmathoneycomb->GABU = atof(tok.c_str());
				tok= getTok(buffer,40,10);
				cmathoneycomb->GBCU = atof(tok.c_str());
				tok= getTok(buffer,50,10);
				cmathoneycomb->GCAU = atof(tok.c_str());
				tok= getTok(buffer,60,10);
				if (tok!="")
					cmathoneycomb->AOPT = atof(tok.c_str());
				else
					cmathoneycomb->AOPT = 0;
			}
			else if (cardNum==3)
			{
				tok= getTok(buffer,0,10);
				cmathoneycomb->XP = atof(tok.c_str());
				tok= getTok(buffer,10,10);
				cmathoneycomb->YP = atof(tok.c_str());
				tok= getTok(buffer,20,10);
				cmathoneycomb->ZP = atof(tok.c_str());
				tok= getTok(buffer,30,10);
				cmathoneycomb->A1 = atof(tok.c_str());
				tok= getTok(buffer,40,10);
				cmathoneycomb->A2 = atof(tok.c_str());
				tok= getTok(buffer,50,10);
				cmathoneycomb->A3 = atof(tok.c_str());
			
			}
			break;
		case 70: //MAT_SPRING_ELASTIC
			cmatspringelastic= new MatSpringElastic;
			matSpringElastics.push_back(cmatspringelastic);
			tok= getTok(buffer,0,10);
			cmatspringelastic->MID = atoi(tok.c_str());
			tok= getTok(buffer,10,10);
			cmatspringelastic->K = atof(tok.c_str());
			break;
		case 71: //SET_PART_LIST
			
			break;
		default:
			break;
		}
		cardNum++;
		      
    } while(!end_of_file);

	for (std::set<std::string>::iterator uiter=unrecognized.begin(); uiter!=unrecognized.end(); uiter++)
		printf("%s\n", uiter->c_str());
	//exit(0);
	return 0;
}

void k_inp_reader::dump()
{

}

//Ignore Comments get delete anything started with !
bool k_inp_reader::isComments(const char* current_line)
{
	if (current_line[0]=='$')
		return true;
	else
		return false;
}

std::string k_inp_reader::getTok(const char* buffer, int start_pos, int len)
{
	static char temp[BUFFER_MAX];
	char temp2[BUFFER_MAX];
	std::string result;
	int slen = static_cast<int>(strlen(buffer));
	int i;
	const char* tok;
	//Try to be able to deal with free format input file as well
	//First, check if there is comma inside the buffer, which means it is free formatted line
	static bool isFreeFormat;
	
	if (start_pos ==0) //check it in the start of the line;
	{
		isFreeFormat = false;

		for (i=0; i<slen; i++)
			if (buffer[i]==',')
			{
				isFreeFormat = true;
				strcpy(temp, buffer);
				result=strtok(temp,",");
				goto trim;
			}
	}
	
	if (isFreeFormat)
	{
		tok = strtok(NULL,",");
		if (tok!=NULL)
			result = tok;
		else
			result = "";
	}
	else
	{
		if (start_pos<slen-1)
		{
			strncpy(temp, &buffer[start_pos], len);
			temp[len]='\0';
			result = temp;
		}
	}
	

	//now trim all the white space
trim:	
	int j=0;
	int rlen = static_cast<int>(result.length());
	for (i=0; i<rlen; i++)
		if (result.at(i)!=' '&&result.at(i)!='\t'&&result.at(i)!='\n') //not white space
			temp2[j++]=result.at(i);
	temp2[j]='\0';
	
	result =temp2;

    return result;
}

bool k_inp_reader::match(const std::string& s1, const std::string& s2)
{
    size_t n = s2.size();
    if (s1.size() < n) return false;
    for (size_t i = 0; i < n; i++) 
	if (s2[i] != '*' && (toupper(s1[i]) != toupper(s2[i]))) return false;
    return true;
}

void k_inp_reader::create_tri_initialization()
{
  for(size_t i=0; i<parts.size(); i++){
    parts_m[parts[i]->pid] = parts[i];
  }

  for(unsigned int i=0; i<3; i++){
	  xmin[i] = 1.0e10;
	  xmax[i] = -1.0e10;
  }
  for(size_t i=0; i<nodes.size(); i++){
    for(unsigned int j=0; j<3; j++){
      if(nodes[i]->x[j]>xmax[j]) xmax[j] = nodes[i]->x[j];
      if(nodes[i]->x[j]<xmin[j]) xmin[j] = nodes[i]->x[j];
    }
    nodes_m[nodes[i]->nid] = nodes[i];
  }
  std::cout<<" min "<< xmin[0]<<" "<<xmin[1]<<" "<<xmin[2]<<std::endl;
  std::cout<<" max "<< xmax[0]<<" "<<xmax[1]<<" "<<xmax[2]<<std::endl;

  for(size_t i=0; i<secShells.size(); i++){
    //for(unsigned int j=0; j<4; j++) secShells[i]->t[j]*= TCORRECTION; //10.0/1000.0;
    secShells_m[secShells[i]->secid] = secShells[i];
  }

  for(std::map<unsigned int, Part*>::iterator itp=parts_m.begin(); itp!=parts_m.end(); itp++){
    itp->second->numsolid = 0;
    itp->second->numshell = 0;
  }

}

void k_inp_reader::create_tri_shell_startup()
{
  for(size_t i=0; i<elemShells.size(); i++){
    elemShells[i]->sign = 1.0;
    for(unsigned j=0; j<4; j++) {
      //elemShells[i]->t[j] *= TCORRECTION; //10.0/1000.0;
      //The check here to either put section thickness into elemShells' thickness

      if (fabs(elemShells[i]->t[j])<1E-12) {
        std::map<unsigned int, Part*>::iterator itp = parts_m.find(elemShells[i]->pid);
		if (itp==parts_m.end()) {
			errcnt++;
			printf("elem shell with pid %d not found in parts_m\n!", elemShells[i]->pid);
		}else{
			std::map<unsigned int,SecShell*>::iterator itssh = secShells_m.find(itp->second->secid);
			if(itssh==secShells_m.end()){
				std::cout<<"Section shell "<<itp->second->secid<<" not found"<<std::endl;
				errcnt++;
			}else
				elemShells[i]->t[j] = itssh->second->t[j];
		}
      }
    }
    std::map<unsigned int, Part*>::iterator itp = parts_m.find(elemShells[i]->pid);
    if(itp!=parts_m.end()){
      for(unsigned int j1=0; j1<4; j1++){
        std::map<unsigned int,Node*>::iterator itng = nodes_m.find(elemShells[i]->n[j1]);
        if(itng!=nodes_m.end()){
          itp->second->nodes_m[elemShells[i]->n[j1]] = *itng->second;
		}else{
			errcnt++;
			std::cout<<"node "<<elemShells[i]->n[j1]<<" not found"<<std::endl;
		}
	  }
	}else{
		std::cout<<"Part "<<elemShells[i]->pid<<" not found"<<std::endl;
		errcnt++;
	}
  }
}

void k_inp_reader::create_tri_shells()
{
	Node* n0[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
  int nodeloc[8];
  int numtri = 0;
  for(size_t i=0; i<elemShells.size(); i++){
    ElemShell* es = elemShells[i];
    std::map<unsigned int, Part*>::iterator itp = parts_m.find(elemShells[i]->pid);
	if(itp==parts_m.end()){
		std::cout<<"Part "<<elemShells[i]->pid<<" not found from shell element"<<std::endl;
		errcnt++;
	}else{
		itp->second->numshell++;
		double xx[4][3], v1[3], v2[3], v3[3], xavg[3];
		for(unsigned int j1=0; j1<4; j1++){
			std::map<unsigned int,Node>::iterator itn = itp->second->nodes_m.find(elemShells[i]->n[j1]);
			if(itn==itp->second->nodes_m.end()){
				std::cout<<"Node "<<elemShells[i]->n[j1]<<" not found"<<std::endl;
				errcnt++;
			}else{
				n0[j1] = &itn->second;
				for(unsigned int k=0; k<3; k++) xx[j1][k] = n0[j1]->x[k];
			}
		} // j1
		if(n0[2]&&n0[3]){
			if(n0[2]->nid==n0[3]->nid){
				numtri++;
				for(unsigned int k=0; k<3; k++){
					v1[k] = xx[1][k] - xx[0][k];
					v2[k] = xx[2][k] - xx[0][k];
					xavg[k] = (xx[0][k] + xx[1][k] + xx[2][k])/3.0;
				}
			}else{
				for(unsigned int k=0; k<3; k++){
					v1[k] = xx[2][k] - xx[0][k];
					v2[k] = xx[3][k] - xx[1][k];
					xavg[k] = 0.25*(xx[0][k] + xx[1][k] + xx[2][k] + xx[3][k]);
				}
			}
		}
		cross_prod(v1, v2, v3);
		double denom = 0.0;
		for(unsigned int k=0; k<3; k++) denom += v3[k]*v3[k];
		denom = sqrt(denom);
		if(denom) for(unsigned int k=0; k<3; k++) v3[k] /= denom;
		// add faces (top and bottom)
		// top face
		face fct;
		for(unsigned int j1=0; j1<4; j1++){
			if(n0[j1]){
				Node node = *n0[j1];
				for(unsigned int k=0; k<3; k++){
					node.x[k] += 0.5*elemShells[i]->t[j1]*TCORRECTION*convert_length*v3[k];
					node.x[k] = 0.001*xavg[k] + 0.999*node.x[k];
				}
				if(n0[2]&&n0[3])
					if(j1<3||n0[2]->nid!=n0[3]->nid) itp->second->nodes_v.push_back(node);
				unsigned int nloc = static_cast<unsigned int>(itp->second->nodes_v.size()-1);
				itp->second->nodes_v[nloc].nidloc = nloc;
				fct.n[j1] = nloc;
				nodeloc[j1] = nloc;
			}
		}
		fct.en.push_back(elemShells[i]->eid);
		itp->second->faces_v.push_back(fct);

		// bottom face
		face fcb;
		for(unsigned int j1=0; j1<4; j1++){
			if(n0[j1]){
				Node node = *n0[j1];
				for(unsigned int k=0; k<3; k++){
					node.x[k] -= 0.5*elemShells[i]->t[j1]*TCORRECTION*convert_length*v3[k];
					node.x[k] = 0.001*xavg[k] + 0.999*node.x[k];
				}
				if(n0[2]&&n0[3])
					if(j1<3||n0[2]->nid!=n0[3]->nid) itp->second->nodes_v.push_back(node);
				unsigned int nloc = static_cast<unsigned int>(itp->second->nodes_v.size()-1);
				itp->second->nodes_v[nloc].nidloc = nloc;
				fcb.n[j1] = nloc;
				nodeloc[4+j1] = nloc;
			}
		}
		fcb.en.push_back(elemShells[i]->eid);
		itp->second->faces_v.push_back(fcb);
		// construct edge faces
		for(unsigned int j1=0; j1<4; j1++){
			unsigned int j2 = j1 + 1;
			if(j2==4) j2 = 0;
			if(j1<3||n0[2]->nid!=n0[3]->nid){
				face fc;
				fc.n[0] = nodeloc[j1];
				fc.n[1] = nodeloc[4+j1];
				fc.n[2] = nodeloc[4+j2];
				fc.n[3] = nodeloc[j2];
				fc.en.push_back(elemShells[i]->eid);
				itp->second->faces_v.push_back(fc);
			}
		}
	}
  } // for(i
  std::cout<<"numtri "<<numtri<<std::endl;
}

void k_inp_reader::create_tri_solids()
{
  // now do solid elements
   /*
       7/-------/6
      /       / |
    /       /   |
  4---------5   |
   |       |    |
   |    3  |    / 2
   |       |  /         ^3
   |       |/           |/ 2
   ---------             ->1
   0       1
   */
  int fcver[6][4] = {1, 2, 6, 5,
                     0, 3, 7, 4,
                     3, 2, 6, 7,
                     0, 1, 5, 4,
                     4, 5, 6, 7,
                     0, 1, 2, 3};

  ;
  for(size_t i=0; i<elemSolids.size(); i++){
	  std::map<unsigned int, Part*>::iterator itp = parts_m.find(elemSolids[i]->pid);
	  if(itp==parts_m.end()){
		  errcnt++;
		  printf("elem solid with pid %d not found in parts_m\n!", elemSolids[i]->pid);
	  }else{
		  std::set<unsigned int>::iterator itlo;
		  unsigned int mid = (unsigned int) itp->second->mid;
		  itlo = midLeaveOut.find(mid);
		  if(itlo!=midLeaveOut.end()) continue;
		  itp->second->numsolid++;
		  Node* n0[8];
		  // add edges, accumulate average thickness and normal direction for each node
		  for(unsigned int j1=0; j1<8; j1++){
			  std::map<unsigned int,Node>::iterator itn = itp->second->nodes_m.find(elemSolids[i]->n[j1]);
			  if(itn==itp->second->nodes_m.end()){
				  std::map<unsigned int,Node*>::iterator itng = nodes_m.find(elemSolids[i]->n[j1]);
				  itp->second->nodes_v.push_back(*itng->second);
				  itp->second->nodes_v[itp->second->nodes_v.size()-1].nidloc = static_cast<unsigned long>(itp->second->nodes_v.size()-1);
				  itp->second->nodes_m[elemSolids[i]->n[j1]] = itp->second->nodes_v[itp->second->nodes_v.size()-1];
				  itn = itp->second->nodes_m.find(elemSolids[i]->n[j1]);
			  }
			  n0[j1] = &itn->second;
		  }
		  for(unsigned int j1=0; j1<8; j1++){
			  unsigned int j2 = j1 + 1;
			  if(j2==8) j2 = 0;
			  if(elemSolids[i]->n[j1]!=elemSolids[i]->n[j2]){
				  if(j1==3) std::cout<<"tetrahedron"<<std::endl;
				  if(j1==4) std::cout<<"wedge"<<std::endl;
				  break;
			  }
		  }
		  for(unsigned int j1=0; j1<6; j1++){
			  std::set<unsigned long> order;
			  for(unsigned int j2=0; j2<4; j2++) order.insert(elemSolids[i]->n[fcver[j1][j2]]);
			  std::ostringstream str;
			  for(std::set<unsigned long>::iterator ito=order.begin();
				  ito!=order.end(); ito++) str <<"a"<<*ito;
				  std::string check = str.str();
			  std::map<std::string, face>::iterator itf = itp->second->faces.find(str.str());
			  if(itf!=itp->second->faces.end()){
				  itf->second.en.push_back(elemSolids[i]->eid);
				  if(itf->second.en.size()>2) std::cout<<" more than two elements per face "<<itf->second.en.size()<<std::endl;
			  }else{
				  face fce;
				  for(unsigned int j2=0; j2<4; j2++) fce.n[j2] = n0[fcver[j1][j2]]->nidloc;
				  fce.en.push_back(elemSolids[i]->eid);
				  itp->second->faces[str.str()] = fce;
			  }
		  }
	  }
  }
  for(std::map<unsigned int, Part*>::iterator itp = parts_m.begin(); itp != parts_m.end(); itp++){
    for(std::map<std::string, face>::iterator itf=itp->second->faces.begin(); itf!=itp->second->faces.end(); itf++){
      itp->second->faces_v.push_back(itf->second);
    }
  }
}

void k_inp_reader::create_tri_write_tris(TriFileContext& context) const
{
  // write out pts and tri files
  for(std::map<unsigned int, Part*>::const_iterator itp = parts_m.begin(); itp != parts_m.end(); itp++){
    size_t num = itp->second->nodes_v.size();
    if(num){
      FILE* pts, *tri;
      std::ostringstream ststm;
      ststm<<outputDir<<"part"<<itp->second->pid;
      std::string st = ststm.str();
      context.names.push_back(st);
      ststm<<".pts";
      st = ststm.str();
      context.ptsfiles.push_back(st);
      pts = fopen(st.c_str(),"wt");
      ststm.str("");
      ststm<<outputDir<<"part"<<itp->second->pid<<".tri";
      st = ststm.str();
      context.trifiles.push_back(st);
      tri = fopen(st.c_str(),"wt");
         
      for(size_t i=0; i<num; i++){
        float rl;
        for(unsigned int k=0; k<3; k++){
          rl = (float)itp->second->nodes_v[i].x[k];
          fprintf(pts,"%f",rl);
          if(k<2) fprintf(pts," ");
        }
        fprintf(pts,"\n");
      }
      for(size_t i=0; i<itp->second->faces_v.size(); i++){
        if(itp->second->faces_v[i].en.size()==1){
          unsigned int n1 = itp->second->faces_v[i].n[0], n2 = itp->second->faces_v[i].n[2], n3;
          n3 = itp->second->faces_v[i].n[1];
          fprintf(tri,"%d %d %d\n",n1, n2, n3);
          n3 = itp->second->faces_v[i].n[3];
          if(n2!=n3) fprintf(tri,"%d %d %d\n",n1, n2, n3);
        }
            
      }
      fclose(pts);
      fclose(tri);
    }
  }
}

std::string stripPath (const char* filename)
{
	std::string result=filename;
	std::string realresult;
	size_t a=result.find_last_of('/');

	if (a==std::string::npos)
		realresult=result;
	else
		realresult=result.substr(a+1, result.length()-a-1);

	return realresult;
}

void k_inp_reader::create_tri_mpmice(const TriFileContext& context)
{
  FILE* mpmice_file = fopen((outputDir+"mpmice_geom").c_str(),"wt");
  if (mpmice_file == NULL) {
    std::string error("Cannot create mpmice_geom file");
    throw error;
  }
  fprintf(mpmice_file,"<union>\n");
  int cnt = 0;
  std::string partname;
  for(size_t i=0; i<context.ptsfiles.size(); i++){
    read_elems(context.trifiles[i],cnt);
    read_vtcs(context.ptsfiles[i],cnt);
	partname=stripPath(context.names[i].c_str());
    fprintf(mpmice_file,"<tri><name>%s</name></tri>\n",partname.c_str());
  }
  fprintf(mpmice_file,"</union>\n");
  fclose(mpmice_file);
}


void k_inp_reader::create_tri_epilogue()
{
  // convert node location length units length units back
  for(size_t i=0; i<nodes.size(); i++){
    for(unsigned int j=0; j<3; j++){
      nodes[i]->x[j] /= convert_length;
    }
  }

  //   for(itp=parts_m.begin(); itp!=parts_m.end(); itp++) 
  //    std::cout<<itp->first<<" "<<itp->second->numsolid<<" "<<itp->second->numshell<<std::endl;
}

void k_inp_reader::create_tri()
{
  double start = currentSeconds();
  create_tri_initialization();
  beneathTires();
  double initialization_end = currentSeconds();
  create_tri_shell_startup();
  double elemShell_end = currentSeconds();
  std::cout<<"start shells"<<std::endl;
  create_tri_shells();
  double node_end = currentSeconds();
  std::cout<<"start solids "<<std::endl;
  create_tri_solids();
  double solids_end = currentSeconds();
  std::cout<<"start tri files"<<std::endl;
  TriFileContext context;
  create_tri_write_tris(context);
  double tri_writes_end = currentSeconds();
  std::cout<<"start mpmice_geom"<<std::endl;
  create_tri_mpmice(context);
  double mpmice_end = currentSeconds();
  std::cout<<"start uns"<<std::endl;
  write_uns();
  create_tri_epilogue();
  double end = currentSeconds();

  std::cout << "Total time = "<<end-start<<"\n";
  std::cout << "initialization = "<<initialization_end-start<<"\n";
  std::cout << "elemShell      = "<<elemShell_end-initialization_end<<"\n";
  std::cout << "node           = "<<node_end-elemShell_end<<"\n";
  std::cout << "solids         = "<<solids_end-node_end<<"\n";
  std::cout << "tri writes     = "<<tri_writes_end-solids_end<<"\n";
  std::cout << "mpmice         = "<<mpmice_end-tri_writes_end<<"\n";
  std::cout << "write_uns      = "<<end-mpmice_end<<"\n";

  if(errcnt){
	  std::cout<<"error processing k file. See above"<<std::endl;
	  exit(1);
  }
}

////////////////
void k_inp_reader::cross_prod(const double* v1, const double* v2, double* v3)
{
   v3[0] = v1[1]*v2[2] - v1[2]*v2[1];
   v3[1] = v1[2]*v2[0] - v1[0]*v2[2];
   v3[2] = v1[0]*v2[1] - v1[1]*v2[0];
}
/////////////////
void k_inp_reader::read_vtcs(std::string ptsfile, int& cnt)
{
   FILE *pts;
   pts = fopen(ptsfile.c_str(),"rt");
   std::vector<float> x(3);
   int i = 1, k=0;
   for(;;){
      i = fscanf(pts,"%f %f %f",&x[0],&x[1],&x[2]);
      if(i<3) break;
      cnt++;
      vertices.push_back(x);
   }
   fclose(pts);
}
/////////////////
void k_inp_reader::read_elems(std::string trifile, int cnt)
{
   FILE *tri;
   tri = fopen(trifile.c_str(),"rt");
   std::vector<int> nodes(3);
   int i = 1, j;
   for(;;){
      i = fscanf(tri,"%d %d %d",&nodes[0],&nodes[1],&nodes[2]);
      if(i<3) break;
      for(j=0; j<3; j++) nodes[j] += cnt;
      elements.push_back(nodes);
   }
   fclose(tri);
}
///////////////////
void k_inp_reader::write_uns()
{
	FILE *df1;
   // write grid in uns format
   df1 = fopen((outputDir+"uns.vtk").c_str(),"wb");
   fprintf(df1,"# vtk DataFile Version 4.0\n");
   fprintf(df1,"Tri Surface\n");
   fprintf(df1,"BINARY\n");
   fprintf(df1,"DATASET UNSTRUCTURED_GRID\n");
//////////   geom.write_uns_vtk(df1,df2);
   size_t nv = vertices.size(), iv;
   fprintf(df1,"POINTS %d float\n",nv);
   int dir;
   float rl;
   for(iv=0; iv<nv; iv++){
      for(dir=0; dir<3; dir++){
         rl = (float)vertices[iv][dir];
         swap_4((char*)&rl);
         fwrite(&rl,sizeof(float),1,df1);
      }
   }
   int nc = static_cast<int>(elements.size()), ic;
   int nc0 = 3;
   int num = nc*(nc0+1);
   int order[8] = {0,1,3,2,4,5,7,6};
   fprintf(df1,"\nCELLS %d %d\n",nc,num);
   swap_4((char*)&nc0);
//   float v1[3], v2[3], v3[3];
//   int nodes0[3];
   for(ic=0; ic<nc; ic++){
      fwrite(&nc0,sizeof(int),1,df1);
      int iv0;
      for(iv0=0; iv0<3; iv0++){
         iv = elements[ic][iv0];
//         nodes0[iv0] = iv;
         swap_4((char*)&iv);
         fwrite(&iv,sizeof(int),1,df1);
      }
      //for(dir=0; dir<3; dir++) v1[dir] = vertices[nodes0[1]][dir] - vertices[nodes0[0]][dir];
      //for(dir=0; dir<3; dir++) v2[dir] = vertices[nodes0[2]][dir] - vertices[nodes0[0]][dir];
      //cross_prod(v1, v2, v3);
      //std::cout<<ic<<" "<<vertices[nodes0[0]][2]<<" "<<v3[0]<<" "<<v3[1]<<" "<<v3[2]<<std::endl;
   }
   fprintf(df1,"\nCELL_TYPES %d\n",nc);
   int type = 5;
   swap_4((char*)&type);
   for(ic=0; ic<nc; ic++) fwrite(&type,sizeof(int),1,df1);
   fclose(df1);
}

///////////////////////
void k_inp_reader::swap_4(char* data)
{
   char b;
   b = data[0]; data[0] = data[3]; data[3] = b;
   b = data[1]; data[1] = data[2]; data[2] = b;
}
///////////////////////
void k_inp_reader::beneathTires()
{
  // determine the extent of boxes for SFEOS beneath tires

	for(unsigned int i=0; i<4; i++)
		for(unsigned int j=0; j<2; j++){
			xymin[i][j] = 1.0e20;
			xymax[i][j] = -1.0e20;
		}

  double x0[2] = {0.5*(xmin[0] + xmax[0]), 0.5*(xmin[1] + xmax[1])};
  beneathTireDepth = 0.05*(xmax[2] - xmin[2]);
  for(size_t i=0; i<nodes.size(); i++){
	  if(nodes[i]->x[2]<beneathTireDepth){
		  int iq;
		  if(nodes[i]->x[0]<x0[0]&&nodes[i]->x[1]<x0[1]) iq = 0;
		  else if(nodes[i]->x[0]>=x0[0]&&nodes[i]->x[1]<x0[1]) iq = 1;
		  else if(nodes[i]->x[0]<x0[0]&&nodes[i]->x[1]>=x0[1]) iq = 2;
		  else /*if(nodes[i]->x[0]>=x0[0]&&nodes[i]->x[1]>=x0[1])*/ iq = 3;
		  for(unsigned int j=0; j<2; j++){
			  if(nodes[i]->x[j]<xymin[iq][j]) xymin[iq][j] = nodes[i]->x[j];
			  if(nodes[i]->x[j]>xymax[iq][j]) xymax[iq][j] = nodes[i]->x[j];
		  }
	  }
  }

  beneathTireDepth *= 2.0;

  for(unsigned int iq=0; iq<4; iq++){
	  double dx = 0.1*(xmax[0] - xmin[1]);
	  xymin[iq][0] -= dx;
	  xymax[iq][0] += dx;
	  double dy = dx;
	  xymin[iq][1] -= dy;
	  xymax[iq][1] += dy;
  }
  if(xymax[0][0]>xymin[1][0]) xymax[0][0] = xymin[1][0];
  if(xymax[2][0]>xymin[3][0]) xymax[2][0] = xymin[3][0];
  if(xymax[0][1]>xymin[2][1]) xymax[0][1] = xymin[2][1];
  if(xymax[1][1]>xymin[3][1]) xymax[1][1] = xymin[3][1];
  /*for(unsigned int iq=0; iq<4; iq++){
	  std::cout<<"SFEOS beneath tire "<<iq<<" xminmax "<<xymin[iq][0]<<" "<<xymax[iq][0]<<std::endl;
	  std::cout<<" yminmax "<<xymin[iq][1]<<" "<<xymax[iq][1]<<std::endl;
  }
  std::cout<<"Depth "<<beneathTireDepth<<std::endl;*/
  mat5min[0] = xymax[0][0];
  if(xymax[2][0]>mat5min[0]) mat5min[0] = xymax[2][0];
  mat5max[0] = xymin[1][0];
  if(xymin[3][0]<mat5max[0]) mat5max[0] = xymin[3][0];

  mat5min[1] = xymax[0][1];
  if(xymax[1][1]>mat5min[1]) mat5min[1] = xymax[1][1];
  mat5max[1] = xymin[2][1];
  if(xymin[3][1]<mat5max[1]) mat5max[1] = xymin[3][1];

  if(mat5max[0]-mat5min[0]<mat5max[1]-mat5min[1]){
	  mat5min[0] = xmin[0] + 0.25*(xmax[0] - xmin[0]);
	  mat5max[0] = xmax[0] - 0.25*(xmax[0] - xmin[0]);
  }

  std::cout<<"mat5 min";
  for(unsigned i=0; i<2; i++) std::cout<<" "<<mat5min[i];
  std::cout<<std::endl;
  std::cout<<"mat5 max";
  for(unsigned i=0; i<2; i++) std::cout<<" "<<mat5max[i];
  std::cout<<std::endl;
}

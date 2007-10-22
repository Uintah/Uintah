// Copyright 2006  Reaction Engineering International
// by Yang

#include "KInpReader.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <set>
#include <cmath>

#define BUFFER_MAX 5000 

Part::~Part()
{
}



k_inp_reader::k_inp_reader()
{
	
}

k_inp_reader::~k_inp_reader()
{
	int i;

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
		
}

int k_inp_reader::is_inp_keyw(std::string oword)
{ 
  char temp[256];
  int i;
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
  else if (match(word, "*DEFINE_CURVE"))
	return 14;
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
  SecBeam* csecbeam;
  SecShell* csecshell;
  SecSolid* csecsolid;

  ElemSolid *celemsolid;
  ElemShell *celemshell;
  ElemBeam *celembeam;
  Node *cnode;
  Part *cpart;

  int i;


  std::ifstream inp(filename);

  do 
	{
		end_of_file = (inp.getline(buffer, BUFFER_MAX, '\n')).eof();
		
		if (isComments(buffer)) //skip the comments line
			continue;
		
		if (buffer[0]=='*') //this is a keyword line
		{
			token = strtok(buffer, " \n");
         //std::cout<<token<<std::endl;
			keywordSection = is_inp_keyw(token);
			cardNum = 0;
			continue;
		}

		//a data line

		switch(keywordSection)
		{
		case 0:
			//end of last keyword, not recognizable word
			break;
		case 1:
			//"KEYWORD didn't do anything
			break;
		case 2:
			break;
		case 3:
			token = strtok(buffer, "\n");
			parse_inp(token);
			break;
		case 4:
			inp.close(); //end of everything;deltT = atof(toks[1].c_str());
			return 0; //normal ending
			break;
		case 5:
			/*
			if (cardNum==0) //card 1
			{
					cnodelist = new NodeList();
					nodeList.push_back(cnodelist)
					cnodelist->sid=atoi(toks[0].c_str());
					cnodelist->da1=atof(toks[1].c_str());
					cnodelist->da2=atof(toks[2].c_str());
					cnodelist->da3=atof(toks[3].c_str());

				}
				else
				{
					for (i=0;i<num_toks;i++)
						(cnodelist->nodes).push_back(atoi(toks[i].c_str()));
				}
			*/
			break;
		case 6:
			/*
				loadbody_z.lcid = atoi(toks[0].c_str());
				loadbody_z.sf = atof(toks[1].c_str());
				loadbody_z.lciddr = atoi(toks[2].c_str());
				loadbody_z.xc = atof(toks[3].c_str());
				loadbody_z.yc = atof(toks[4].c_str());
				loadbody_z.zc = atof(toks[5].c_str());
				*/
			break;
		case 7:
			/*
			if (cardNum==0)
			{
					cloadcurve = new LoadCurve();
					loadcurveList.push_back(cloadcurve);
					cloadcurve->lcid = atoi(toks[0].c_str());
					cloadcurve->sidr = atoi(toks[1].c_str());
					cloadcurve->sfa = atof(toks[2].c_str());
					cloadcurve->sfo = atof(toks[3].c_str());
					cloadcurve->offa = atof(toks[4].c_str());
					cloadcurve->offo = atof(toks[5].c_str());
					cloadcurve->dattyp = atoi(toks[6].c_str());

				}
				else
				{
					cloadcurve->points.push_back(std::pair<double, double>(atof(toks[0].c_str()), atof(toks[1].c_str())));
				}
				*/
			break;
		case 8:
			/*
			if (cardNum==0)
				{
					cinitvel = new InitVel();
					initvelList.push_back(cinitvel);
					cinitvel->id = atoi(toks[0].c_str());
					cinitvel->styp = atoi(toks[1].c_str());
					cinitvel->omega = atof(toks[2].c_str());
					cinitvel->vx = atof(toks[3].c_str());
					cinitvel->vy = atof(toks[4].c_str());
					cinitvel->vz = atof(toks[5].c_str());

				}
				else //card2
				{
					cinitvel->xc = atof(toks[0].c_str());
					cinitvel->yc = atof(toks[1].c_str());
					cinitvel->zc = atof(toks[2].c_str());
					cinitvel->nx = atof(toks[3].c_str());
					cinitvel->ny = atof(toks[4].c_str());
					cinitvel->nz = atof(toks[5].c_str());
					cinitvel->phase = atoi(toks[6].c_str());
				}
			*/
			break;
		case 9:
			break;
	  	case 10:
			break;
		case 11:
			break;
		case 12:
			break;
		case 13:
			break;
		case 14:
			break;
		case 15:
			break;
		case 16:
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
				cpart->eosid = atoi(tok.c_str());
				tok= getTok(buffer,40,10);
				cpart->hgid = atoi(tok.c_str());
				tok= getTok(buffer,50,10);
				cpart->grav = atoi(tok.c_str());
				tok= getTok(buffer,60,10);
				cpart->adpopt = atoi(tok.c_str());
				tok= getTok(buffer,70,10);
				cpart->tmid = atoi(tok.c_str());
			}
			break;
		case 17: //section shell
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
		case 18:
			break;
		case 19: //section solid
			csecsolid = new SecSolid;
			secSolids.push_back(csecsolid);

			tok= getTok(buffer,0,10);
			csecsolid->secid = atoi(tok.c_str());
			tok= getTok(buffer,10,10);
			csecsolid->elform = atoi(tok.c_str());
			tok= getTok(buffer,20,10);
			csecsolid->aet = atoi(tok.c_str());
			break;
		case 20:
			break;
		case 21:
			break;
		case 22:
			break;
		case 23:
			break;
		case 24: //section_beam
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
		case 25: //element_solid
				
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
		case 26: //element_shell
		case 261:
		case 262:
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

				tok= getTok(buffer,64,16);
				celemshell->psi = atof(tok.c_str());

				cardNum=-1; //so the carNum++ at the end will reset it to be 0
				
			}
			break;
		case 27: //element_beam
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
		case 28: // node
			cnode = new Node;
			tok= getTok(buffer,0,8);
			cnode->nid = atoi(tok.c_str());
			tok= getTok(buffer,8,16);
			cnode->x[0] = atof(tok.c_str());
			tok= getTok(buffer,24,16);
			cnode->x[2] = -atof(tok.c_str());
			tok= getTok(buffer,40,16);
			cnode->x[1] = atof(tok.c_str());
			tok= getTok(buffer,56,8);
			cnode->tc = atof(tok.c_str());
			tok= getTok(buffer,0,8);
			cnode->rc = atof(tok.c_str());
			nodes.push_back(cnode);

			break;
		default:
			return -1;
		}
		cardNum++;
		      
    } while(!end_of_file);

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
	std::string result;
	int slen = strlen(buffer);
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
				return result;
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

void k_inp_reader::create_tri()
{
   int num = parts.size(), i, j;
   for(i=0; i<num; i++){
      parts_m[parts[i]->pid] = parts[i];
   }

   num = nodes.size();
   double xmin[3]={1.0e10,1.0e10,1.0e10}, xmax[3]={-1.0e10,-1.0e10,-1.0e10};
   for(i=0; i<num; i++){
      for(j=0; j<3; j++){
         nodes[i]->x[j] /= 1000.0;
	 if(j==1) nodes[i]->x[j] += 0.1;
         if(nodes[i]->x[j]>xmax[j]) xmax[j] = nodes[i]->x[j];
         if(nodes[i]->x[j]<xmin[j]) xmin[j] = nodes[i]->x[j];
      }
      nodes_m[nodes[i]->nid] = nodes[i];
   }
   std::cout<<" min "<< xmin[0]<<" "<<xmin[1]<<" "<<xmin[2]<<std::endl;
   std::cout<<" max "<< xmax[0]<<" "<<xmax[1]<<" "<<xmax[2]<<std::endl;

   num = secShells.size();
   for(i=0; i<num; i++){
      for(j=0; j<4; j++) secShells[i]->t[j] *= TCORRECTION; //10.0/1000.0;
      secShells_m[secShells[i]->secid] = secShells[i];
   }

   std::map<unsigned int, Part*>::iterator itp;
   for(itp=parts_m.begin(); itp!=parts_m.end(); itp++){
      itp->second->numsolid = 0;
      itp->second->numshell = 0;
   }


   std::set<unsigned long> order;
   std::set<unsigned long>::iterator ito;
   std::map<std::string, edge>::iterator ite;
   std::map<unsigned int,SecShell*>::iterator itssh;
   std::map<unsigned int,Node*>::iterator itng;
   std::map<unsigned int,Node>::iterator itn;
   std::map<unsigned long, unsigned long>::iterator itl;
   std::map<unsigned long, ElemShell*>::iterator itesh;

   num = elemShells.size();
   int j1, j2, k, numtri = 0;
   for(i=0; i<num; i++){
      elemShells[i]->sign = 1.0;
      for(j=0; j<4; j++) 
	  {
		  elemShells[i]->t[j] *= TCORRECTION; //10.0/1000.0;
		  //The check here to either put section thickness into elemShells' thickness

		  if (fabs(elemShells[i]->t[j])<1E-12)
		  {
			itp = parts_m.find(elemShells[i]->pid);
			itssh = secShells_m.find(itp->second->secid);
			elemShells[i]->t[j] = itssh->second->t[j];
		  }
	  }
      itp = parts_m.find(elemShells[i]->pid);
	  if(itp!=parts_m.end()){
      for(j1=0; j1<4; j1++){
         itng = nodes_m.find(elemShells[i]->n[j1]);
		 if(itng!=nodes_m.end()){
         itp->second->nodes_m[elemShells[i]->n[j1]] = *itng->second;
		 }
      }
	  }
   }

	

   Node* n0[8];
   int nodeloc[8];
   for(i=0; i<num; i++){
      ElemShell* es = elemShells[i];
      itp = parts_m.find(elemShells[i]->pid);
      itssh = secShells_m.find(itp->second->secid);
      itp->second->numshell++;
      double xx[4][3], v1[3], v2[3], v3[3], xavg[3];
      for(j1=0; j1<4; j1++){
         itn = itp->second->nodes_m.find(elemShells[i]->n[j1]);
         n0[j1] = &itn->second;
         for(k=0; k<3; k++) xx[j1][k] = n0[j1]->x[k];
      } // j1
      if(n0[2]->nid==n0[3]->nid){
         numtri++;
         for(k=0; k<3; k++){
            v1[k] = xx[1][k] - xx[0][k];
            v2[k] = xx[2][k] - xx[0][k];
            xavg[k] = (xx[0][k] + xx[1][k] + xx[2][k])/3.0;
         }
      }else{
         for(k=0; k<3; k++){
            v1[k] = xx[2][k] - xx[0][k];
            v2[k] = xx[3][k] - xx[1][k];
            xavg[k] = 0.25*(xx[0][k] + xx[1][k] + xx[2][k] + xx[3][k]);
         }
      }
      cross_prod(v1, v2, v3);
      double denom = 0.0;
      for(k=0; k<3; k++) denom += v3[k]*v3[k];
      denom = sqrt(denom);
      if(denom) for(k=0; k<3; k++) v3[k] /= denom;
      // add faces (top and bottom)
      // top face
      face fct;
      for(j1=0; j1<4; j1++){
         Node node = *n0[j1];
         for(k=0; k<3; k++){
            node.x[k] += 0.5*elemShells[i]->t[j1]*v3[k];
            node.x[k] = 0.001*xavg[k] + 0.999*node.x[k];
         }
         if(j1<3||n0[2]->nid!=n0[3]->nid) itp->second->nodes_v.push_back(node);
         unsigned int nloc = itp->second->nodes_v.size()-1;
         itp->second->nodes_v[nloc].nidloc = nloc;
         fct.n[j1] = nloc;
         nodeloc[j1] = nloc;
      }
      fct.en.push_back(elemShells[i]->eid);
      itp->second->faces_v.push_back(fct);

      // bottom face
      face fcb;
      for(j1=0; j1<4; j1++){
         Node node = *n0[j1];
         for(k=0; k<3; k++){
            node.x[k] -= 0.5*elemShells[i]->t[j1]*v3[k];
            node.x[k] = 0.001*xavg[k] + 0.999*node.x[k];
         }
         if(j1<3||n0[2]->nid!=n0[3]->nid) itp->second->nodes_v.push_back(node);
         unsigned int nloc = itp->second->nodes_v.size()-1;
         itp->second->nodes_v[nloc].nidloc = nloc;
         fcb.n[j1] = nloc;
         nodeloc[4+j1] = nloc;
      }
      fcb.en.push_back(elemShells[i]->eid);
      itp->second->faces_v.push_back(fcb);
   // construct edge faces
      for(j1=0; j1<4; j1++){
         j2 = j1 + 1;
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
   } // for(i
   std::cout<<"numtri "<<numtri<<std::endl;


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

   std::map<std::string, face>::iterator itf;
   num = elemSolids.size();
   for(i=0; i<num; i++){
      itp = parts_m.find(elemSolids[i]->pid);
      itp->second->numsolid++;
      // add edges, accumulate average thickness and normal direction for each node
      for(j1=0; j1<8; j1++){
         itn = itp->second->nodes_m.find(elemSolids[i]->n[j1]);
         if(itn==itp->second->nodes_m.end()){
            itng = nodes_m.find(elemSolids[i]->n[j1]);
            itp->second->nodes_v.push_back(*itng->second);
            itp->second->nodes_v[itp->second->nodes_v.size()-1].nidloc = itp->second->nodes_v.size()-1;
            itp->second->nodes_m[elemSolids[i]->n[j1]] = itp->second->nodes_v[itp->second->nodes_v.size()-1];
            itn = itp->second->nodes_m.find(elemSolids[i]->n[j1]);
         }
         n0[j1] = &itn->second;
      }
      for(j1=0; j1<8; j1++){
         j2 = j1 + 1;
         if(j2==8) j2 = 0;
         if(elemSolids[i]->n[j1]!=elemSolids[i]->n[j2]){
            if(j1==3) std::cout<<"tetrahedron"<<std::endl;
            if(j1==4) std::cout<<"wedge"<<std::endl;
            break;
         }
      }
      for(j1=0; j1<6; j1++){
         order.clear();
         for(j2=0; j2<4; j2++) order.insert(elemSolids[i]->n[fcver[j1][j2]]);
         std::ostringstream str;
         for(ito=order.begin(); ito!=order.end(); ito++) str <<"a"<<*ito;
         std::string check = str.str();
         itf = itp->second->faces.find(str.str());
         if(itf!=itp->second->faces.end()){
            itf->second.en.push_back(elemSolids[i]->eid);
            if(itf->second.en.size()>2) std::cout<<" more than two elements per face "<<itf->second.en.size()<<std::endl;
         }else{
            face fce;
            for(j2=0; j2<4; j2++) fce.n[j2] = n0[fcver[j1][j2]]->nidloc;
            fce.en.push_back(elemSolids[i]->eid);
            itp->second->faces[str.str()] = fce;
         }
      }
   }
   for(itp = parts_m.begin(); itp != parts_m.end(); itp++){
      for(itf=itp->second->faces.begin(); itf!=itp->second->faces.end(); itf++){
         itp->second->faces_v.push_back(itf->second);
      }
   }
   // write out pts and tri files

   std::vector<std::string> ptsfiles, trifiles, names;
   for(itp = parts_m.begin(); itp != parts_m.end(); itp++){
      num = itp->second->nodes_v.size();
      if(num){
         FILE* pts, *tri;
         std::ostringstream ststm;
         ststm<<"part"<<itp->second->pid;
         std::string st = ststm.str();
         names.push_back(st);
         ststm<<".pts";
         st = ststm.str();
         ptsfiles.push_back(st);
         pts = fopen(st.c_str(),"wt");
         ststm.str("");
         ststm<<"part"<<itp->second->pid<<".tri";
         st = ststm.str();
         trifiles.push_back(st);
         tri = fopen(st.c_str(),"wt");
         
         for(i=0; i<num; i++){
            float rl;
            for(k=0; k<3; k++){
               rl = (float)itp->second->nodes_v[i].x[k];
               fprintf(pts,"%f",rl);
               if(k<2) fprintf(pts," ");
            }
            fprintf(pts,"\n");
         }
         num = itp->second->faces_v.size();
         for(i=0; i<num; i++){
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

   num = ptsfiles.size();

   FILE* mpmice_file;
   mpmice_file = fopen("mpmice_geom","wt");
   fprintf(mpmice_file,"<union>\n");
   int cnt = 0;
   for(i=0; i<num; i++){
      read_elems(trifiles[i],cnt);
      read_vtcs(ptsfiles[i],cnt);
      fprintf(mpmice_file,"<tri><name>%s</name></tri>\n",names[i].c_str());
   }
   fprintf(mpmice_file,"</union>\n");
   fclose(mpmice_file);

   write_uns();

//   for(itp=parts_m.begin(); itp!=parts_m.end(); itp++) 
  //    std::cout<<itp->first<<" "<<itp->second->numsolid<<" "<<itp->second->numshell<<std::endl;
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
   df1 = fopen("uns.vtk","wb");
   fprintf(df1,"# vtk DataFile Version 4.0\n");
   fprintf(df1,"AMR Code Field Data\n");
   fprintf(df1,"BINARY\n");
   fprintf(df1,"DATASET UNSTRUCTURED_GRID\n");
//////////   geom.write_uns_vtk(df1,df2);
   int nv = vertices.size(), iv;
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
   int nc = elements.size(), ic;
   int nc0 = 3;
   int num = nc*(nc0+1);
   int order[8] = {0,1,3,2,4,5,7,6};
   fprintf(df1,"\nCELLS %d %d\n",nc,num);
   swap_4((char*)&nc0);
   float v1[3], v2[3], v3[3];
   int nodes0[3];
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

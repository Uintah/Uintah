/*************************************************************
Copyright (C) 1990, 1991, 1993 Andy C. Hung, all rights reserved.
PUBLIC DOMAIN LICENSE: Stanford University Portable Video Research
Group. If you use this software, you agree to the following: This
program package is purely experimental, and is licensed "as is".
Permission is granted to use, modify, and distribute this program
without charge for any purpose, provided this license/ disclaimer
notice appears in the copies.  No warranty or maintenance is given,
either expressed or implied.  In no event shall the author(s) be
liable to you or a third party for any special, incidental,
consequential, or other damages, arising out of the use or inability
to use the program for any purpose (or the loss of data), even if we
have been advised of such possibilities.  Any public reference or
advertisement of this source code should refer to it as the Portable
Video Research Group (PVRG) code, and not by any author(s) (or
Stanford University) name.
*************************************************************/
/*
************************************************************
huffman.c

This file contains the Huffman routines.  They are constructed to use
no look-ahead in the stream.

************************************************************
*/

/*LABEL huffman.c */

#include <stdio.h>
#include "stream.h"
#include "globals.h"
#include "ctables.h"
#include "csize.h"

/*PUBLIC*/

extern void inithuff();
extern int Encode();
extern int Decode();
extern void PrintDhuff();
extern void PrintEhuff();
extern void PrintTable();

static DHUFF *MakeDhuff();
static EHUFF *MakeEhuff();
static void LoadETable();
static void LoadDTable();
static int GetNextState();
static void AddCode();

/*PRIVATE*/

#define fgetb mgetb
#define fputv mputv
#define fputb mputb

/* Actual Tables */

extern IMAGE *CImage;
extern FRAME *CFrame;
extern int Loud;
extern int ErrorValue;
extern int FrameInterval;

int NumberBitsCoded=0;

#define GetLeft(sval,huff) (((huff->state[(sval)]) >> 16)& 0x0000ffff)
#define GetRight(sval,huff) ((huff->state[(sval)]) & 0xffff)

#define SetLeft(number,sval,huff) huff->state[(sval)]=\
  (((huff->state[(sval)]) & 0xffff)|(number<<16));
#define SetRight(number,sval,huff) huff->state[(sval)]=\
  (((huff->state[(sval)]) & 0xffff0000)|(number));

#define EmptyState 0xffff
#define Numberp(value) ((value & 0x8000) ? 1 : 0)
#define MakeHVal(value) (value | 0x8000)
#define GetHVal(value) (value & 0x7fff)

DHUFF *MBADHuff;
DHUFF *MVDDHuff;
DHUFF *CBPDHuff;
DHUFF *T1DHuff;
DHUFF *T2DHuff;
DHUFF *IntraDHuff;
DHUFF *PredictedDHuff;
DHUFF *InterpolatedDHuff;
DHUFF *DCLumDHuff;
DHUFF *DCChromDHuff;
DHUFF **ModuloDHuff;

EHUFF *MBAEHuff;
EHUFF *MVDEHuff;
EHUFF *CBPEHuff;
EHUFF *T1EHuff;
EHUFF *T2EHuff;
EHUFF *IntraEHuff;
EHUFF *PredictedEHuff;
EHUFF *InterpolatedEHuff;
EHUFF *DCLumEHuff;
EHUFF *DCChromEHuff;
EHUFF **ModuloEHuff;

/*START*/
/*BFUNC

inithuff() initializes all of the Huffman structures to the
appropriate values. It must be called before any of the tables are
used.

EFUNC*/

void inithuff()
{
  BEGIN("inithuff");
  int i,j,count,largest,smallest,size;
  int **MyCoef;

  MyCoef = (int **) calloc(FrameInterval+1,sizeof (int *));
  for(i=2;i<=FrameInterval;i++)
    {
      MyCoef[i]= (int *) calloc(3*(i+2),sizeof (int));

      size = csize[i-1];
      largest = 1 << (size);
      smallest = largest - i;
/*
      printf("%d: Size: %d  Smallest: %d\n",i,size,smallest);
*/
      for(count=j=0;j<smallest;count++,j++)
	{
	  MyCoef[i][3*j] = j;
	  MyCoef[i][3*j+1] = size-1; /* size */
	  MyCoef[i][3*j+2] = count;
	}
      count <<= 1;
      for(j=smallest;j<i;j++,count++)
	{
	  MyCoef[i][3*j] = j;
	  MyCoef[i][3*j+1] = size;
	  MyCoef[i][3*j+2] = count;
	}
      MyCoef[i][3*i] = -1;
      MyCoef[i][3*i+1] = -1;
    }

/*  
  for(i=2;i<=FrameInterval;i++)
    {
      for(j=0;j<i+1;j++)
	{
	  printf("Value: %d  Size: %d  Code: %d\n",
		 MyCoef[i][3*j],
		 MyCoef[i][3*j+1],
		 MyCoef[i][3*j+2]);
	}
    }
  exit(-1);
*/

  MBADHuff = MakeDhuff();
  MVDDHuff = MakeDhuff();
  CBPDHuff = MakeDhuff();
  T1DHuff = MakeDhuff();
  T2DHuff = MakeDhuff();
  IntraDHuff = MakeDhuff();
  PredictedDHuff = MakeDhuff();
  InterpolatedDHuff = MakeDhuff();
  DCLumDHuff = MakeDhuff();
  DCChromDHuff = MakeDhuff();
  ModuloDHuff = (DHUFF **) calloc(FrameInterval+1,sizeof(DHUFF *));

  for(i=2;i<=FrameInterval;i++)
    {
      ModuloDHuff[i] = MakeDhuff();
    }

  MBAEHuff = MakeEhuff(40);
  MVDEHuff = MakeEhuff(40);
  CBPEHuff = MakeEhuff(70);
  T1EHuff = MakeEhuff(8192);
  T2EHuff = MakeEhuff(8192);

  IntraEHuff = MakeEhuff(20);
  PredictedEHuff = MakeEhuff(20);
  InterpolatedEHuff = MakeEhuff(20);
  DCLumEHuff = MakeEhuff(20);
  DCChromEHuff = MakeEhuff(20);
  ModuloEHuff = (EHUFF **) calloc(FrameInterval+1,sizeof(EHUFF *));
  for(i=2;i<=FrameInterval;i++)
    {
      ModuloEHuff[i] = MakeEhuff(i+2);
    }

  LoadDTable(MBACoeff,MBADHuff);
  LoadETable(MBACoeff,MBAEHuff);
  LoadDTable(MVDCoeff,MVDDHuff);
  LoadETable(MVDCoeff,MVDEHuff);
  LoadDTable(CBPCoeff,CBPDHuff);
  LoadETable(CBPCoeff,CBPEHuff);
  LoadDTable(TCoeff1,T1DHuff);
  LoadETable(TCoeff1,T1EHuff);
  LoadDTable(TCoeff2,T2DHuff);
  LoadETable(TCoeff2,T2EHuff);

  LoadDTable(IntraTypeCoeff,IntraDHuff);
  LoadETable(IntraTypeCoeff,IntraEHuff);
  LoadDTable(PredictedTypeCoeff,PredictedDHuff);
  LoadETable(PredictedTypeCoeff,PredictedEHuff);
  LoadDTable(InterpolatedTypeCoeff,InterpolatedDHuff);
  LoadETable(InterpolatedTypeCoeff,InterpolatedEHuff);
  LoadDTable(DCLumCoeff,DCLumDHuff);
  LoadETable(DCLumCoeff,DCLumEHuff);
  LoadDTable(DCChromCoeff,DCChromDHuff);
  LoadETable(DCChromCoeff,DCChromEHuff);
  for(i=2;i<=FrameInterval;i++)
    {
      LoadDTable(MyCoef[i],ModuloDHuff[i]);
      LoadETable(MyCoef[i],ModuloEHuff[i]);
    }
}

/*BFUNC

MakeDhuff() constructs a decoder Huffman table and returns the
structure.

EFUNC*/

static DHUFF *MakeDhuff()
{
  BEGIN("MakeDhuff");
  int i;
  DHUFF *temp;

  temp = MakeStructure(DHUFF);
  temp->NumberStates=1;
  for(i=0;i<512;i++) {temp->state[i] = -1;}
  return(temp);
}

/*BFUNC

MakeEhuff() constructs an encoder huff with a designated table-size.
This table-size, n, is used for the lookup of Huffman values, and must
represent the largest positive Huffman value.

EFUNC*/

static EHUFF *MakeEhuff(n)
     int n;
{
  BEGIN("MakeEhuff");
  int i;
  EHUFF *temp;

  temp = MakeStructure(EHUFF);
  temp->n = n;
  temp->Hlen = (int *) calloc(n,sizeof(int));
  temp->Hcode = (int *) calloc(n,sizeof(int));
  for(i=0;i<n;i++)
    {
      temp->Hlen[i] = -1;
      temp->Hcode[i] = -1;
    }
  return(temp);
}

/*BFUNC

LoadETable() is used to load an array into an encoder table.  The
array is grouped in triplets and the first negative value signals the
end of the table.

EFUNC*/

static void LoadETable(array,table)
     int *array;
     EHUFF *table;
{
  BEGIN("LoadETable");

  while(*array>=0)
    {
      if (*array>table->n)
	{
	  WHEREAMI();
	  printf("Table overflow.\n");
	  exit(ERROR_BOUNDS);
	}
      table->Hlen[*array] = array[1];
      table->Hcode[*array] = array[2];
      array+=3;
    }
}

/*BFUNC

LoadDHUFF() is used to load an array into the DHUFF structure. The
array consists of trios of Huffman definitions, the first one the
value, the next one the size, and the third one the code.

EFUNC*/

static void LoadDTable(array,table)
     int *array;
     DHUFF *table;
{
  BEGIN("LoadDTable");

  while(*array>=0)
    {
      AddCode(array[1],array[2],array[0],table);
      array+=3;
    }
}

/*BFUNC

GetNextState() returns the next free state of the decoder Huffman
structure.  It exits an error upon overflow.

EFUNC*/

static int GetNextState(huff)
     DHUFF *huff;
{
  BEGIN("GetNextState");

  if (huff->NumberStates==512)
    {
      WHEREAMI();
      printf("Overflow\n");
      exit(ERROR_BOUNDS);
    }
  return(huff->NumberStates++);
}

/*BFUNC

Encode() encodes a symbol according to a designated encoder Huffman
table out to the stream. It returns the number of bits written to the
stream and a zero on error.

EFUNC*/

int Encode(val,huff)
     int val;
     EHUFF *huff;
{
  BEGIN("Encode");

  if (val < 0)
    {
      WHEREAMI(); /* Serious error, illegal, notify... */
      printf("Out of bounds val:%d.\n",val);
      return(0);
    }
  else if (val>=huff->n)
    return(0); /* No serious error, can occur with some values */
  else if (huff->Hlen[val]<0)
    return(0);  /* No serious error: can pass thru by alerting routine.*/
  else
    {
/*      printf("Value: %d|%x  Length: %d  Code: %d\n",
	     val,val,huff->Hlen[val],huff->Hcode[val]);*/
      NumberBitsCoded+=huff->Hlen[val];
      fputv(huff->Hlen[val],huff->Hcode[val]);
      return(huff->Hlen[val]);
    }
}


/*BFUNC

Decode() returns a symbol read off the stream using the designated
Huffman structure.

EFUNC*/

int Decode(huff)
     DHUFF *huff;
{
  BEGIN("Decode");
  int Next,cb;
  int CurrentState=0;

  while(1)
    {
      cb = fgetb();
      if (cb)
	{
	  Next =  GetLeft(CurrentState,huff);
	  if (Next == EmptyState)
	    {
	      WHEREAMI();
	      printf("Invalid State Reached.\n");
	      exit(ERROR_BOUNDS);
	    }
	  else if (Numberp(Next))
	    return(GetHVal(Next));
	  else
	    CurrentState = Next;
	}
      else
	{
	  Next =  GetRight(CurrentState,huff);
	  if (Next == EmptyState)
	    {
	      WHEREAMI();
	      printf("Invalid State Reached.\n");
	      exit(ERROR_BOUNDS);
	    }
	  else if (Numberp(Next))
	    return(GetHVal(Next));
	  else
	    CurrentState = Next;
	}
    }
}

/*BFUNC

AddCode() adds a Huffman code to the decoder structure. It is called
everytime a new Huffman code is to be defined. This function exits
when an invalid code is attempted to be placed in the structure.

EFUNC*/

static void AddCode(n,code,value,huff)
     int n;
     int code;
     int value;
     DHUFF *huff;
{
  BEGIN("AddCode");
  int i,Next;
  int CurrentState=0;

  if (value < 0)
    {
      WHEREAMI();
      printf("Negative addcode value: %d\n",value);
      exit(ERROR_BOUNDS);
    }
  for(i=n-1;i>0;i--)
    {
      if (code & (1 << i))
	{
	  Next = GetLeft(CurrentState,huff);
	  if (Next == EmptyState)
	    {
	      Next = GetNextState(huff);
	      SetLeft(Next,CurrentState,huff);
	      CurrentState = Next;
	    }
	  else if (Numberp(Next))
	    {
	      WHEREAMI();
	      printf("Bad Value/State match:\n");
	      printf("Length: %d   Code: %d  Value: %d\n",
		     n,code,value);
	      exit(ERROR_BOUNDS);
	    }
	  else
	    {
	      CurrentState = Next;
	    }
	}
      else
	{
	  Next = GetRight(CurrentState,huff);
	  if (Next == EmptyState)
	    {
	      Next = GetNextState(huff);
	      SetRight(Next,CurrentState,huff);
	      CurrentState = Next;
	    }
	  else if (Numberp(Next))
	    {
	      WHEREAMI();
	      printf("Bad Value/State match:\n");
	      printf("Length: %d   Code: %d  Value: %d\n",
		     n,code,value);
	      exit(ERROR_BOUNDS);
	    }
	  else
	    {
	      CurrentState = Next;
	    }
	}
    }
  if (code & 1)
    {
      Next = GetLeft(CurrentState,huff);
      if (Next != EmptyState)
	{
	  WHEREAMI();
	  printf("Overflow on Huffman Table: Nonunique prefix.\n");
	  printf("Length: %d   Code: %d|%x  Value: %d|%x\n",
		 n,code,code,value,value);
	  exit(ERROR_BOUNDS);
	}
      SetLeft(MakeHVal(value),CurrentState,huff);
    }
  else
    {
      Next = GetRight(CurrentState,huff);
      if (Next != EmptyState)
	{
	  WHEREAMI();
	  printf("Overflow on Huffman Table: Nonunique prefix.\n");
	  printf("Length: %d   Code: %d|%x  Value: %d|%x\n",
		 n,code,code,value,value);
	  exit(ERROR_BOUNDS);
	}
      SetRight(MakeHVal(value),CurrentState,huff);
    }
}


/*BFUNC

PrintDHUFF() prints out the decoder Huffman structure that is passed
into it.

EFUNC*/

void PrintDhuff(huff)
     DHUFF *huff;
{
  int i;

  printf("Modified Huffman Decoding Structure: %x\n",huff);
  printf("Number of states %d\n",huff->NumberStates);
  for(i=0;i<huff->NumberStates;i++)
    {
      printf("State: %d  Left State: %x  Right State: %x\n",
	     i,
	     GetLeft(i,huff),
	     GetRight(i,huff));
    }
}

/*BFUNC

PrintEhuff() prints the encoder Huffman structure passed into it.

EFUNC*/

void PrintEhuff(huff)
     EHUFF *huff;
{
  BEGIN("PrintEhuff");
  int i;

  printf("Modified Huffman Encoding Structure: %x\n",huff);
  printf("Number of values %d\n",huff->n);
  for(i=0;i<huff->n;i++)
    {
      if (huff->Hlen[i]>=0)
	{
	  printf("Value: %x  Length: %d  Code: %x\n",
	     i,huff->Hlen[i],huff->Hcode[i]);
	}
    }
}

/*BFUNC

PrintTable() prints out 256 elements in a nice byte ordered fashion.

EFUNC*/

void PrintTable(table)
     int *table;
{
  int i,j;

  for(i=0;i<16;i++)
    {
      for(j=0;j<16;j++)
	printf("%2x ",*(table++));
      printf("\n");
    }
}

/*END*/

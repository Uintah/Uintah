/* cuthillMcKee.c -- swapnodes to minimize the bandwidth of our matrix */
#include <stdlib.h>
#include <stdio.h>
#include <iostream.h>
#include "vector.h"

typedef struct _link
{
  int val;
  struct _link *next;
} LINK;

LINK *endQueue, *queuePtr;
int *visited;
int currLocation = 1;
int queueSize = 0;


void printMyQueue() {
    LINK *tempPtr = queuePtr;

    printf("Queue: ");
    for (int i=1; i<=queueSize; i++) {
	printf("%d ", tempPtr->val);
	tempPtr=tempPtr->next;
    }
    printf("\n");
}




void deQueue(int *newLocation) {
    LINK *tempPtr = queuePtr;
//cerr << "Making nl[" << queuePtr->val<<"]"<< " "<< currLocation << "\n";
    newLocation[queuePtr->val] = currLocation;
    currLocation++;
    queuePtr = queuePtr->next;
    queueSize--;
}
    
void enQueue(int index) {
    endQueue->next = (LINK *) malloc (sizeof (LINK));
    endQueue = endQueue->next;
    endQueue->val = index;
    queueSize++;
    visited[index] = 1;
}

void enQueueChildren(double **a) {
    int *done;
    int i = queuePtr->val;
    int j,l,m;
    LINK *kids;
    int size, nodeIndex, bigIndex;

    done = (int *) malloc (((int)a[i][0]+1) * sizeof (int));
    for (j=1; j<=a[i][0]; j++)
	done[j]=0;

    for (l=1;l<=a[i][0];l++) {
	size=1000;
//	kids = nbrDBase[i].next;
	for (m=1; m<=a[i][0];m++) {
	    if ((a[(int)a[i][m*2-1]][0] <= size) && (!done[m])) {
		size = (int)a[(int)a[i][m*2-1]][0];
		nodeIndex = (int)a[i][m*2-1];
		bigIndex = m;
	    }
	}
	done[bigIndex] = 1;
	if (!visited[nodeIndex]) {
	    enQueue(nodeIndex);
	    visited[nodeIndex] = 1;
	}
    }
    free(done);
}
    

void startQueue(int index) {
    queuePtr = (LINK *) malloc (sizeof(LINK));
    endQueue = queuePtr;
    queueSize = 1;
    queuePtr->val = index;
    visited[index] = 1;
}

void reversePointOrder(int *newLocation, int numNodes) 
{
  int i;

  for (i=1;i<=numNodes;i++) 
    {
      newLocation[i] = numNodes+1-newLocation[i];
    }
}

int cuthillMcKee(double **a, int numNodes, int *newLocation) {
  int i,j,k;
  int size, index;

  visited = (int *) malloc ((numNodes+1) * sizeof (int));

  for (i=1;i<=numNodes;i++)
    visited[i] = 0;

  int done=0;
  int tries=0;
  while (!done) {
      tries++;
      index=-1;
      size = 8000;  
      for (i=1;i<=numNodes;i++) {
	  if ((int)a[i][0] <= size && visited[i]==0) {
	      size = (int)a[i][0];
	      index=i;
	  }
      }	
      if (index==-1) {
	  cerr << "Coudln't find a node to start with!\n";
	  exit(0);
      }
//      cerr << "Starting with index: "<<index<<"\n";
      startQueue(index);
//      printMyQueue();
      while (queueSize != 0) {
	  enQueueChildren(a);
//	  printMyQueue();
	  deQueue(newLocation);
//	  printMyQueue();
      }
      done=1;
      for (i=1; i<=numNodes; i++) {
	  if (!visited[i]) {
	      done=0;
	  }
      }
  }
  //printIntVector(newLocation, numNodes);
  reversePointOrder(newLocation, numNodes);
  //printIntVector(newLocation, numNodes);
  
  free(visited);
  return tries;
}
  

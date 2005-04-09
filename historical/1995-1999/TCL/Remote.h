
/*
 *  Remote.h: Header file of all remote message formats and socket stuff.
 *   No general sockets include files.  I like to include explicitly what I
 *   need, so I understand what is being pulled into the make.
 *
 *  Written by:
 *   Michelle Miller
 *   Department of Computer Science
 *   University of Utah
 *   Nov. 1997
 *
 *  Copyright (C) 1998 SCI Group
 */

#ifndef SCI_project_Remote_h
#define SCI_project_Remote_h  1

#define SOCKET_TYPE     SOCK_STREAM
#define SOCKET_DOMAIN   AF_INET
#define PROTO           0       // IPPROTO_TCP
#define MAX_Q_LENGTH    10
#define BASE_PORT	8888

#define HOSTNAME	50
#define BUFSIZE         512

#if 0
#define bcopy(src, dst, len)    memcpy(dst, src, len)
#define bzero(src, len)         memset(src, 0, len)
#endif


typedef struct {                // msgBody = 4+80+84 = 168
        char name[80];
        char id[84];
        int  handle;
} CreateModMsg;

typedef struct {		// msgBody = 4+4+4+4+80+4 = 100
        int  outModHandle;
        int  oport;
        int  inModHandle;
        int  iport;
        char connID[80];
        int  connHandle;
} CreateLocConnMsg;

typedef struct {		// msgBody = 100+4+4 = 108, this is wrong!
	bool fromRemote;	// true means data flows from slave to master
	int  outModHandle;
        int  oport;
        int  inModHandle;
        int  iport;
        char connID[80];
        int  connHandle;
	int  socketPort;
} CreateRemConnMsg;

typedef struct {	 	// msgBody = 4
	int modHandle;
} ExecuteMsg;

typedef struct { 		// msgBody = 2*4 = 8
        int modHandle;
        int connHandle;       
} TriggerPortMsg;

typedef struct {	 	// msgBody = 4
	int modHandle;
} DeleteModMsg;

typedef struct {	 	// msgBody = 4
	int connHandle;
} DeleteLocConnMsg;

typedef struct {	 	// msgBody = 4
	int connHandle;
} DeleteRemConnMsg;

typedef struct {
    unsigned    type;
    union {
        CreateModMsg 	cm;
	CreateLocConnMsg clc;
	CreateRemConnMsg crc;
	ExecuteMsg 	e;
	TriggerPortMsg 	tp;
	DeleteModMsg	dm;
	DeleteLocConnMsg dlc;
	DeleteRemConnMsg drc;
    } u;
} Message;

#define CREATE_MOD            	1
#define CREATE_LOC_CONN        	2
#define CREATE_REM_CONN        	3
#define EXECUTE_MOD             4
#define TRIGGER_PORT           	5
#define DELETE_MOD           	6
#define DELETE_LOC_CONN		7
#define DELETE_REM_CONN		8
#define RESCHEDULE             	9
#define MULTISEND              	10

enum Func {
   getDouble,
   getInt,
   getString,
   exec
};

typedef struct {
    char tclName[128];
    Func f;
    union {                     // Return values go in the union
        int     tint;
        double  tdouble;
        char  	tstring[256];
    } un;

} TCLMessage;

// function prototypes
int setupConnect (int port);
int acceptConnect (int in_socket);
int requestConnect (int port, char* host);
int sendRequest (TCLMessage* msg, int skt);
int receiveReply (TCLMessage* msg, int skt);

#endif

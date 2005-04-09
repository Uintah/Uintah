/*              Quarks distributed shared memory system.
 * 
 * Copyright (c) 1995 The University of Utah and
 * the Computer Systems Laboratory (CSL).  All rights reserved.
 *
 * Permission to use, copy, modify and distribute this software is hereby
 * granted provided that (1) source code retains these copyright, permission,
 * and disclaimer notices, and (2) redistributions including binaries
 * reproduce the notices in supporting documentation, and (3) all advertising
 * materials mentioning features or use of this software display the following
 * acknowledgement: ``This product includes software developed by the Computer 
 * Systems Laboratory at the University of Utah.''
 *
 * THE UNIVERSITY OF UTAH AND CSL ALLOW FREE USE OF THIS SOFTWARE IN ITS "AS
 * IS" CONDITION.  THE UNIVERSITY OF UTAH AND CSL DISCLAIM ANY LIABILITY OF
 * ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * CSL requests users of this software to return to csl-dist@cs.utah.edu any
 * improvements that they make and grant CSL redistribution rights.
 *
 * 	Utah $Hdr$
 *	Author: Dilip Khandekar, University of Utah CSL
 */
/**************************************************************************
 *
 * connection.c: connection management for message channels
 *
 *************************************************************************/

#include "connection.h"
#include "buffer.h"
#include "thread.h"

static Connection *contable=0;
static int        contable_ent=0;


/******************************************************************
           routines for connection table handling
 *****************************************************************/

static void init_contable()
{
    int i;

    contable = (Connection *) taballoc((Address) &contable,
				       (Size) sizeof(Connection),
				       &contable_ent);
    for (i=0; i<contable_ent; i++)
	contable[i].nodeid = INVALID_ID;
}

int contable_index(Id nodeid)
{
    int i;

    if (!contable)
	init_contable();

    /* if the node exists in a disconnected fashion in the connection
     * table, send the index. Otherwise it is considered non-existent
     */
    for (i=0; i<contable_ent; i++)
	if ((contable[i].nodeid == nodeid) && 
	    (contable[i].discflag == 0))
		return i;
    
    return -1;
}

Port *contable_port(int ct_index)
{
    return (&contable[ct_index].port);
}

static int free_ctentry()
{
    int i,j;

    if (!contable)
	init_contable();

    for (i=0; i<contable_ent; i++)
	if (contable[i].nodeid == INVALID_ID)
	    return i;
    
    j = contable_ent;

    /* connection table is full. must expand */
    contable = (Connection *) tabexpand((Address) &contable, &contable_ent);

    for (i=j; i<contable_ent; i++)
	contable[i].nodeid = INVALID_ID;

    return j;
}



/******************************************************************
 routines used by the initiator of a (dis)connection to the server 
 *****************************************************************/

void bind_server(char *hostname)
{
    /* Set up the connection table entry for the server. Contact
     * the server and get a nodeid.
     */
    int            i, size, lt_index;
    Port           *sport, from_port;
    struct hostent *ent;
    Message        *msg, *reply;
    Id             nodeid;

    /* initialize the connection table */
    if (!contable)
	init_contable();
    
    /* set up the server entry */
    sport = (Port *) malloc(sizeof(Port));
    sport->sin.sin_family = AF_INET;
    sport->sin.sin_port   = SERVER_PORTNUM;
    fprintf(stderr, "server hostname is %s\n", hostname);

    if (!(ent = gethostbyname(hostname)))
    {
	perror("gethostbyname");
	PANIC("Cannot bind with the server");
    }
    sport->sin.sin_addr.s_addr = *(unsigned long *) ent->h_addr_list[0];
    contable[0].nodeid = 0;
    copy_port(&contable[0].port, sport);

    /* Send message to the server asking a new nodeid. 
     */
    msg = new_buffer();	
    msg->next = 0;
    MSG_SET_FAMILY(msg, MSG_FAMILY_ASYNCH);
    MSG_SET_TYPE(msg, MSG_TYPE_MESSAGE);
    msg->op = MSG_OP_GET_NODEID;
    msg->seqno = 1;
    msg->numfrags = 0;
    msg->fragno = 0;

    msg->length = sizeof(Port);
    ASSERT(localport);
    mem_copy((char *) localport, (char *) msg->data, sizeof(Port));

    msg->to_thread = DSM_SERVER_THREAD;
    msg->from_thread = 0;
    send_packet(sport, (char *) msg + MSG_PACKET_OFFSET, 
		msg->length + HEADER_SIZE);
    
    mumble("indefinite wait for reply from server ");
    reply = receive(0);

    nodeid = extract_long(reply->data, FIRST);
    if (nodeid <= 0)
	PANIC("invalid nodeid ");
    Qknodeid = nodeid;
    if (Qknodeid == 1) Qkmaster = 1;
    
    printf("My nodeid = %d\n", Qknodeid);
    free_buffer(msg);
    free_buffer(reply);
}

void register_server()
{
    /* tell the server that initialization has been done and 
     * I am ready to receive messages, so that the server can
     * give out my address to any other client.
     */
    
    Message *msg, *reply;

    MSG_INIT(msg, MSG_OP_REGISTER_NODE);
    reply = ssend_msg(DSM_SERVER_THREAD, msg);
    if (extract_long(reply->data, FIRST) != 0)
	PANIC("Could not register with the server");
    free_buffer(reply);
}
     
void disconnect_server()
{
    /* contacts the server and tells about terminating connection. 
     */
    Message *msg, *reply;

    MSG_INIT(msg, MSG_OP_FREE_NODEID);
    reply = ssend_msg(DSM_SERVER_THREAD, msg);
    if (extract_long(reply->data, FIRST) != 0)
	PANIC("error in diconnecting server ");
    free_buffer(reply);

    MSG_INIT(msg, MSG_OP_FREE_NODEID_COMMIT);
    asend_msg(DSM_SERVER_THREAD, msg);
    contable[0].nodeid = INVALID_ID;
}



/*********************************************************************
 routines used by the initiator of a (dis)connection to another client
 ********************************************************************/

int get_node_info(Id nodeid)
{
    /* A message is to be sent to node "nodeid", however, there is 
     * no connection to the node. Contact the server and get information 
     * about the node. connect_node_request() will actually make the
     * connection. 
     */
    Message *msg, *reply;
    int lt_index = Thread(thread_self());
    int ct_index;
    int done = 0;

    MSG_INIT(msg, MSG_OP_GET_NODEINFO);
    MSG_INSERT(msg, nodeid);

    while (!done)
    {
	reply = ssend_msg(DSM_SERVER_THREAD, msg);
	if (extract_long(reply->data, FIRST) == 0)
	    done = 1;
	free_buffer(reply);
    }

    ct_index = free_ctentry();
    contable[ct_index].nodeid = nodeid;
    copy_port(&contable[ct_index].port, (Port *) (reply->data+4));

    return 1;
}


int connect_node_request(Id nodeid)
{
    /* This node wants to establish a connection to node "nodeid".
     */
    Message *msg, *reply;
    int lt_index = Thread(thread_self());
    
    MSG_INIT(msg, MSG_OP_CONNECT);
    MSG_INSERT(msg, Qknodeid);
    MSG_INSERT_BLK(msg, localport, sizeof(Port));
    reply = ssend_msg(construct_threadid(nodeid, DSM_THREAD), msg);
    if (extract_long(reply->data, FIRST) != 0)
	PANIC("could not connect to remote node");

    free_buffer(reply);
    return 1;
}

static int disconnect_node_request(Id nodeid)
{
    Message *msg, *reply;

    MSG_INIT(msg, MSG_OP_DISCONNECT);
    reply = ssend_msg(construct_threadid(nodeid, DSM_THREAD), msg);
    if (extract_long(reply->data, FIRST) != 0)
	PANIC("Could not disconnect with remote node");
    free_buffer(reply);
   
    MSG_INIT(msg, MSG_OP_DISCONNECT_COMMIT);
    asend_msg(construct_threadid(nodeid, DSM_THREAD), msg);

    disconnect_node(nodeid);
}
    
void disconnect_all_clients()
{
    /* contacts all the clients connected to the node and tells about
     * terminating connection.
     */
    int ct_index;

    for (ct_index = 0; ct_index < contable_ent; ct_index++)
    {
	if ((contable[ct_index].nodeid != 0)  /* not a server */
	    && (contable[ct_index].nodeid != INVALID_ID))
	{
	    disconnect_node(contable[ct_index].nodeid);
	}
    }
}



/*************************************************************************
 routines used by the client or server receiving a (dis)connection request
 ************************************************************************/

void accept_node(Id nodeid, Port *port)
{	
    /* Node "nodeid" has just come up. It is requesting a connection 
     * establishment. As part of that, register its nodeid and port.
     */
    int ct_index;

    if ((ct_index = contable_index(nodeid)) != -1)  /* already exists */
    {
	if (!cmp_port(&contable[ct_index].port, port))
	    PANIC("Inconsistent port information");
    }
    else
    {
	ct_index = free_ctentry(); 
	contable[ct_index].nodeid = nodeid;
	copy_port(&contable[ct_index].port, port);
    }
}

void disconnect_node_mark(Id nodeid)
{
    /* Node "nodeid" is going down. Mark the entries in contable and
     * rtable as disconnected. Disconnect only after confirmation of
     * receipt of the reply sent from here.
     */
    int ct_index;

    if ((ct_index = contable_index(nodeid)) == -1)
	PANIC("Node does not exist");
    
    contable[ct_index].discflag = 1;
    rtable_setdiscflag(nodeid);
}

void disconnect_node(Id nodeid)
{
    /* Actually clean up the contable and rtable entries to finally
     * disconnect.
     */
    int i;

    for (i=0; i<contable_ent; i++)
	if (contable[i].nodeid == nodeid)
	    break;

    contable[i].nodeid = INVALID_ID;
    contable[i].discflag = 0;
    rtable_clean_nentry(nodeid);
}

int disconnected(Id nodeid)
{
    /* return 1 iff the node entry exists in a disconnected state.
     */
    int i;

    for (i=0; i<contable_ent; i++)
    {
	if ((contable[i].nodeid == nodeid) && 
	    (contable[i].discflag == 1))
	    return 1;
	else
	    return 0;
    }
    return 0;
}    


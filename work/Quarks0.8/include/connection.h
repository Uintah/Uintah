#ifndef _CONNECTION_H_
#define _CONNECTION_H_

#include "message.h"
#include "port.h"
#include "types.h"

typedef struct Connection {
    int      discflag;
    Id       nodeid;
    Port     port;
} Connection;

extern int      contable_index(Id);
extern Port     *contable_port(int);

/* For clients, to connect and disconnect to the server */
extern void     bind_server(char *);
extern void     register_server();
extern void     disconnect_server();
extern int      get_node_info(Id);
extern int      connect_node_request(Id);
extern void     disconnect_all_clients();

/* For clients and servers alike, to connect and three-phase-disconnect
 * to other clients.
 */
extern void     accept_node(Id, Port *);
extern void     disconnect_node_mark(Id);
extern void     disconnect_node(Id);
extern int      disconnected(Id);

#endif  /* _CONNECTION_H_ */

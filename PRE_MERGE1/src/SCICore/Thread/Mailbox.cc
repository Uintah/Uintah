
#include "Mailbox.h"
#include "Thread.h"

/*
 * A thread-safe, fixed-length FIFO queue which allows multiple
 * concurrent senders and receivers.  Multiple threads send <b>Item</b>s
 * to the mailbox, and multiple thread may receive <b>Item</b>s from the
 * mailbox.  Items are typically pointers to a message structure.
 */


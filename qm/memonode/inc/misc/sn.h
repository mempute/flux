#ifndef _H_SN
#define _H_SN

#include "misc/pt.h"
#ifndef OPT_WIN
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#endif

#define IP_ADDR_LEN		129
#define NON_BLOCK_CONNECT		1	//연결할때만 비블럭으로
#define NON_BLOCK_STILL			2	//계속 비블럭으로

typedef struct {
	intx aiFamily;
	intx aiSocktype;
    intx aiProtocol;
	intx aiAddrLen;
	intx resolveInfo;
	struct sockaddr_in aiAddrExp;
	struct sockaddr *aiAddr;
} SockAddrInfo;

#ifdef __cplusplus
extern "C" {
#endif
extern intx tcpopen();
extern void xshutdown(sockx);
extern void xshutdown2(sockx fd, intx front_end);
extern intx _seraire(bytex *svc_name, bytex *bound_ip, intx ip_port);
extern intx seraire(bytex *, intx);
extern intx seristen(intx, struct sockaddr_in *);
extern uintx sitename2eaddr(bytex *h_namea);
extern intx _cliaire(bytex *h_namea, bytex *bound_ip, bytex *sf_net_svc, intx sf_net_port, intx non_blk, intx tmout);
extern intx  cliaire(bytex *, bytex *, intx);
extern intx  cliaire2(uintx, bytex *, intx);
// mission critical append
extern intx gethostnaddr(); 
extern intx gethostcaddr(bytex *);
extern intx gethostcaddr2(bytex *ip, bytex *sample);
extern intx soc_nonblock(intx soc, intx nowait);

extern intx trans(intx p, bytex *buf, intx len);
extern intx ftrans(intx p, bytex *buf, intx len);
extern intx ftrans2(intx p, bytex *buf, intx head_len, intx data_len);
extern intx ftrans3(intx p, bytex *buf, intx head_len, intx data_len);
extern intx rntrans(intx p, bytex *buf, intx len);
extern intx trans2(intx p, bytex *head, intx hlen, bytex *data, intx dlen);
extern intx lentrans(intx p, intx len);
extern intx catche(intx p, bytex *buf, uintx alen);
extern intx rncatche(intx p, bytex *buf, intx len);
extern intx catche2(intx p, bytex *head, intx hlen, bytex *data, intx dlen);
extern void error_display();

extern intx nonblock_connect(int sockfd, const struct sockaddr *saptr, intx salen, intx tmout);
extern intx isSameAddr(SockAddrInfo *ai_info, bytex *host);
extern sockx GetAddrInfo_(SockAddrInfo *ai_info, bytex *host, bytex *serv, intx passive);
extern void GetPairSocket(bytex *channel_addr, intx mcast, bytex *serv, SockAddrInfo *send_addr, 
							sockx &recvfd, sockx &sendfd, intx loop);
extern sockx GetSendSocket(bytex *channel_addr, intx mcast, bytex *serv, SockAddrInfo *send_addr, intx loop);

#ifdef	NON_GETADDRINFO_PROTO
extern int ga_aistruct(struct addrinfo ***, const struct addrinfo *, const void *, int);
extern struct addrinfo		*ga_clone(struct addrinfo *);
extern int ga_echeck(const char *, const char *, int, int, int, int);
extern int ga_nsearch(const char *, const struct addrinfo *, struct search *);
extern int ga_port(struct addrinfo *, int , int);
extern int ga_serv(struct addrinfo *, const struct addrinfo *, const char *);
extern int ga_unix(const char *, struct addrinfo *, struct addrinfo **);
int		gn_ipv46(char *, size_t, char *, size_t, void *, size_t, int, int, int);
#endif

#ifdef __cplusplus
}
#endif

#endif

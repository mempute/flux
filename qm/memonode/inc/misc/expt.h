#ifndef _H_ERR
#define _H_ERR

#include "misc/pt.h"
#define ERRMSG_SZ	1024
#define ERR_MSG_HEAD	"Error Code: %d "
class ExptObject {
public:
	intx ercd;
	intx eroff;
	bytex msg[ERRMSG_SZ];
};

#ifdef __cplusplus
extern "C" {
#endif
    
extern intx throwExpt(intx texpt, char *fmt, ...);
extern void replaceExpt(ExptObject *e, intx replace_ecd, char *fmt, ...);
extern intx relayExpt(ExptObject e, char *fmt, ...);
extern intx bypassExpt(intx texpt, bytex *er_msg);
extern void debugStamp(bytex *msg);
extern void debug_point_func(bytex *msg);
//OS system errorso가 실행한 모든 변경작업을 취소하고 so thread exit
#define EXPT_LEVEL1	-1 
//xdb program error
#define EXPT_LEVEL2	-2
//user error
#define EXPT_LEVEL3	-3

#define EXPT_LEVEL4	-4

#define EXPT_LEVEL5	-5

#define EXPT_HOLT_REQ	-6
#ifdef __cplusplus
}
#endif

#endif

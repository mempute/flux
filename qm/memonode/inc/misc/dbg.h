#ifndef _H_DBG
#define _H_DBG
#include "misc/pt.h"

extern divadx *p_addr_dbg;

#ifdef __cplusplus
extern "C" {
#endif

extern void init_time_stamp(void);
extern void print_time_stamp(bytex *msg);

#define InitDbgPoint(p_dbg, ptr) if(p_dbg == 0) p_dbg = (divadx *)ptr;

extern void init_dbg_log(void);
extern void print_dbg_message(char *fmt, ...);
extern void print_dbg_message2(char *fmt, ...);
extern void print_dbg_message3(char *fmt, ...);
extern void break_dbg_point(divadx *p_adr, bytex *msg);
#ifdef DBG_LOG
extern void regist_dbg_pointer(void *vp, bytex *nm);
extern void remove_dbg_pointer(void *vp);
extern void print_dbg_pointer(void);
#else
#define regist_dbg_pointer(vp, nm)
#define remove_dbg_pointer(vp)
#define print_dbg_pointer()
#endif

#define msgdisp print_dbg_message //에러메시지 출력

#ifdef DBG_LOG5 //약한 경고성 메제지
#define tracechk print_dbg_message
#else
#define tracechk
#endif

#ifdef DBG_LOG4
#define tracechk6(a, b, c, d, e, f) print_dbg_message(a, b, c, d, e, f)
#else
#define tracechk6(a, b, c, d, e, f)
#endif

#ifdef DBG_LOG3 //단위문제 복잡 디버깅 목적 로그
#define tracelog(a) print_dbg_message(a)
#define tracelog2(a, b) print_dbg_message(a, b)
#define tracelog3(a, b, c) print_dbg_message(a, b, c)
#define tracelog4(a, b, c, d) print_dbg_message(a, b, c, d)
#define tracelog5(a, b, c, d, e) print_dbg_message(a, b, c, d, e)
#define tracelog6(a, b, c, d, e, f) print_dbg_message(a, b, c, d, e, f)
#define tracelog7(a, b, c, d, e, f, g) print_dbg_message(a, b, c, d, e, f, g)
#define tracelog8(a, b, c, d, e, f, g, h) print_dbg_message(a, b, c, d, e, f, g, h)
#define tracelog9(a, b, c, d, e, f, g, h, i) print_dbg_message(a, b, c, d, e, f, g, h, i)
#else
#define tracelog(a)
#define tracelog2(a, b)
#define tracelog3(a, b, c)
#define tracelog4(a, b, c, d)
#define tracelog5(a, b, c, d, e)
#define tracelog6(a, b, c, d, e, f)
#define tracelog7(a, b, c, d, e, f, g)
#define tracelog8(a, b, c, d, e, f, g, h)
#define tracelog9(a, b, c, d, e, f, g, h, i)
#endif

#ifdef DBG_LOG2 //단순 프로우 체크 디버깅 목적 로그
#define printdbg print_dbg_message
#define printdbg2(a, b) print_dbg_message(a, b)
#define printdbg3(a, b, c) print_dbg_message(a, b, c)
#define printdbg4(a, b, c, d) print_dbg_message(a, b, c, d)
#define printdbg5(a, b, c, d, e) print_dbg_message(a, b, c, d, e)
#define printdbg6(a, b, c, d, e, f) print_dbg_message(a, b, c, d, e, f)
#define printdbg7(a, b, c, d, e, f, g) print_dbg_message(a, b, c, d, e, f, g)
#define printdbg8(a, b, c, d, e, f, g, h) print_dbg_message(a, b, c, d, e, f, g, h)
#else
#define printdbg
#define printdbg2(a, b)
#define printdbg3(a, b, c)
#define printdbg4(a, b, c, d)
#define printdbg5(a, b, c, d, e)
#define printdbg6(a, b, c, d, e, f)
#define printdbg7(a, b, c, d, e, f, g)
#define printdbg8(a, b, c, d, e, f, g, h)
#endif

#ifdef MSG_LOG //전체적 정상적 플로우 체크 로그
#define printmesg(a) print_dbg_message(a)
#define printmesg2(a, b) print_dbg_message(a, b)
#define printmesg3(a, b, c) print_dbg_message(a, b, c)
#define printmesg4(a, b, c, d) print_dbg_message(a, b, c, d)
#else
#define printmesg(a)
#define printmesg2(a, b)
#define printmesg3(a, b, c)
#define printmesg4(a, b, c, d)
#endif

#ifdef __cplusplus
}
#endif

#endif

#ifndef _H_FIO
#define _H_FIO

#include "misc/xt.h"
//#include "misc/imem.h" //mnj@
#include <fcntl.h>
#include <errno.h>

#define EXPT_RET_XFILE_ERROR		-10
#define EXPT_RET_XFILE_SET_ERR		-11

#define FILE_PATH_LEN	1024
#define INODE_STR_LEN	32

typedef struct XFileContain_ {
	bytex *xfileName, *xfilePath;
	struct XFileContain_ *ptrLeft, *ptrRight;
} XFileContain;//k125so.

#ifdef OPT_WIN
#include <io.h>
/*
#define XO_RDWR	O_RDWR | O_BINARY
#define XO_RDONLY O_RDONLY | O_BINARY
#define openx(a, b, c)			_open(a, b, c)
#define closex(a)			_close(a)
#define writex(a, b, c)		_write(a, b, c)
#define readx(a, b, c)		_read(a, b, c)
#define lseekx(a, b, c)		_lseek(a, b, c)
#define flushx(a)

#define XFileOpen(fname, fp) fp = openx(fname, XO_RDWR | O_CREAT, 0777)
#define XFileAppendOpen(fname, fp) fp = openx(fname, XO_RDWR | O_CREAT | O_APPEND, 0777)
#define XFileTruncOpen(fname, fp) fp = openx(fname, XO_RDWR | O_CREAT | O_TRUNC, 0777) 
#define XFileRDOpen(fname, fp) fp = openx(fname, XO_RDONLY, 0777)
#define XFileWROpen(fname, fp) fp = openx(fname, XO_RDWR, 0777)
#define XFileSize(fp, size) size = lseekx(fp, 0, SEEK_END)
typedef intx filex;
*/
#ifdef OPT_ADDR64
#define lseekx(a, b, c)		_fseeki64(a, b, c)
#define ltellx(a)			_ftelli64(a)
#else
#define lseekx(a, b, c)		fseek(a, b, c)
#define ltellx(a)			ftell(a)
#endif
#else

#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <errno.h>
/*
#define XO_RDWR	O_RDWR
#define XO_RDONLY O_RDONLY
#define openx(a, b, c)		open(a, b, c)
#define closex(a)			close(a)
#define writex(a, b, c)		write(a, b, c)
#define readx(a, b, c)		read(a, b, c)
#define lseekx(a, b, c)		lseek(a, b, c)
#define flushx(a)			fsync(a)

#define XFileOpen(fname, fp) fp = openx(fname, XO_RDWR | O_CREAT, 0777)
#define XFileFailOpen(fp) fp < 0
#define XFileAppendOpen(fname, fp) fp = openx(fname, XO_RDWR | O_CREAT | O_APPEND, 0777)
#define XFileTruncOpen(fname, fp) fp = openx(fname, XO_RDWR | O_CREAT | O_TRUNC, 0777)
#define XFileRDOpen(fname, fp) fp = openx(fname, XO_RDONLY, 0777)
#define XFileWROpen(fname, fp) fp = openx(fname, XO_RDWR, 0777)
#define XFileSize(fp, size) size = lseekx(fp, 0, SEEK_END)
typedef intx filex;
*/

#define lseekx(a, b, c)		fseek(a, b, c)
#define ltellx(a)			ftell(a)


#endif

#define openx(a, b, c)		fopen(a, b)
#define closex(a)			fclose(a)
#define writex(a, b, c)		fwrite(b, 1, c, a)
#define readx(a, b, c)		fread(b, 1, c, a)
#define flushx(a)			fflush(a)

typedef FILE *filex;
#define XFileOpen(fname, fp) if((fp = openx(fname, "r+b", 0777)) == nullx) fp = openx(fname, "w+b", 0777)
#define XFileFailOpen(fp) fp == nullx
#define XFileAppendOpen(fname, fp) fp = openx(fname, "a+b", 0777)
#define XFileTruncOpen(fname, fp) fp = openx(fname, "w+b", 0777) 
#define XFileRDOpen(fname, fp) fp = openx(fname, "rb", 0777)
#define XFileWROpen(fname, fp) fp = openx(fname, "r+b", 0777)
#define XFileClose(fp) closex(fp)
#define XFileFlush(fp) 
#define XFileFlush3(fp) if(flushx(fp) != 0) throwExpt(EXPT_RET_XFILE_ERROR, (bytex *)"xfile flush fail\n")
#define XFileSet(fp, off) if(lseekx(fp, off, SEEK_SET) != 0) throwExpt(EXPT_RET_XFILE_SET_ERR, (bytex *)"xfile set fail %u\n", off) //k369_15so.
#define XFileRemove(fname) remove(fname)
#define XFileRename(old_name, new_name) if(rename(old_name, new_name) < 0) throwExpt(EXPT_RET_XFILE_ERROR, "xfile rename fail %d %s\n", errno, strerror(errno))
		
#define XFileCur(fp, off) lseekx(fp, off, SEEK_CUR) //k125so.
#define XFileEnd(fp, off) lseekx(fp, off, SEEK_END) 
#define XFileGet(fp, off) off = ltellx(fp)
#define XFileSize(fp, size) {\
	if(lseekx(fp, 0, SEEK_END) != 0) throwExpt(EXPT_RET_XFILE_ERROR, "CRITICAL: xfile size seek fail\n");\
	size = ltellx(fp);\
	if(size < 0) throwExpt(EXPT_RET_XFILE_ERROR, "CRITICAL: xfile size ftell fail\n");\
}

#define XFileWrite(fp, data, size) {\
	sizex i_, j_;\
	sizex rv_;\
	for(i_ = j_ = 0;; ) {\
		rv_ = writex(fp, data + i_, size - i_);\
		if(rv_ < 0) throwExpt(EXPT_RET_XFILE_ERROR, (bytex *)"CRITICAL: xfile write fail\n");\
		i_ = i_ + rv_;\
		if(i_ < size) {\
			msgdisp((bytex *)"CRITICAL: xfile write incomplete size: %u wbyte: %u\n", size, i_);\
			if(++j_ > 100) throwExpt(EXPT_RET_XFILE_ERROR, (bytex *)"xfile write fail2\n");\
		} else break;\
	}\
}
#define XFileRead_(fp, data, size, i) {\
	sizex rv_;\
	for(i = 0; i < size; ) {\
		rv_ = readx(fp, data + i, size - i);\
		if(rv_ <= 0) break;\
		i = i + rv_;\
	}\
	if(i != size) {\
		throwExpt(EXPT_RET_XFILE_ERROR, (bytex *)"CRITICAL: xfile read fail %d %d %s\n", i, size, strerror(errno));\
	}\
}
#define XFileRead(fp, data, size) {\
	sizex i_;\
	XFileRead_(fp, data, size, i_);\
}
//?½ì? ?¬ì´ì¦ˆê? 0?´ë©´ ???½ì? ê²ƒìœ¼ë¡?ê°„ì£¼?˜ê³  ì¢…ë£Œ?œë‹¤. ì¦?sizeë§Œí¼ ?½ì??Šê³  ?°ì´?°ê? ?ˆëŠ” ë§Œí¼ ?½ëŠ”??
#define XFileReadEOF(fp, data, size, i) {\
	sizex rv_;\
	for(i = 0; i < size; ) {\
		rv_ = readx(fp, data + i, size - i);\
		if(rv_ == 0) break;\
		else if(rv_ < 0) throwExpt(-1, "XFileRead2 fail\n");\
		i = i + rv_;\
	}\
}

#define XFileBaseWrite(fp, off, data, size) {\
	XFileSet(fp, off);\
	XFileWrite(fp, data, size);\
}
#define XFileBaseRead(fp, off, data, size) {\
	XFileSet(fp, off);\
	XFileRead(fp, data, size);\
}

//?ˆë„?°ì´??? ë‹‰?¤ì´??fopenë¥˜ë? ?¬ìš©?˜ëŠ” ê³³ì—???¬ìš©
#define XFileOpen2(fname, fp) fp = fopen(fname, "r+b")
#define XFileTruncOpen2(fname, fp) fp = fopen(fname, "w+b")
#define XFileAppendOpen2(fname, fp) fp = fopen(fname, "a+b")
#define XFileClose2(fp) fclose(fp)
#define XFileFlush2(fp) fflush(fp)
#define XFileWrite2(fp, data, size) fwrite(data, size, 1, fp)
#define XFileRead2(fp, data, size) fread(data, size, 1, fp)
#define XFileSet2(fp, off) lseekx(fp, off, SEEK_SET)
#define XFileCur2(fp, off) lseekx(fp, off, SEEK_CUR) //k125so.
#define XFileEnd2(fp, off) lseekx(fp, off, SEEK_END) 
#define XFileGet2(fp, off) off = ltellx(fp)
#define XFileSize2(fp, size) {\
	lseekx(fp, 0, SEEK_END);\
	size = ltellx(fp);\
}

typedef intx (*FPXFileSearch)(void *, bytex *path, bytex *slot);//k61so.

#ifdef __cplusplus
extern "C" {
#endif
#ifdef OPT_WIN
//?ˆìœ¼ë©?1, ?†ìœ¼ë©?0	(_access???†ìœ¼ë©?-1, ?ˆìœ¼ë©?0)
#define XFileExist(path) (!_access(path, 0))
#else
#define XFileExist(path) (!access(path, 0))
//extern intx XFileExist(bytex *path);

#endif


extern void XFileGetInode(bytex *fname, bytex *inode);
extern void XFileGetTime(bytex *fname, shortx gtime[]);
extern intx XFileMakePath(bytex *path);
extern intx XFileList(FPXFileSearch call_back_fp, void *user_data, bytex *path, bytex *slot);//k61so.
//extern XFileContain* XFileList2(InstantMem *imem, bytex *path, bytex *exp); //mnj@
extern intx XFileNextFile(decx handle, bytex *get_file, bytex *exp);
extern decx XFileOpenDir(bytex *path, bytex *get_file, bytex *exp);

#ifdef __cplusplus
}
#endif

#endif

#pragma once

#if OPT_LINUX
#include <iconv.h>
#endif
#include "misc/mtree.h"
#include "qnet/qapi.h"
#include "misc/mut.h"
#include "misc/xf.h"
#include "memput.h"

class DimenItem {
public:
	intx idxKernel;
	bytex *vdimen;//feature size만큼의 연속 공간
	DimenItem *ptrLeft, *ptrRight;//하나의 시퀀스를 구성하는 feature들의 리스트
};//시퀀스 단위
class SeqItem {
public:
	DimenItem *vseq;//한개 시퀀스 적재
	SeqItem *ptrLeft, *ptrRight;//시퀀스 리스트 구성
};
class StrideBind {
public:
	intx iStrBind;
	SeqItem *seqList;//스트라이드 단위내의 시퀀스 리스트 적재
	StrideBind *ptrLeft, *ptrRight;//시퀀스 리스트 구성
};
class BindItem {
public:
	intx numBind;
	intx cntStride;
	intx nmaxReduce;
	StrideBind *strideList;
	BindItem *ptrLeft, *ptrRight;
};//한개 소스 시퀀스로부터 파생되는 조합 패스 시퀀스들의 소스 시퀀스 단위 묶음

class QCLink {
public:
	//intx cntResetQcl;
	QueryContext qText;
	QCLink *ptrLeft2, *ptrRight2, *ptrLeft3, *ptrRight3;
	void initQCLink(void);
	void closeQCLink(void);
};
class SemLink {//한개 transform요청애 대응하여 한개 생성
public:
	sytex waitOn;
	intx reqCnt, prsCnt;
	intx reqWidth;
	intx nExcContig;
	lsemx semSig;
	hmutex mutSig;
	SemLink *ptrLeft2, *ptrRight2, *ptrLeft3, *ptrRight3;
	intx initSLink(intx ith);
	void closeSLink(void);
	void resetSLink(void);
	void returnSignal(void);
	void waitSignal(void);
};
class MPClient;
class ReqLink {
public:
	intx iStride;
	QCLink *qcrl;
	SemLink *returnSig;
	DimenItem *reqKernel;//한개 소스 시퀀스를 매 스트라이드 마다 커널(윈도우) 사이즈 만큼의 시퀀스 조각
	BindItem *returnBind;//한개 소스 시퀀스로부터 파생되는 조합 패스 시퀀스들을 받을 소스 시퀀스 단위 묶음
	MPClient *mpc;
	ReqLink *ptrLeft, *ptrRight, *ptrLeft2, *ptrRight2, *ptrLeft3, *ptrRight3;

	void requestActor(SemLink *sl, DimenItem *ditem, BindItem *bitem);
	void executeReq(void);
};

class MPClient {
public:
	QCLink *qcLink2, *qcLink3;
	SemLink *semLink2, *semLink3;
	ReqLink *reqLink, *reqLink2, *reqLink3;
	hmutex mutmpc, mutmpc2;
	lsemx sigmpc, sigmpc2;
	QueryContext qcResource, *qcres;
	intx idxSLink, rntmpc, wntmpc, cntActor;
	intx limreq, cntreq, cntwreq;

	MPClient();
	~MPClient();

	SemLink *getSLink(void);
	void putSLink(SemLink *sl);
	QCLink *getQLink(sytex block);
	void putQLink(QCLink *qcl, sytex);
	ReqLink *getRLink(void);
	void putRLink(ReqLink *rl);
	void mpcActor(void);
	void mpcRequest(ReqLink *rl);
	void bootActor(intx cnt_act, intx lim_req);
};

typedef void *(*array_generfp)(void *tcr, intx ndim, intx dims[], intx dt);
typedef void *(*array_datafp)(void *arobj);

extern void *transform(void *b, void *e, intx dt, intx ndims, intx dims[], intx width, intx stride, doublex exc_contig_r,
	sytex zero_pading, sytex src_make, void *tcr, array_generfp agen, array_datafp adat, void *&src_array, bool one_batch);
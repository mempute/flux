
#include "rsc.h"

using namespace memput::mp;

void QCLink::initQCLink(void)
{
	InitSelPage((&qText));
}
void QCLink::closeQCLink(void)
{
	ReleaseSelPage((&qText));
}

intx SemLink::initSLink(intx ith)
{
	intx rv;

	CRE_LSEM_(semSig, 0, ith, rv);
	if(rv < 0) return -1;

	CRE_MUT_(mutSig, rv);
	if(rv < 0) return -1;

	return 0;
}
void SemLink::closeSLink(void)
{
	CLOSE_MUT_(mutSig);
	CLOSE_LSEM_(semSig);
}
void SemLink::returnSignal(void)
{
	LOCK_MUT_(mutSig);
	prsCnt++;
	if(waitOn && reqCnt == prsCnt) SIG_LSEM_(semSig);//조합 패스 생성 요청즉에서 대기상태이고 요청 건수를 모두 처리했으면 시그널
	UNLOCK_MUT_(mutSig);
}
void SemLink::waitSignal(void)
{
	LOCK_MUT_(mutSig);
	waitOn = 1;
	if(prsCnt < reqCnt) {//요청 건수를 모두 처리할때까지 대기
		UNLOCK_MUT_(mutSig);
		WAIT_LSEM_(semSig);
	} else UNLOCK_MUT_(mutSig);//요청 건수를 모두 처리했으면 대기않음.
}
void SemLink::resetSLink(void)
{
	waitOn = 0;
	reqCnt = prsCnt = 0;
}

extern StrideBind *make_array_batch(QueryContext *qc, DimenItem *kernel, intx width, intx exc_cnt, intx &nseq);
void ReqLink::requestActor(SemLink *sl, DimenItem *ditem, BindItem *bitem)
{
	returnSig = sl;
	reqKernel = ditem;
	returnBind = bitem;
	LOCK_MUT_(returnSig->mutSig);
	iStride = bitem->cntStride++;
	returnSig->reqCnt++;
	UNLOCK_MUT_(returnSig->mutSig);

	mpc->mpcRequest(this);
}
void ReqLink::executeReq(void)
{
	StrideBind *strb, *head, *str_bind;
	intx nseq;

	str_bind = make_array_batch(&qcrl->qText, reqKernel, returnSig->reqWidth, returnSig->nExcContig, nseq);
	if(str_bind == nullx) goto LB1;

	LOCK_MUT_(returnSig->mutSig);
	str_bind->iStrBind = iStride;
	head = returnBind->strideList;
	if(head) {
		for(strb = head->ptrLeft;; strb = strb->ptrLeft) {
			if(strb == head || strb->iStrBind < str_bind->iStrBind) break;
		}
		if(strb->iStrBind > str_bind->iStrBind) {
			if(head != strb) throwFault(-1, "absord exec req error\n");
			HEAD_LIST(head, str_bind);
			returnBind->strideList = head;
		} else if(strb->ptrRight) {
			INSERT_AFTER(strb, str_bind);
		} else APPEND_LIST(head, str_bind);
	} else {
		APPEND_LIST(head, str_bind);
		returnBind->strideList = head;
	}
	for(str_bind = head->ptrLeft;str_bind != head; str_bind = str_bind->ptrLeft) {//순서 체크 디버깅용
		for(strb = head;strb != str_bind; strb = strb->ptrRight) {
			if(strb->iStrBind > str_bind->iStrBind) {
				printf("reverse early: %d later: %d\n", strb->iStrBind, str_bind->iStrBind);exit(1);
			}
			//else printf("early: %d later: %d\n", strb->iStrBind, str_bind->iStrBind);
		}
	}
	if(returnBind->nmaxReduce < nseq) returnBind->nmaxReduce = nseq;
	returnBind->numBind += nseq;
	UNLOCK_MUT_(returnSig->mutSig);
LB1:;
	returnSig->returnSignal();
}
#define PYART_BASE	600000

MPClient::MPClient()
{
	intx rv;

	memset(this, 0x00, sizeof(MPClient));
	qcres = &qcResource;
	InitSelPage(qcres);

	CRE_MUT_(mutmpc, rv);
	if(rv < 0) {
		throwFault(-1, "init CRE_MUT_ fail");
	}
	CRE_MUT_(mutmpc2, rv);
	if(rv < 0) {
		throwFault(-1, "init CRE_MUT_ fail");
	}
	CRE_LSEM_(sigmpc, 0, PYART_BASE, rv);
	if(rv < 0) {
		printf("get sig mb CRE_USEM_ fail");
		throwFault(-1, "get sig mb CRE_USEM_ fail\n");
	}
	CRE_LSEM_(sigmpc2, 0, (PYART_BASE + ++idxSLink), rv);
	if(rv < 0) {
		printf("get sig mb CRE_USEM_ fail");
		throwFault(-1, "get sig mb CRE_USEM_ fail\n");
	}
}
MPClient::~MPClient()
{
	ReqLink rl;
	rl.ptrLeft = rl.ptrRight = &rl;//모든 actor 쓰레드가 계속 반복적으로 이 rl을 get list하게 설정한다.
	rl.reqKernel = nullx;//actor 쓰레드 종료 메세지
	mpcRequest(&rl);//모든 actor 쓰레드 종료

	CLOSE_MUT_(mutmpc);
	CLOSE_MUT_(mutmpc2);
	CLOSE_LSEM_(sigmpc);
	CLOSE_LSEM_(sigmpc2);

	CAT_LIST3(SemLink, semLink2, semLink3);
	for(;semLink2; semLink2 = semLink2->ptrRight2) semLink2->closeSLink();
	CAT_LIST3(QCLink, qcLink2, qcLink3);
	for(;qcLink2; qcLink2 = qcLink2->ptrRight2) qcLink2->closeQCLink();

	ReleaseSelPage(qcres);
}
SemLink *MPClient::getSLink(void)
{
	SemLink *sl;

	LOCK_MUT_(mutmpc);
	if(semLink2) {
		GET_LIST2(semLink2, sl);//pool list
		sl->resetSLink();
		UNLOCK_MUT_(mutmpc);
		return sl;
	}
	SelAlloc(qcres, sizeof(SemLink), *(bytex **)&sl);
	if(sl->initSLink(PYART_BASE + ++idxSLink) < 0) {
		UNLOCK_MUT_(mutmpc);
		throwFault(-1, "get slink init slink fail");
	}
	APPEND_LIST3(semLink3, sl);//해재용 보관 리스트
	UNLOCK_MUT_(mutmpc);
	sl->resetSLink();

	return sl;
}
void MPClient::putSLink(SemLink *sl)
{
	LOCK_MUT_(mutmpc);
	APPEND_LIST2(semLink2, sl);
	UNLOCK_MUT_(mutmpc);
}
QCLink *MPClient::getQLink(sytex block)
{
	QCLink *qcl;

	if(block) LOCK_MUT_(mutmpc);
	if(qcLink2) {
		GET_LIST2(qcLink2, qcl);
		if(block) UNLOCK_MUT_(mutmpc);
		return qcl;
	}
	SelAlloc(qcres, sizeof(QCLink), *(bytex **)&qcl);
	memset(qcl, 0x00, sizeof(QCLink));
	qcl->initQCLink();
	APPEND_LIST3(qcLink3, qcl);
	if(block) UNLOCK_MUT_(mutmpc);

	return qcl;
}
void MPClient::putQLink(QCLink *qcl, sytex block)
{
	QueryContext *qc = &qcl->qText;

	if(block) LOCK_MUT_(mutmpc);
	//if(qcl->cntResetQcl++ < 100) {
	ResetSelPage(qc);
	//} else {
	//	qcl->cntResetQcl = 0;
	//	ReleaseSelPage(qc);
	//	InitSelPage(qc);
	//}
	APPEND_LIST2(qcLink2, qcl);
	if(block) UNLOCK_MUT_(mutmpc);
}
ReqLink *MPClient::getRLink(void)
{
	ReqLink *rl;
LB1:;
	LOCK_MUT_(mutmpc);
	if(reqLink2) {
		GET_LIST2(reqLink2, rl);
		rl->qcrl = getQLink(0);
		UNLOCK_MUT_(mutmpc);
		return rl;
	}
	if(cntreq == limreq) {//transform요청은 병렬로 수행될수있고 limreq가 최소 한 계정의 요청수 보단 커야한다.
		cntwreq++;
		UNLOCK_MUT_(mutmpc);
		WAIT_LSEM_(sigmpc);
		goto LB1;
	}
	cntreq++;
	SelAlloc(qcres, sizeof(ReqLink), *(bytex **)&rl);
	memset(rl, 0x00, sizeof(ReqLink));
	rl->mpc = this;
	rl->qcrl = getQLink(0);
	APPEND_LIST3(reqLink3, rl);
	UNLOCK_MUT_(mutmpc);

	return rl;
}
void MPClient::putRLink(ReqLink *rl)
{
	LOCK_MUT_(mutmpc);
	putQLink(rl->qcrl, 0);
	APPEND_LIST2(reqLink2, rl);
	if(cntwreq) {
		cntwreq--;
		SIG_LSEM_(sigmpc);
	}
	UNLOCK_MUT_(mutmpc);
}
void MPClient::mpcRequest(ReqLink *rl)
{
	LOCK_MUT_(mutmpc2);
	APPEND_LIST(reqLink, rl);
	if(wntmpc) SIG_LSEM_(sigmpc2);//요청 대기하고 있는 쓰레드가 있으면 시그널
	UNLOCK_MUT_(mutmpc2);
}
void MPClient::mpcActor(void)
{
	ReqLink *rl;

	while(1) {
		try {
			LOCK_MUT_(mutmpc2);
			while(1) {
				if(reqLink) {
					GET_LIST(reqLink, rl);
					rntmpc++;
					UNLOCK_MUT_(mutmpc2);
					if(rl->reqKernel == nullx) return;//mp client 전채 종료
					rl->executeReq();
					LOCK_MUT_(mutmpc2);
					rntmpc--;
				} else {
					wntmpc++;
					UNLOCK_MUT_(mutmpc2);
					WAIT_LSEM_(sigmpc2);
					LOCK_MUT_(mutmpc2);
					wntmpc--;
					continue;
				}
			}
		} catch(ExptObject eo) {
			printf("actor error\n%s", eo.msg);
			LOCK_MUT_(mutmpc2);
			rntmpc--;
			UNLOCK_MUT_(mutmpc2);
		}
	}
}
thrret ThrActor(thrarg arg)
{
	MPClient *mpc = (MPClient *)arg;

	mpc->mpcActor();

	return 0;
}
void MPClient::bootActor(intx cnt_act, intx lim_req)
{
	limreq = lim_req;

	for(;cntActor < cnt_act; cntActor++) {
		xthread_create((void *)ThrActor, this);
	}
}

template<typename dat_t>
SeqItem *read_array_batch(QueryContext *qc, dat_t *pnext, dat_t *pend, intx n_seq, intx dimen, intx nrest)
{
	intx iseq = 0, idim = 0, rv = 1, irest = nrest;
	SeqItem *batch_head = nullx, *seq_item;
	DimenItem *dim_item;
	intx szdim = dimen * sizeof(dat_t);//feature데이터 사이즈
	bytex *pdim, zero_pad = 0;

	SelAlloc(qc, sizeof(SeqItem), *(bytex **)&seq_item);//시퀀스 적재 구조 할당.
	seq_item->vseq = nullx;
	SelAlloc(qc, szdim, pdim);
	while(pnext < pend) {
		if(zero_pad) *((dat_t *)pdim + idim) = 0;//제로 패딩
		else *((dat_t *)pdim + idim) = *pnext;//원소 데이터 적재.
		if(++idim == dimen) {//feature(디멘젼)은 시퀀스의 한개 엘레먼트에 해당하고 피쳐 사이즈 만큼 데이터가 모두 적재뙛으면 
			idim = 0;
			SelAlloc(qc, sizeof(DimenItem), *(bytex **)&dim_item);
			dim_item->vdimen = pdim;
			//printf("[%d]", *(intx *)dim_item->vdimen);
			APPEND_LIST(seq_item->vseq, dim_item);//시퀀스의 한개 피쳐 엘레먼트 리스팅
			SelAlloc(qc, szdim, pdim);//다음 피쳐 슬롯 할당 준비
			if(++iseq >= n_seq) {//한개 시퀀스의 엘레먼트 피쳐들이 모두 적재됐으면
				if(nrest) {//시퀀스 길이가 스트라이드 경계에 맞지않고 제로 패딩 설정이면 
					if(nrest == irest--) {//제로패딩 첫번째이면
						zero_pad = 1;//제로패딩 플래그 설정
						continue;//시퀀스 끝 제로패딩 중이므로 배치 리스팅 하지 않고 계속 제로 패딩 시퀀스 구성
					} else if(irest < 0) {//매 시퀀스에서 제로패딩이 모두 체워졌으면 완로된 시퀀스 배치 리스팅
						irest = nrest;//초기 제로패딩 갯수 설정
						zero_pad = 0;//제로패딩 플래그 리셋
					} else continue;
				}
				//printf(" ]111\n");
				APPEND_LIST(batch_head, seq_item);//이번 시퀀스를 배치에 리스팅
				SelAlloc(qc, sizeof(SeqItem), *(bytex **)&seq_item);
				seq_item->vseq = nullx;
				iseq = 0;
			}
		}
		if(zero_pad == 0) pnext++;
	}
	return batch_head;
}
StrideBind *make_array_batch(QueryContext *qc, DimenItem *kernel, intx width, intx exc_cnt, intx &nseq) //스트라이드 단위내 시퀀스 생성
{
	StrideBind *str_bind;
	SeqItem *seq_list = nullx, *seq_item;
	DimenItem **stack, *ptop, *end, *dim_item, *head;
	intx sz_st = sizeof(DimenItem *) * width, i = 0, top = 0, *istack, idx, contiguous, contig_cnt = 0;

	stack = (DimenItem **)malloc(sz_st);
	istack = (intx *)malloc(width * sizeof(intx));//on_position
	//printf("222[ ");
	for(end = kernel;end && top < width; end = end->ptrRight, top++) {
		*(stack + top) = end;//시작부터 커널사이즈까지 스택 적재
		*(istack + top) = top;//on_position
		//printf("[%d](%d)", *(intx *)end->vdimen, top);
	}
	//printf(" ]\n");
	for(nseq = 0;top;) {
		if(width > exc_cnt) {
			for(i = contig_cnt = 0, contiguous = *istack;i < top; i++) {
				if(contiguous == *(istack + i)) {
					contiguous++;
					if(++contig_cnt >= exc_cnt) break;
				} else {
					contiguous = *(istack + i);
					contiguous++;
					contig_cnt = 1;//두개의 숫자가 연속하면 2가 되게
				}
			}
		}
		if(contig_cnt < exc_cnt) {//연속 시퀀스가 일정 갯수 이하여만 시퀀스 리스팅
			SelAlloc(qc, sizeof(SeqItem), *(bytex **)&seq_item);
			//printf("333[ ");
			for(i = 0, head = nullx;i < top; i++) {//스택의 바닥부터 탑까지 피쳐 엘레먼트로 파생 시퀀스로 구성
				SelAlloc(qc, sizeof(DimenItem), *(bytex **)&dim_item);
				dim_item->vdimen = (*(stack + i))->vdimen;
				dim_item->idxKernel = *(istack + i);//on_position
				APPEND_LIST(head, dim_item);
				//printf("[%d](%d)", *(intx *)dim_item->vdimen, dim_item->idxKernel);
			}
			//printf(" ]\n");
			seq_item->vseq = head;
			APPEND_LIST(seq_list, seq_item);//시퀀스 리스팅
			nseq++;
		}
		ptop = *(stack + --top);//스택의 탑을 제거
		idx = *(istack + top);//on_position
		for(ptop = ptop->ptrRight, idx++;ptop != end; ptop = ptop->ptrRight, top++, idx++) {
			*(stack + top) = ptop;//제거된 탑의 다음부터 커널사이즈 끝까지 적재
			*(istack + top) = idx;//on_position
		}
	}
	free(stack);
	free(istack);

	if(seq_list) {
		SelAlloc(qc, sizeof(StrideBind), *(bytex **)&str_bind);
		str_bind->seqList = seq_list;
		return str_bind;
	} else return nullx;
}
#define ZERO_PAD	(dat_t)0 //(dat_t)1e-9
template<typename dat_t>
void *make_rs_array(intx dt, intx nbatch, intx nbind, intx nreduce, intx width, intx nfeat, BindItem *bitem,
	void *tcr, array_generfp agen, array_datafp adat, sytet zero_pading, bool one_batch)
{
	SeqItem *sitem;
	StrideBind *stritem;
	DimenItem *ditem;
	intx ndim, nrest = 0, nzero = 0;
	intx dims[4];
	intx l, k, i, j, a, b, c, off;

	if(zero_pading == 2) {//width와 stride로 나위어진 시퀀스중 마지막 시퀀스가 width보다 적을 경우 남는 시퀀스로 
		if(nbind % nreduce) {//파생된 시퀀스 이후 모자르는 시퀀스 공간을 제로값으로 채운다.
			nrest = (nbind / nreduce + 1) * nreduce - nbind;//모자르는 width단위 길이 파생(reduced)서브 시퀀스 갯수
			nzero = nrest * width * nfeat;//모자르는 서브 시퀀스의 zero값으로 채울 원소 갯수
		}
	}
	if(nbatch > 1 || one_batch) {//배치가 2개 이상이면 4차원 행렬 생성 리턴
		ndim = 4;
		dims[0] = nbatch;
		dims[1] = nbind + nrest;
		dims[2] = width;
		dims[3] = nfeat;
	} else {//nbatch == 1
		ndim = 3;//배치가 1개이면 3차원 행렬 생성 리턴
		dims[0] = nbind + nrest;
		dims[1] = width;
		dims[2] = nfeat;
	}
	void *result;
	if(tcr) {
		result = agen(tcr, ndim, dims, dt);
		if(result == NULL) throwFault(-1, "make rs array new array fail\n");
	} else result = agen;//tcr이 널로 주어지면 agen은 플럭스로 직접 주어저셔 생성 없이 주어진 플럭스에 쓰기한다.
	dat_t* resultDataPtr = (dat_t *)adat(result);

	for(l = 0; l < nbatch; l++, bitem = bitem->ptrRight) {
		a = l * (nbind + nrest) * width * nfeat;
		for(k = 0, stritem = bitem->strideList;k < nbind; stritem = stritem->ptrRight) {//한개 소스 시퀀스로부터 조합되는
			for(sitem = stritem->seqList;sitem; k++, sitem = sitem->ptrRight) {//파생 시퀀스 리스트
				b = k * width * nfeat;
				//printf("444[ ");
				for(i = 0, ditem = sitem->vseq; i < width; i++) {//파생 시퀀스의 시퀀스 피쳐 엘레먼트 리스트
					c = i * nfeat;
					if(ditem && i == ditem->idxKernel) {//non_position,if(ditem) {//파생 시퀀스의 피쳐 에레먼트가 있으면 한개 피쳐의 디멘젼들을 적재
						for(j = 0; j < nfeat; j++) {
							off = a + b + c + j;
							resultDataPtr[off] = *((dat_t *)ditem->vdimen + j);
							//printf("[%d](%d)", *(intx *)ditem->vdimen, i);
						}
						ditem = ditem->ptrRight;
					} else {//건너뛴 엘레먼트면//파생 시퀀스의 피쳐 에레먼트가 시퀀스 갯수보다 적어서 이후 없으면 0값을 적재
						for(j = 0; j < nfeat; j++) {
							off = a + b + c + j;
							resultDataPtr[off] = ZERO_PAD;
							//printf("[%d](%d)", 0, i);
						}
					}
				}
				//printf(" ]\n");
			}
		}
		if(nzero) {//모자르는 서브 시퀀스를 제로값으로 채운다.
			for(intt n = ++off + nzero;off < n; off++) resultDataPtr[off] = ZERO_PAD;
		}
	}
	return result;
}
template<typename dat_t>
void *_make_source_array(intx dt, SeqItem *batch_data, intx nbatch, intx width, intx nfeat, intx stride, intx nstride, void *tcr, array_generfp agen, array_datafp adat)
{
	SeqItem *sitem;
	DimenItem *ditem, *ditem2;
	intx s, istride, ndim, zero_pading = 0;//나중에 매개변수로 받게 수정
	intt dims[4];
	intx l, i, j, a, b, c, nbind = nstride + 1;

	ndim = 4;
	dims[0] = nbatch;
	dims[1] = nbind;
	dims[2] = width;
	dims[3] = nfeat;

	void *result;
	if(tcr) {
		result = agen(tcr, ndim, dims, dt);
		if(result == NULL) throwFault(-1, "make source array new array fail\n");
	} else result = agen;//tcr이 널로 주어지면 agen은 플럭스로 직접 주어저셔 생성 없이 주어진 플럭스에 쓰기한다.
	dat_t* resultDataPtr = (dat_t *)adat(result);

	for(l = 0, sitem = batch_data;sitem; sitem = sitem->ptrRight, l++) {
		ditem = sitem->vseq;
		a = l * nbind * width * nfeat;
		for(s = stride, istride = 0;istride <= nstride; ditem = ditem->ptrRight, s++) {//한개 소스 시퀀스의 피쳐 엘레먼트 리스트
			if(s == stride) {//처음부터 수행, 마지막에는 한개 피쳐 엘레먼트만 있어도 수행됨.
				s = 0;
				istride++;
			} else continue;
			b = (istride - 1) * width * nfeat;
			//printf("555[ ");
			for(i = 0, ditem2 = ditem; i < width; i++) {//파생 시퀀스의 시퀀스 피쳐 엘레먼트 리스트
				c = i * nfeat;
				if(ditem2) {//파생 시퀀스의 피쳐 에레먼트가 있으면 한개 피쳐의 디멘젼들을 적재
					for(j = 0; j < nfeat; j++) {
						resultDataPtr[a + b + c + j] = *((dat_t *)ditem2->vdimen + j);
						//printf("[%d](%d)", *(intx *)ditem2->vdimen, i);
					}
					ditem2 = ditem2->ptrRight;
				} else {//시퀀스 갯수보다 적어 이후 없으면 0값을 적재
					for(j = 0; j < nfeat; j++) {
						resultDataPtr[a + b + c + j] = ZERO_PAD;
						//printf("[%d](%d)", 0, i);
					}
				}
			}
			//printf(" ]\n");
		}
	}
	return result;
}
void *make_source_array(intx dt, SeqItem *batch_data, intx nbatch, intx width, intx nfeat, intx stride, intx nstride, void *tcr, array_generfp agen, array_datafp adat)
{
	switch(dt) {
	case BYTEX_TP:
		return _make_source_array<bytex>(dt, batch_data, nbatch, width, nfeat, stride, nstride, tcr, agen, adat);
	case SHORTX_TP:
		return _make_source_array<shortx>(dt, batch_data, nbatch, width, nfeat, stride, nstride, tcr, agen, adat);
	case FLOATX_TP:
		return _make_source_array<floatx>(dt, batch_data, nbatch, width, nfeat, stride, nstride, tcr, agen, adat);
	case INTX_TP:
		return _make_source_array<intx>(dt, batch_data, nbatch, width, nfeat, stride, nstride, tcr, agen, adat);
	case LONGX_TP:
		return _make_source_array<longx>(dt, batch_data, nbatch, width, nfeat, stride, nstride, tcr, agen, adat);
	case DOUBLEX_TP:
		return _make_source_array<doublex>(dt, batch_data, nbatch, width, nfeat, stride, nstride, tcr, agen, adat);
	default:
		return nullx;
	}
}
void *transform(void *b, void *e, intx dt, intx ndims, intx dims[], intx width, intx stride, doublex exc_contig_r,
	sytex zero_pading, sytex src_make, void *tcr, array_generfp agen, array_datafp adat, void *&src_array, bool one_batch)
{

	intx nbatch = (intx)dims[0], nseq = (intx)dims[1], dimen = 1, i;

	for(i = 2;i < ndims; i++) dimen *= (intx)dims[i];//3번째 이하 차수는 모두 디멘젼(feature)으로 처리

	QCLink *qcl = rsc::mpClient->getQLink(1);//python array read 할당용 객체 획득
	QueryContext *qc = &qcl->qText;
	SeqItem *batch_data, *sitem;
	intx nrest, nstride;

	if(stride > width) {
		nstride = nseq / stride;
		if(zero_pading && nseq % stride < width) nrest = width - nseq % stride;
		else nrest = 0;
	} else {
		if(zero_pading == 1 || zero_pading == 2) {//시퀀스 길이가 스트라이드 단위로 남는 것이 있으면 시퀀스 길이를 오버하는 것을 0로 패딩
			//zero_pading이 1인 경우 예로 시퀀스가 8, 스트라이드 간격이 6이면 디음 스트라이드할때 남은 사이즈가 2이므로
			//스트라이드 간격에 못미치므로 남은 2두개만 전개된다.(이때 nrest는 움수가 설정됨) 스트라이드가 4이면 다음 스트라이드할때
			//남은 사이즈가 4이므로 스트라이드 간격에 맞아 스트라이드가 전개되고 사이즈가 12[4 + 8(width)]가 되야하는데 시퀀스
			//사이즈는 8이므로 모자르는 사이즈 4개가 nrest로 설정되어 read_array_batch에서 제로패딩되어 전개된다. zero_pading이 
			//2케이스이면 전자의 경우 모자르는 사이즈 6의 전개분 만큼 make_rs_array에서 제로패딩되고 후자의 경우는 제로패딩되어 
			//전개됐으므로 더이상의 처리는 없고 결과는 양측이 동일하게 된다. nrest가 0인 경우는 예로 시퀀스가 32이고 스트라이드가
			//6일 경우 스트라이드 4번째(옵셋 24)에서 커널 길이 8을 더하면 32가 되어 시퀀스 길이에 스트라이드에 정합되므로 더 이상 
			//스트라이드를 진행하지 않고 따라서 제로 패딩 필요없게 된다.
			nstride = nseq / stride;
			nrest = stride * (nstride - 1) + width - nseq;//마지막 스트라이드의 시작점이 마지막 커널 시작점을 초과하는 만큼 nrest로하여 제로값을 패딩해준다.
			if(nseq % stride == 0 || nrest == 0) nstride--;//ㄱ)에서 <= 로 비교하므로 나머지가 없은 경우만 감소시킨다.
		} else if(zero_pading == 3) {//inner zero padding
			intt inner_seq = nseq - width;
			nstride = inner_seq / stride;
			if(inner_seq % stride) {
				nstride++;
				nrest = nstride * stride + width - inner_seq;
			} else nrest = 0;
		} else if(zero_pading == 4) {//outer zero padding
			nstride = nseq / stride;
			if(nseq % stride == 0) nstride--;
			nrest = nstride * stride;
			nrest = nrest + width - nseq;
		} else {
			nstride = nseq / stride - width / stride;
			intx end = nstride * stride + width;
			if(end == nseq) {//nseq == 64, stride == 2, width == 8, nstride == 28, end == 64, 소스 시퀀스의 
				nrest = 0;		//마지막 스트라이드 에서 width를 더한 길이가 소스시퀀스 길이 nseq와 정합되게 nstride가 설정된다.
			} else {//소스 시퀀스 별로 마지막 스트라이드에서 width를 더한 길이가 소스시퀀스 길이 nseq보다 크게 되도록 nstride를 설정한다.
				if(end < nseq) nstride++;//nseq == 64, stride == 5, width == 8, nstride == 11, end == 63, nstride == 12, end == 68
				//else //nseq == 64, stride == 3, width == 8, nstride == 19, end == 65
				if(zero_pading) nrest = width - (nseq - nstride * stride);//zero_pading == 2, 위에서 nseq는 64이므로 end에 모자른 길이를 0 패딩
				else nrest = 0;//마지막 스트라이드에서 width를 더한 end보다 nseq가 모자르는 길이 만큼 패딩
			}
		}
	}
	//if(zero_pading && stride <= width) {
	//	intx end = nseq - width;
	//	nrest = (end % stride ? width - (end % stride) : 0);
	//} else nrest = 0;//stride가 width보가 큰경우는 제로패딩 경우없이 스트라이드 단위에서 남는 짜투리는 밑에서 무시된다.

	switch(dt) {
	case BYTEX_TP:
		batch_data = read_array_batch<bytex>(qc, (bytex *)b, (bytex *)e, nseq, dimen, nrest);
		break;
	case SHORTX_TP:
		batch_data = read_array_batch<shortx>(qc, (shortx *)b, (shortx *)e, nseq, dimen, nrest);
		break;
	case FLOATX_TP:
		batch_data = read_array_batch<floatx>(qc, (floatx *)b, (floatx *)e, nseq, dimen, nrest);
		break;
	case INTX_TP:
		batch_data = read_array_batch<intx>(qc, (intx *)b, (intx *)e, nseq, dimen, nrest);
		break;
	case LONGX_TP:
		batch_data = read_array_batch<longx>(qc, (longx *)b, (longx *)e, nseq, dimen, nrest);
		break;
	case DOUBLEX_TP:
		batch_data = read_array_batch<doublex>(qc, (doublex *)b, (doublex *)e, nseq, dimen, nrest);
		break;
	default:
		return nullx;
	}
	DimenItem *ditem;
	SemLink *sl = rsc::mpClient->getSLink();
	ReqLink *rl, *head_rl = nullx, *next_rl;
	BindItem *bitem = nullx, *rbatch = nullx;
	void *rarray;
	intx istride;

	sl->reqWidth = width;//커널 사이즈 설정.
	sl->nExcContig = (intx)(width * exc_contig_r);
	//소스 시퀀스 한개 단위로 스트라이드 간격으로 스트라이드하며 커널 사이즈내에서 파생 조합 시퀀스 생성 요청
	for(sitem = batch_data, nbatch = 0;sitem; sitem = sitem->ptrRight, nbatch++) {
		ditem = sitem->vseq;
		SelAlloc(qc, sizeof(BindItem), *(bytex **)&bitem);//소스 시퀀스당 파생되는 시퀀스 묶음 할당
		memset(bitem, 0x00, sizeof(BindItem));
		APPEND_LIST(rbatch, bitem);//한개 소스 시퀀스에서 파생되는 모든 파생 조합 시퀀스를 묶음 단위로 결과 배치 리스팅 됨
		for(i = stride, istride = 0;istride <= nstride; ditem = ditem->ptrRight, i++) {//ㄱ.한개 소스 시퀀스의 피쳐 엘레먼트 리스트
			if(i == stride) {//처음부터 수행, 마지막에는 한개 피쳐 엘레먼트만 있어도 수행됨.
				i = 0;
				istride++;
				rl = rsc::mpClient->getRLink();
				APPEND_LIST2(head_rl, rl);//반납하기위해 리스팅
				rl->requestActor(sl, ditem, bitem);//ditem는 파생 조합 시작 피쳐 엘레먼트
			}
		}
	}
	if(src_make > 0) src_array = make_source_array(dt, batch_data, nbatch, width, dimen, stride, nstride, tcr, agen, adat);
	sl->waitSignal();
	if(src_make < 0) src_array = (void *)bitem->nmaxReduce;
	else if(src_make == 0) src_array = nullx;

	switch(dt) {//마지막 배치의 bitem->numBind으로 호출한다. width가 모든 배치에서 같은 길이이므로 
	case BYTEX_TP:					//모든 파생 조합 배치의 numBind도 동일하다.
		rarray = make_rs_array<bytex>(dt, nbatch, bitem->numBind, bitem->nmaxReduce, width, dimen, rbatch, tcr, agen, adat, zero_pading, one_batch);
		break;
	case SHORTX_TP:
		rarray = make_rs_array<shortx>(dt, nbatch, bitem->numBind, bitem->nmaxReduce, width, dimen, rbatch, tcr, agen, adat, zero_pading, one_batch);
		break;
	case FLOATX_TP:
		rarray = make_rs_array<floatx>(dt, nbatch, bitem->numBind, bitem->nmaxReduce, width, dimen, rbatch, tcr, agen, adat, zero_pading, one_batch);
		break;
	case INTX_TP:
		rarray = make_rs_array<intx>(dt, nbatch, bitem->numBind, bitem->nmaxReduce, width, dimen, rbatch, tcr, agen, adat, zero_pading, one_batch);
		break;
	case LONGX_TP:
		rarray = make_rs_array<longx>(dt, nbatch, bitem->numBind, bitem->nmaxReduce, width, dimen, rbatch, tcr, agen, adat, zero_pading, one_batch);
		break;
	case DOUBLEX_TP:
		rarray = make_rs_array<doublex>(dt, nbatch, bitem->numBind, bitem->nmaxReduce, width, dimen, rbatch, tcr, agen, adat, zero_pading, one_batch);
		break;
	default:
		return nullx;
	}
	for(rl = head_rl;rl; rl = next_rl) {
		next_rl = rl->ptrRight2;
		rsc::mpClient->putRLink(rl);
	}
	rsc::mpClient->putSLink(sl);
	rsc::mpClient->putQLink(qcl, 1);

	return rarray;
}
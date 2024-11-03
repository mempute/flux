
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
	if(waitOn && reqCnt == prsCnt) SIG_LSEM_(semSig);//���� �н� ���� ��û�￡�� �������̰� ��û �Ǽ��� ��� ó�������� �ñ׳�
	UNLOCK_MUT_(mutSig);
}
void SemLink::waitSignal(void)
{
	LOCK_MUT_(mutSig);
	waitOn = 1;
	if(prsCnt < reqCnt) {//��û �Ǽ��� ��� ó���Ҷ����� ���
		UNLOCK_MUT_(mutSig);
		WAIT_LSEM_(semSig);
	} else UNLOCK_MUT_(mutSig);//��û �Ǽ��� ��� ó�������� ������.
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
	for(str_bind = head->ptrLeft;str_bind != head; str_bind = str_bind->ptrLeft) {//���� üũ ������
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
	rl.ptrLeft = rl.ptrRight = &rl;//��� actor �����尡 ��� �ݺ������� �� rl�� get list�ϰ� �����Ѵ�.
	rl.reqKernel = nullx;//actor ������ ���� �޼���
	mpcRequest(&rl);//��� actor ������ ����

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
	APPEND_LIST3(semLink3, sl);//����� ���� ����Ʈ
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
	if(cntreq == limreq) {//transform��û�� ���ķ� ����ɼ��ְ� limreq�� �ּ� �� ������ ��û�� ���� Ŀ���Ѵ�.
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
	if(wntmpc) SIG_LSEM_(sigmpc2);//��û ����ϰ� �ִ� �����尡 ������ �ñ׳�
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
					if(rl->reqKernel == nullx) return;//mp client ��ä ����
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
	intx szdim = dimen * sizeof(dat_t);//feature������ ������
	bytex *pdim, zero_pad = 0;

	SelAlloc(qc, sizeof(SeqItem), *(bytex **)&seq_item);//������ ���� ���� �Ҵ�.
	seq_item->vseq = nullx;
	SelAlloc(qc, szdim, pdim);
	while(pnext < pend) {
		if(zero_pad) *((dat_t *)pdim + idim) = 0;//���� �е�
		else *((dat_t *)pdim + idim) = *pnext;//���� ������ ����.
		if(++idim == dimen) {//feature(�����)�� �������� �Ѱ� ������Ʈ�� �ش��ϰ� ���� ������ ��ŭ �����Ͱ� ��� ���猠���� 
			idim = 0;
			SelAlloc(qc, sizeof(DimenItem), *(bytex **)&dim_item);
			dim_item->vdimen = pdim;
			//printf("[%d]", *(intx *)dim_item->vdimen);
			APPEND_LIST(seq_item->vseq, dim_item);//�������� �Ѱ� ���� ������Ʈ ������
			SelAlloc(qc, szdim, pdim);//���� ���� ���� �Ҵ� �غ�
			if(++iseq >= n_seq) {//�Ѱ� �������� ������Ʈ ���ĵ��� ��� ���������
				if(nrest) {//������ ���̰� ��Ʈ���̵� ��迡 �����ʰ� ���� �е� �����̸� 
					if(nrest == irest--) {//�����е� ù��°�̸�
						zero_pad = 1;//�����е� �÷��� ����
						continue;//������ �� �����е� ���̹Ƿ� ��ġ ������ ���� �ʰ� ��� ���� �е� ������ ����
					} else if(irest < 0) {//�� ���������� �����е��� ��� ü�������� �Ϸε� ������ ��ġ ������
						irest = nrest;//�ʱ� �����е� ���� ����
						zero_pad = 0;//�����е� �÷��� ����
					} else continue;
				}
				//printf(" ]111\n");
				APPEND_LIST(batch_head, seq_item);//�̹� �������� ��ġ�� ������
				SelAlloc(qc, sizeof(SeqItem), *(bytex **)&seq_item);
				seq_item->vseq = nullx;
				iseq = 0;
			}
		}
		if(zero_pad == 0) pnext++;
	}
	return batch_head;
}
StrideBind *make_array_batch(QueryContext *qc, DimenItem *kernel, intx width, intx exc_cnt, intx &nseq) //��Ʈ���̵� ������ ������ ����
{
	StrideBind *str_bind;
	SeqItem *seq_list = nullx, *seq_item;
	DimenItem **stack, *ptop, *end, *dim_item, *head;
	intx sz_st = sizeof(DimenItem *) * width, i = 0, top = 0, *istack, idx, contiguous, contig_cnt = 0;

	stack = (DimenItem **)malloc(sz_st);
	istack = (intx *)malloc(width * sizeof(intx));//on_position
	//printf("222[ ");
	for(end = kernel;end && top < width; end = end->ptrRight, top++) {
		*(stack + top) = end;//���ۺ��� Ŀ�λ�������� ���� ����
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
					contig_cnt = 1;//�ΰ��� ���ڰ� �����ϸ� 2�� �ǰ�
				}
			}
		}
		if(contig_cnt < exc_cnt) {//���� �������� ���� ���� ���Ͽ��� ������ ������
			SelAlloc(qc, sizeof(SeqItem), *(bytex **)&seq_item);
			//printf("333[ ");
			for(i = 0, head = nullx;i < top; i++) {//������ �ٴں��� ž���� ���� ������Ʈ�� �Ļ� �������� ����
				SelAlloc(qc, sizeof(DimenItem), *(bytex **)&dim_item);
				dim_item->vdimen = (*(stack + i))->vdimen;
				dim_item->idxKernel = *(istack + i);//on_position
				APPEND_LIST(head, dim_item);
				//printf("[%d](%d)", *(intx *)dim_item->vdimen, dim_item->idxKernel);
			}
			//printf(" ]\n");
			seq_item->vseq = head;
			APPEND_LIST(seq_list, seq_item);//������ ������
			nseq++;
		}
		ptop = *(stack + --top);//������ ž�� ����
		idx = *(istack + top);//on_position
		for(ptop = ptop->ptrRight, idx++;ptop != end; ptop = ptop->ptrRight, top++, idx++) {
			*(stack + top) = ptop;//���ŵ� ž�� �������� Ŀ�λ����� ������ ����
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

	if(zero_pading == 2) {//width�� stride�� �������� �������� ������ �������� width���� ���� ��� ���� �������� 
		if(nbind % nreduce) {//�Ļ��� ������ ���� ���ڸ��� ������ ������ ���ΰ����� ä���.
			nrest = (nbind / nreduce + 1) * nreduce - nbind;//���ڸ��� width���� ���� �Ļ�(reduced)���� ������ ����
			nzero = nrest * width * nfeat;//���ڸ��� ���� �������� zero������ ä�� ���� ����
		}
	}
	if(nbatch > 1 || one_batch) {//��ġ�� 2�� �̻��̸� 4���� ��� ���� ����
		ndim = 4;
		dims[0] = nbatch;
		dims[1] = nbind + nrest;
		dims[2] = width;
		dims[3] = nfeat;
	} else {//nbatch == 1
		ndim = 3;//��ġ�� 1���̸� 3���� ��� ���� ����
		dims[0] = nbind + nrest;
		dims[1] = width;
		dims[2] = nfeat;
	}
	void *result;
	if(tcr) {
		result = agen(tcr, ndim, dims, dt);
		if(result == NULL) throwFault(-1, "make rs array new array fail\n");
	} else result = agen;//tcr�� �η� �־����� agen�� �÷����� ���� �־����� ���� ���� �־��� �÷����� �����Ѵ�.
	dat_t* resultDataPtr = (dat_t *)adat(result);

	for(l = 0; l < nbatch; l++, bitem = bitem->ptrRight) {
		a = l * (nbind + nrest) * width * nfeat;
		for(k = 0, stritem = bitem->strideList;k < nbind; stritem = stritem->ptrRight) {//�Ѱ� �ҽ� �������κ��� ���յǴ�
			for(sitem = stritem->seqList;sitem; k++, sitem = sitem->ptrRight) {//�Ļ� ������ ����Ʈ
				b = k * width * nfeat;
				//printf("444[ ");
				for(i = 0, ditem = sitem->vseq; i < width; i++) {//�Ļ� �������� ������ ���� ������Ʈ ����Ʈ
					c = i * nfeat;
					if(ditem && i == ditem->idxKernel) {//non_position,if(ditem) {//�Ļ� �������� ���� ������Ʈ�� ������ �Ѱ� ������ ��������� ����
						for(j = 0; j < nfeat; j++) {
							off = a + b + c + j;
							resultDataPtr[off] = *((dat_t *)ditem->vdimen + j);
							//printf("[%d](%d)", *(intx *)ditem->vdimen, i);
						}
						ditem = ditem->ptrRight;
					} else {//�ǳʶ� ������Ʈ��//�Ļ� �������� ���� ������Ʈ�� ������ �������� ��� ���� ������ 0���� ����
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
		if(nzero) {//���ڸ��� ���� �������� ���ΰ����� ä���.
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
	intx s, istride, ndim, zero_pading = 0;//���߿� �Ű������� �ް� ����
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
	} else result = agen;//tcr�� �η� �־����� agen�� �÷����� ���� �־����� ���� ���� �־��� �÷����� �����Ѵ�.
	dat_t* resultDataPtr = (dat_t *)adat(result);

	for(l = 0, sitem = batch_data;sitem; sitem = sitem->ptrRight, l++) {
		ditem = sitem->vseq;
		a = l * nbind * width * nfeat;
		for(s = stride, istride = 0;istride <= nstride; ditem = ditem->ptrRight, s++) {//�Ѱ� �ҽ� �������� ���� ������Ʈ ����Ʈ
			if(s == stride) {//ó������ ����, ���������� �Ѱ� ���� ������Ʈ�� �־ �����.
				s = 0;
				istride++;
			} else continue;
			b = (istride - 1) * width * nfeat;
			//printf("555[ ");
			for(i = 0, ditem2 = ditem; i < width; i++) {//�Ļ� �������� ������ ���� ������Ʈ ����Ʈ
				c = i * nfeat;
				if(ditem2) {//�Ļ� �������� ���� ������Ʈ�� ������ �Ѱ� ������ ��������� ����
					for(j = 0; j < nfeat; j++) {
						resultDataPtr[a + b + c + j] = *((dat_t *)ditem2->vdimen + j);
						//printf("[%d](%d)", *(intx *)ditem2->vdimen, i);
					}
					ditem2 = ditem2->ptrRight;
				} else {//������ �������� ���� ���� ������ 0���� ����
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

	for(i = 2;i < ndims; i++) dimen *= (intx)dims[i];//3��° ���� ������ ��� �����(feature)���� ó��

	QCLink *qcl = rsc::mpClient->getQLink(1);//python array read �Ҵ�� ��ü ȹ��
	QueryContext *qc = &qcl->qText;
	SeqItem *batch_data, *sitem;
	intx nrest, nstride;

	if(stride > width) {
		nstride = nseq / stride;
		if(zero_pading && nseq % stride < width) nrest = width - nseq % stride;
		else nrest = 0;
	} else {
		if(zero_pading == 1 || zero_pading == 2) {//������ ���̰� ��Ʈ���̵� ������ ���� ���� ������ ������ ���̸� �����ϴ� ���� 0�� �е�
			//zero_pading�� 1�� ��� ���� �������� 8, ��Ʈ���̵� ������ 6�̸� ���� ��Ʈ���̵��Ҷ� ���� ����� 2�̹Ƿ�
			//��Ʈ���̵� ���ݿ� ����ġ�Ƿ� ���� 2�ΰ��� �����ȴ�.(�̶� nrest�� ����� ������) ��Ʈ���̵尡 4�̸� ���� ��Ʈ���̵��Ҷ�
			//���� ����� 4�̹Ƿ� ��Ʈ���̵� ���ݿ� �¾� ��Ʈ���̵尡 �����ǰ� ����� 12[4 + 8(width)]�� �Ǿ��ϴµ� ������
			//������� 8�̹Ƿ� ���ڸ��� ������ 4���� nrest�� �����Ǿ� read_array_batch���� �����е��Ǿ� �����ȴ�. zero_pading�� 
			//2���̽��̸� ������ ��� ���ڸ��� ������ 6�� ������ ��ŭ make_rs_array���� �����е��ǰ� ������ ���� �����е��Ǿ� 
			//���������Ƿ� ���̻��� ó���� ���� ����� ������ �����ϰ� �ȴ�. nrest�� 0�� ���� ���� �������� 32�̰� ��Ʈ���̵尡
			//6�� ��� ��Ʈ���̵� 4��°(�ɼ� 24)���� Ŀ�� ���� 8�� ���ϸ� 32�� �Ǿ� ������ ���̿� ��Ʈ���̵忡 ���յǹǷ� �� �̻� 
			//��Ʈ���̵带 �������� �ʰ� ���� ���� �е� �ʿ���� �ȴ�.
			nstride = nseq / stride;
			nrest = stride * (nstride - 1) + width - nseq;//������ ��Ʈ���̵��� �������� ������ Ŀ�� �������� �ʰ��ϴ� ��ŭ nrest���Ͽ� ���ΰ��� �е����ش�.
			if(nseq % stride == 0 || nrest == 0) nstride--;//��)���� <= �� ���ϹǷ� �������� ���� ��츸 ���ҽ�Ų��.
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
			if(end == nseq) {//nseq == 64, stride == 2, width == 8, nstride == 28, end == 64, �ҽ� �������� 
				nrest = 0;		//������ ��Ʈ���̵� ���� width�� ���� ���̰� �ҽ������� ���� nseq�� ���յǰ� nstride�� �����ȴ�.
			} else {//�ҽ� ������ ���� ������ ��Ʈ���̵忡�� width�� ���� ���̰� �ҽ������� ���� nseq���� ũ�� �ǵ��� nstride�� �����Ѵ�.
				if(end < nseq) nstride++;//nseq == 64, stride == 5, width == 8, nstride == 11, end == 63, nstride == 12, end == 68
				//else //nseq == 64, stride == 3, width == 8, nstride == 19, end == 65
				if(zero_pading) nrest = width - (nseq - nstride * stride);//zero_pading == 2, ������ nseq�� 64�̹Ƿ� end�� ���ڸ� ���̸� 0 �е�
				else nrest = 0;//������ ��Ʈ���̵忡�� width�� ���� end���� nseq�� ���ڸ��� ���� ��ŭ �е�
			}
		}
	}
	//if(zero_pading && stride <= width) {
	//	intx end = nseq - width;
	//	nrest = (end % stride ? width - (end % stride) : 0);
	//} else nrest = 0;//stride�� width���� ū���� �����е� ������ ��Ʈ���̵� �������� ���� ¥������ �ؿ��� ���õȴ�.

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

	sl->reqWidth = width;//Ŀ�� ������ ����.
	sl->nExcContig = (intx)(width * exc_contig_r);
	//�ҽ� ������ �Ѱ� ������ ��Ʈ���̵� �������� ��Ʈ���̵��ϸ� Ŀ�� ��������� �Ļ� ���� ������ ���� ��û
	for(sitem = batch_data, nbatch = 0;sitem; sitem = sitem->ptrRight, nbatch++) {
		ditem = sitem->vseq;
		SelAlloc(qc, sizeof(BindItem), *(bytex **)&bitem);//�ҽ� �������� �Ļ��Ǵ� ������ ���� �Ҵ�
		memset(bitem, 0x00, sizeof(BindItem));
		APPEND_LIST(rbatch, bitem);//�Ѱ� �ҽ� ���������� �Ļ��Ǵ� ��� �Ļ� ���� �������� ���� ������ ��� ��ġ ������ ��
		for(i = stride, istride = 0;istride <= nstride; ditem = ditem->ptrRight, i++) {//��.�Ѱ� �ҽ� �������� ���� ������Ʈ ����Ʈ
			if(i == stride) {//ó������ ����, ���������� �Ѱ� ���� ������Ʈ�� �־ �����.
				i = 0;
				istride++;
				rl = rsc::mpClient->getRLink();
				APPEND_LIST2(head_rl, rl);//�ݳ��ϱ����� ������
				rl->requestActor(sl, ditem, bitem);//ditem�� �Ļ� ���� ���� ���� ������Ʈ
			}
		}
	}
	if(src_make > 0) src_array = make_source_array(dt, batch_data, nbatch, width, dimen, stride, nstride, tcr, agen, adat);
	sl->waitSignal();
	if(src_make < 0) src_array = (void *)bitem->nmaxReduce;
	else if(src_make == 0) src_array = nullx;

	switch(dt) {//������ ��ġ�� bitem->numBind���� ȣ���Ѵ�. width�� ��� ��ġ���� ���� �����̹Ƿ� 
	case BYTEX_TP:					//��� �Ļ� ���� ��ġ�� numBind�� �����ϴ�.
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
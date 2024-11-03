#pragma once

#include <iostream>
#include <vector>
#include <initializer_list>

#include "matrix.h"
#include "trace.h"

#define TENSOR(ts) ((Tensor *)ts)
#define DMATFX(fx) TENSOR(fx->quantum)->mxData
#define GMATFX(fx) TENSOR(fx->quantum)->mxGrad
#define DMATFX_GID(fx) DMATFX(fx)->didground
#define GMATFX_GID(fx) GMATFX(fx)->didground
//#define DMATFX_SHADOW_IT(fx) DMATFX(fx)->settleShadow
//#define GMATFX_SHADOW_IT(fx) GMATFX(fx)->settleShadow
#define DMATFX_SET_GROUND(fx, gid) DMATFX(fx)->groundSet(gid)
#define GMATFX_SET_GROUND(fx, gid) GMATFX(fx)->groundSet(gid)
#define MATFX_INSTON(fx) TENSOR(fx->quantum)->inston
class Tensor : public Typer {
public:
	ubytet tsType;
	bool inston;
	//intt qdim, *qshape;
	Matrixr *mxData, *mxGrad;
	Trace *tsTcr;
	intt colorVer;
	Tensor(Tracer *tcr, ubytet dtp)
	{
		tsType = dtp;
		tsTcr = (Trace *)tcr;
		mxData = mxGrad = nullptr;
		colorVer = tsTcr->colorVersion;
		inston = 0;
	}
	void instTensor(intt ndim, intt *axid, intt gid, Flux *mast)
	{
		//qdim = ndim;
		//qshape = axid;
		mxData = tsTcr->instMatrix(mxData, tsType, ndim, axid, false, gid, mast ? ((Tensor *)mast->quantum)->mxData : nullptr);
		mxGrad = tsTcr->instMatrix(mxGrad, tsType, ndim, axid, true, gid, mast ? ((Tensor *)mast->quantum)->mxGrad : nullptr);
		if(mxData->maxmSize > 0 && mxGrad->maxmSize > 0) inston = 1;
		//else ����ܰ迡���� ���޸𸮰� ���߶� �Ҵ���еɼ��ְ� �̶��� �׷������� �ܰ迡�� ���Ҵ�ȴ�.
		//�̰�� �Ҵ�������� ��ý��� off�Ǿ� ���۷��̼��� ������� �ʰ� ���常 �ȴ�.
	}
	void *begin_p(intt off)
	{
		if(mxData == nullptr) throwFault(-1, "empty tensor\n");
		return mxData->begin_p(off);
	}
	void *begin_wp(intt off)
	{
		if(mxData == nullptr) throwFault(-1, "empty tensor\n");
		return mxData->begin_wp(off);
	}
	void *end_p(void)
	{
		return mxData->end_p();
	}
};

class Apply : public Typer {
public:
	intt apCode, arrangeGround;
	bool dirty, meanApply, trainherit, loadOnExec;
	intt nfanOut, nfanIn, ifanIn, ibfanOut, nbfanOut;
	intt prefPhoto[MX_DIM], suffPhoto[MX_DIM], nprefPhoto, nsuffPhoto;
	Trace *apTcr;
	Capsule *lapInput;
	FxAnchor *lapOuput;
	intt vbackW;
	Apply(Trace *tcr)
	{
		apTcr = tcr;
		nfanOut = nfanIn = ifanIn = ibfanOut = 0;
		lapInput = nullx;
		lapOuput = nullx;
		vbackW = 0;
		meanApply = 0;
		trainherit = 0;
		loadOnExec = 0;
		arrangeGround = -1;
	}
	virtual intt mcalcapp(intt n)
	{
		return 0;
	}
	virtual void *forward(TContext *tcxt, Matrixr *&bw_mutmx) = 0;
	virtual Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx) = 0;
	intt oidground(void)//��ǥ�� ����� ù��°�� �׶��� ���̵� ����
	{
		return DMATFX_GID(lapOuput->fxPoint);
	}
	intt iidground(void)//��ǥ�� �Է��� ù��°�� �׶��� ���̵� ����
	{
		return DMATFX_GID(lapInput->vcaps);
	}
	void registPhoto(Flux *fxp, Flux *fxs)
	{
		dirty = false;
		if(fxp) {
			nprefPhoto = fxp->fdim;
			memcpy(prefPhoto, fxp->fshape, fxp->fdim * sizeof(intt));
		}
		if(fxs) {
			nsuffPhoto = fxs->fdim;
			memcpy(suffPhoto, fxs->fshape, fxs->fdim * sizeof(intt));
		}
	}
	sytet invariance(Flux *fxp, Flux *fxs, Flux *&fxr)
	{
		sytet rv = 1;

		if(fxp) {
			if(nprefPhoto != fxp->fdim) throwFault(-1, "inconsistancy dimension\n");
			if(memcmp(&prefPhoto[1], &fxp->fshape[1], (nprefPhoto - 1) * sizeof(intt))) throwFault(-1, "not first dim inconsistancy\n");
			if(prefPhoto[0] != fxp->fshape[0]) {
				if(fxp->fxType == trainable) throwFault(-1, "train not variable\n");
				prefPhoto[0] = fxp->fshape[0];
				dirty = true;
				rv = 0;
				fxr = fxp;
			}
		}
		if(fxs) {
			if(nsuffPhoto != fxs->fdim) throwFault(-1, "inconsistancy dimension\n");
			if(memcmp(&suffPhoto[1], &fxs->fshape[1], (nsuffPhoto - 1) * sizeof(intt))) throwFault(-1, "not first dim inconsistancy\n");
			if(suffPhoto[0] != fxs->fshape[0]) {
				if(fxs->fxType == trainable) throwFault(-1, "train not variable\n");
				suffPhoto[0] = fxs->fshape[0];
				dirty = true;
				rv = 0;
				fxr = fxs;
			}
		}
		if(rv == 0) return 0;//����ġ
		if(dirty) return 1;//��ġ
		else {//ù��°
			dirty = true;
			return -1;
		}
	}
	sytet invariance2(Flux *fxp) //ù��° ��ũ�� ����ġ�� ����Ѵ�.
	{
		if(nprefPhoto != fxp->fdim) throwFault(-1, "inconsistancy dimension\n");
		if(memcmp(&prefPhoto[1], &fxp->fshape[1], (nprefPhoto - 1) * sizeof(intt))) throwFault(-1, "not first dim inconsistancy\n");
		if(prefPhoto[0] != fxp->fshape[0]) {
			if(fxp->fxType == trainable) throwFault(-1, "train not variable\n");
			prefPhoto[0] = fxp->fshape[0];
			dirty = true;
			return 0;//����ġ
		}
		if(dirty) return 1;//��ġ
		else {//ù��° 
			dirty = true;
			return -1;
		}
	}
	void invariance3(Flux *ip, bool pref) //ù��° ��ũ�� ����ġ�� ����Ѵ�.
	{
		if(pref) {
			if(nprefPhoto != ip->fdim) throwFault(-1, "inconsistancy dimension\n");
			if(memcmp(prefPhoto, ip->fshape, nprefPhoto * sizeof(intt))) throwFault(-1, "dim inconsistancy\n");
		} else {
			if(nsuffPhoto != ip->fdim) throwFault(-1, "inconsistancy dimension\n");
			if(memcmp(suffPhoto, ip->fshape, nsuffPhoto * sizeof(intt))) throwFault(-1, "dim inconsistancy\n");
		}
	}
	Apply *dFanOut(void)
	{
		return this;
	}
	void faninCouning(intt i)
	{
		LOCK_MUT_(apTcr->mutTrc);
		ifanIn += i;
		UNLOCK_MUT_(apTcr->mutTrc);
	}
	bool checkInFwap(void)
	{
		if(apTcr->bwVersion != vbackW) {
			return false;
		}
		LOCK_MUT_(apTcr->mutTrc);
		//if(++apTcr->ifwFork > apTcr->nfwFork) {
		//	//exit(1);//mmm
		//}
		if(++ifanIn == nfanIn) {
			UNLOCK_MUT_(apTcr->mutTrc);
			return true;
		} else {
			UNLOCK_MUT_(apTcr->mutTrc);
			if(ifanIn > nfanIn) throwFault(-1, "check in fault %d %d\n", ifanIn, nfanIn);
			else return false;
		}
	}
	bool checkInBwap(void)
	{
		if(apTcr->bwVersion != vbackW) throwFault(-1, "check in bw version fault\n");
		//if(++apTcr->ibwFork > apTcr->nbwFork) {
		//	//exit(1);//mmm
		//}
		if(++ibfanOut == nbfanOut) {
			//printf("check apply ap: %d num out: %d out cnt: %d pass\n", apCode, nbfanOut, ibfanOut);
			return true;
		} else {
			//printf("check apply ap: %d num out: %d out cnt: %d skip\n", apCode, nbfanOut, ibfanOut);
			if(ibfanOut > nbfanOut) throwFault(-1, "check in fault\n");
			else return false;
		}
	}
	void listingApin(Flux *fx)
	{
		if(fx == nullx) return;
		Capsule *api = new(apTcr)Capsule;
		api->vcaps = fx;
		api->dataShadow = api->gradShadow = nullptr;
		APPEND_LIST(lapInput, api);
	}
	void arrangeLock(Flux *infx, Matrixr *bw_mutmx_r, Capsule *cap, bool fw)
	{
		bool dev_arrange = TRACER(infx->fxTcr)->devArrange && TRACER(infx->fxTcr)->cpmMode <= 0 ? true : false;

		if(infx->ptrMastfx) {//infx�� ptrMastfx���� reshape���� ������ fx�̸� �޸𸮴� �����Ǵµ�
			//�÷����� ���� ���� �����ǹǷ� nbRefer�� �������� �޸𸮰� ���������� �������� �˼������Ƿ�
			bw_mutmx_r = TENSOR(infx->ptrMastfx->quantum)->mxGrad;//������ ���Ѵ�.
			LOCK_MUT_(bw_mutmx_r->mutmtx);
		} else {//if(infx->nbRefer > 1) {//infx�� ������������ �ʾƵ� split�� ��� Ȥ�� �޸� ���޿����� ��°� gid�� �ٸ��������Ƿ� arrange����.
			bw_mutmx_r = TENSOR(infx->quantum)->mxGrad;
			LOCK_MUT_(bw_mutmx_r->mutmtx);
			if(dev_arrange) {
				if(infx != cap->vcaps) throwFault(-1, "bw lock inconsistant fx\n");
				if(fw) {
					//�Է°� ��� �׶��尡 �ٸ��� �Է� �׶��� �޸𸮸� ������ �޸𸮷� �����ϰ� ������ �޸𸮸� ��Ŀ��
					if(DMATFX_GID(infx) != arrangeGround) {// || infx->ds) {
						//if(cap->dbgIdShadow != cap->dataShadow->didshadow || infx->sizefx() != cap->dataShadow->shadowsz) exit(1);
						//printf("%p arange device %p %p %p %p %d %d %d %d %p\n", this, oper, cap, infx, oper->lapOuput->fxPoint, DMATFX_GID(bwfx), iGround, bwfx->scaleout, oper->apCode, cap->dataShadow);
						DMATFX(infx)->arrangeDevice(1, cap->dataShadow->didshadow, cap->dataShadow->devShadow, 1);//ground to shadow copy
						//printf("%p arange device2 %p %p %p\n", this, oper, cap, infx);
					} else DMATFX_SET_GROUND(infx, arrangeGround);
				} else {
					if(GMATFX_GID(infx) != arrangeGround) {// || infx->ds) {
						//printf("%p backward dev set %p %p %p %d %d %d %d %p\n", this, oper, infx, oper->lapOuput->fxPoint, DMATFX_GID(infx), iGround, infx->scaleout, oper->apCode, cap->gradShadow);
						GMATFX(infx)->arrangeDevice(1, cap->gradShadow->didshadow, cap->gradShadow->devShadow, 0);//��.�׶��� -> ������ ī��(�����Ŀ��� �б� ���� ��� ������ �������� �б������κ��� ���� ���������ϹǷ�)
						cap->setShadow(arrangeGround);//data shadow focus
					} else {
						GMATFX_SET_GROUND(infx, arrangeGround);
						DMATFX_SET_GROUND(infx, arrangeGround);
					}
				}
			}
		}
	}
	void arrangeUnlock(Flux *infx, Capsule *cap, bool fw)
	{
		bool dev_arrange = TRACER(infx->fxTcr)->devArrange && TRACER(infx->fxTcr)->cpmMode <= 0 ? true : false;

		if(infx->ptrMastfx) {
			UNLOCK_MUT_(TENSOR(infx->ptrMastfx->quantum)->mxGrad->mutmtx);
		} else {//if(infx->nbRefer > 1) {
			if(fw == false && dev_arrange) {
				//�Է°� ��� �׶��尡 �ٸ��� �Է� ������ �޸𸮸� �׶��� �޸𸮷� �����ϰ� �׶��� �޸𸮸� ��Ŀ��
				if(GMATFX_GID(infx) != arrangeGround) {// || infx->ds) {
					//printf("backward dev arrange %p %p %p %d %d %d %d\n", oper, infx, oper->lapOuput->fxPoint, DMATFX_GID(infx), iGround, infx->scaleout, oper->apCode);
					GMATFX(infx)->arrangeDevice(0, cap->gradShadow->didshadow, cap->gradShadow->devShadow, 0);//shadow to ground copy
				}
			}
			UNLOCK_MUT_(TENSOR(infx->quantum)->mxGrad->mutmtx);
		}
	}
	void multiArrangeLock(bool fw = true) //�Է� �������� �Ѳ����� ��ó���ϰ� ������ ��� ȣ��
	{
		if(arrangeGround < 0) return;//���� �������� ȣ��ɶ��� ���� �ʴ´�. arrangeLock�� ����忡���� ȣ��ǹǷ� �̰� ����ȴ´�.
		Trace *trc = TRACER(lapInput->vcaps->fxTcr);
		Matrixr *mx = nullptr;
		//������� ������ �ƴϸ� �������� �ʴ´�. ������ ������ �ƴϸ� ���� ����.
		if(fw && (trc->devArrange == 0 || trc->cpmMode > 0)) return;
		//�Է��� �����϶��� ���� �� ������ �� ó��(����� ����)
		if(lapInput->ptrRight) LOCK_MUT_(trc->mutArrange);
		if(fw) {
			for(FxAnchor *fxa = lapOuput; fxa; fxa = fxa->ptrRight) {//��Ŀ���� �׶��� �޸𸮰� 
				//�ƴϸ� �� ����� �Է����� �ϴ� ���� ���ö��� ���࿡�� ��Ŀ���� ������� ������ ���̱⶧����
				//���ö��� �������� �ٽ� ��Ŀ���� �׶��� �޸𸮷� ���� �Ѵ�.
				DMATFX_SET_GROUND(fxa->fxPoint, -1);
			}
		}
		for(Capsule *cap = lapInput; cap; cap = cap->ptrRight) {
			arrangeLock(cap->vcaps, mx, cap, fw);
		}
		if(lapInput->ptrRight) UNLOCK_MUT_(trc->mutArrange);
	}
	void multiArrangeUnlock(bool fw = true)
	{
		if(arrangeGround < 0) return;
		Trace *trc = TRACER(lapInput->vcaps->fxTcr);
		//������� ������ �ƴϸ� �������� �ʴ´�. ������ ������ �ƴϸ� ���� ����.
		if(fw && (trc->devArrange == 0 || trc->cpmMode > 0)) return;

		for(Capsule *cap = lapInput; cap; cap = cap->ptrRight) {
			arrangeUnlock(cap->vcaps, cap, fw);
		}
	}
};
#define APC_ARITH	0
#define APC_DOT		1
#define APC_SPLIT	2
#define APC_CONCAT	3
#define APC_RESHAPE	4
#define APC_TRANSPOSE	5
#define APC_SOFTMAX		6
#define APC_SOFTCROSS	7
#define APC_SUM			8
#define APC_MEANSQ		9
#define APC_ACTF		10
#define APC_ADMOPT		11 //optimizer�� �� ���̿� ��ġ ���Ѿ� ��.
#define APC_SGDOPT		12
#define APC_EMBEDDING	13
#define APC_SLICE		14
#define APC_ONEHOT		15
#define APC_ONE			16
#define APC_TWO			17
#define APC_FILL		18
#define APC_COMB		19
#define APC_LAYN		20
#define APC_MATM		21
#define APC_LAYER_N		22
#define APC_BYPASS		23
#define APC_PART		24
#define APC_ADJUST		25
#define APC_SCOOP		26
#define APC_BSUM		27
#define APC_RSUM		28
#define APC_SINPOS		29
#define APC_OVERWRITE	30
#define APC_SWITCH_OUT	31
#define APC_FORK		32
#define APC_ARGMAX		33
#define APC_VMAX		34
class ApArith : public Apply {
public:
	sytet opArith;
	ubytet tScalarv;
	bool broOne;
	bytet svbuf[sizeof(unitt)];
	Flux *arPrefix, *arSuffix, *arOut;
	ArithVar fwArv, bwPreArv, bwSufArv;
	bytet arith_msg[16];//����� ����

	//cdim - ��ε�ĳ��Ʈ�� ���� ������ ��� �����, rdim - ���� ����� �����Ŀ��� �ҽ� �����
	intt set_bro_dim(intt cdim, intt crank[], intt rdim, intt rrank[], intt bro_dim[], intt bro_idx[])
	{
		intt i, j, k, n = cdim - rdim, ndim;

		for(i = j = 0;i < n; i++) {//������ cdim�� �������� ä���.
			ndim = MRANK_SIZE(crank, i) / MRANK_SIZE(crank, i + 1);
			if(ndim == 1) continue;//������� 1�̸� ��ŵ
			bro_idx[j] = i;
			bro_dim[j++] = ndim;
		}
		for(k = 0;k < rdim - 1; k++, i++) {//rdim�� 1�� ������� cdim���� ä���.
			if(rrank[k] > 1) continue;
			ndim = MRANK_SIZE(crank, i) / MRANK_SIZE(crank, i + 1);
			if(ndim == 1) continue;
			bro_idx[j] = i;
			bro_dim[j++] = ndim;
		}
		return j;
	}
	ApArith(Trace *tcr, Flux *fxp, Flux *fxs, Flux *fxo, ubytet qtype, void *sval, sytet arith_op, bool bro_one) : Apply(tcr)
	{
		apCode = APC_ARITH;
		opArith = arith_op;
		arPrefix = fxp;
		arSuffix = fxs;
		arOut = fxo;
		nfanIn = (fxs ? 2 : 1);
		nfanOut = 1;
		fwArv.paintVar = 0;
		broOne = bro_one;
		if(sval) {
			tScalarv = fxp->qType;
			adj_val_type(svbuf, sval, fxp->qType, qtype, 0);
		} else tScalarv = 0;
		if(apTcr->trcmsg[0] != '\0') strcpy(arith_msg, apTcr->trcmsg);
		else strcpy(arith_msg, "arith");
		 
		registPhoto(fxp, fxs);
		listingApin(fxp);
		listingApin(fxs);
	}
	void setArithv(void)
	{
		fwArv.paintVar = 0;
		fwArv.narBro = 0;
		TENSOR(arPrefix->quantum)->mxData->msetArithv(arSuffix ? TENSOR(arSuffix->quantum)->mxData : nullx, TENSOR(arOut->quantum)->mxData, &fwArv, tScalarv);

		if(fwArv.narPre) {
			bwPreArv.paintVar = 1;
			bwPreArv.bwGetOri = BWDIV_PREF;//preǥ��
			bwPreArv.zarPre = arOut->fxSize;
			bwPreArv.zarOut = arPrefix->fxSize;
			bwPreArv.narMast = fwArv.narMast;
			memcpy(bwPreArv.arRankMast, fwArv.arRankMast, fwArv.narMast * sizeof(intt));
			bwPreArv.narPre = fwArv.narMast;//�����Ŀ����� master�� pre
			memcpy(bwPreArv.arRankPre, fwArv.arRankMast, fwArv.narMast * sizeof(intt));
			if(fwArv.narSuf) {
				bwPreArv.zarSuf = arSuffix->fxSize;
				bwPreArv.narSuf = fwArv.narSuf;
				memcpy(bwPreArv.arRankSuf, fwArv.arRankSuf, fwArv.narSuf * sizeof(intt));
			}//else suff�� ��Į��
			bwPreArv.narRet = fwArv.narPre;
			memcpy(bwPreArv.arRankRet, fwArv.arRankPre, fwArv.narPre * sizeof(intt));
			bwPreArv.narBro = set_bro_dim(bwPreArv.narMast, bwPreArv.arRankMast, bwPreArv.narRet, bwPreArv.arRankRet, bwPreArv.broDimRet, bwPreArv.broIdxRet);
			if(bwPreArv.narBro) fwArv.narBro = 1;
			if(opArith == AOP_DIV) bwPreArv.bopAtrith = ABP_DIV_PREF;
			else if(opArith == AOP_MINUS) bwPreArv.bopAtrith = ABP_MINUS_PREF;
			else bwPreArv.bopAtrith = opArith;
		}//else pre�� ��Į�� ������ ����
		if(fwArv.narSuf) {
			bwSufArv.paintVar = 1;
			bwSufArv.bwGetOri = BWDIV_SUFF;//suffǥ��
			bwSufArv.zarPre = arOut->fxSize;
			bwSufArv.zarOut = arSuffix->fxSize;
			bwSufArv.narMast = fwArv.narMast;
			memcpy(bwSufArv.arRankMast, fwArv.arRankMast, fwArv.narMast * sizeof(intt));
			bwSufArv.narPre = fwArv.narMast;//�����Ŀ����� master�� pre
			memcpy(bwSufArv.arRankPre, fwArv.arRankMast, fwArv.narMast * sizeof(intt));
			if(fwArv.narPre) {
				bwSufArv.zarSuf = arPrefix->fxSize;
				bwSufArv.narSuf = fwArv.narPre;
				memcpy(bwSufArv.arRankSuf, fwArv.arRankPre, fwArv.narPre * sizeof(intt));
			}//else pre�� ��Į��
			if(opArith == AOP_DIV) {//div���� �����Ĵ� �и� ���� �����̹Ƿ� suff�� suff��
				if(fwArv.narSuf) bwSufArv.zarSuf = arSuffix->fxSize;
				bwSufArv.narSuf = fwArv.narSuf;
				memcpy(bwSufArv.arRankSuf, fwArv.arRankSuf, fwArv.narSuf * sizeof(intt));
			}
			bwSufArv.narRet = fwArv.narSuf;
			memcpy(bwSufArv.arRankRet, fwArv.arRankSuf, fwArv.narSuf * sizeof(intt));
			bwSufArv.narBro = set_bro_dim(bwSufArv.narMast, bwSufArv.arRankMast, bwSufArv.narRet, bwSufArv.arRankRet, bwSufArv.broDimRet, bwSufArv.broIdxRet);
			if(bwSufArv.narBro) {
				if(fwArv.narBro) fwArv.narBro = 3;
				else fwArv.narBro = 2;
			}
			if(opArith == AOP_DIV) bwSufArv.bopAtrith = ABP_DIV_SUFF;//���� ���Ʒ��� �ٲ�� ABP_MINUS_SUFF�� ��������, ���߿� ����
			else if(opArith == AOP_MINUS) bwSufArv.bopAtrith = ABP_MINUS_SUFF;
			else bwSufArv.bopAtrith = opArith;
		}//else suff�� ��Į�� ������ ����
		if(fwArv.narPre == 0) fwArv.tpArith = AR_T_BROLC;
		else if(fwArv.narSuf == 0) fwArv.tpArith = AR_T_BRORC;
		else if(broOne) fwArv.tpArith = AR_T_ONEBRO;
		else if(fwArv.narBro) fwArv.tpArith = AR_T_BRO;
		else fwArv.tpArith = AR_T_O2O;
		bwPreArv.tpArith = bwSufArv.tpArith = fwArv.tpArith;
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		Flux *fxr;
		sytet iv = invariance(arPrefix, arSuffix, fxr);
		if(iv == 0) {
			arOut->resizing2(fxr, arith_msg);
		}
		if(iv <= 0 || fwArv.paintVar == 0) setArithv();
		if(apTcr->pathPrint) printf("arith fw: %d\n", opArith);
		
		TENSOR(arPrefix->quantum)->mxData->marith(tcxt, (fwArv.narSuf ? TENSOR(arSuffix->quantum)->mxData : nullx), TENSOR(arOut->quantum)->mxData,
			&fwArv, tScalarv, svbuf, nullx, opArith);
		multiArrangeUnlock();
		return arOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);//prefix, suffix�� ������ �Ҷ� ������ ���� �ǹǷ� �ΰ� ���ÿ� ���� �����ؾ� �Ѵ�.
		if(apTcr->pathPrint) printf("arith bw: %d\n", opArith);
		if(fwArv.narPre) {
			//arrangeLock(arPrefix, bw_mutmx, lapInput, 0);
			TENSOR(arOut->quantum)->mxGrad->marith(tcxt, (fwArv.narSuf ? TENSOR(arSuffix->quantum)->mxData : nullx), TENSOR(arPrefix->quantum)->mxGrad,
				&bwPreArv, tScalarv, svbuf, (void *)1, bwPreArv.bopAtrith);//ABP_BWTEST);//
			//arrangeUnlock(arPrefix, lapInput, 0);
		}
		if(fwArv.narSuf) {//pre�� �÷����̸� ����Ʈ �ι�° ��
			//arrangeLock(arSuffix, bw_mutmx, fwArv.narPre ? lapInput->ptrRight : lapInput, 0);
			if(opArith == AOP_DIV) {
				TENSOR(arOut->quantum)->mxGrad->marith(tcxt, TENSOR(arSuffix->quantum)->mxData, TENSOR(arSuffix->quantum)->mxGrad,
					&bwSufArv, tScalarv, svbuf, (void *)1, bwSufArv.bopAtrith);
			} else {
				TENSOR(arOut->quantum)->mxGrad->marith(tcxt, (fwArv.narPre ? TENSOR(arPrefix->quantum)->mxData : nullx), TENSOR(arSuffix->quantum)->mxGrad,
					&bwSufArv, tScalarv, svbuf, (void *)1, bwSufArv.bopAtrith);//ABP_BWTEST);//
			}
			//arrangeUnlock(arSuffix, fwArv.narPre ? lapInput->ptrRight : lapInput, 0);
		}
		multiArrangeUnlock(0);
		return lapInput;
	}
};

#define FX_SIZE_RK(fx, ik) MTX_SIZE_RK(TENSOR(fx->quantum)->mxData, ik)
#define FX_SIZE_RK2(fx, ik) MTX_SIZE_RK2(TENSOR(fx->quantum)->mxData, ik)
#define FX_SIZE_RK3(fx, ik) (FX_SIZE_RK2(fx, ik) < 0 ? 0 : (fx->fdim - 1 == ik ? 1 : FX_SIZE_RK(fx, ik + 1)));
#define FX_SIZE_RK4(fx, ik) (fx->fdim - 1 == ik ? 1 : FX_SIZE_RK(fx, ik + 1));
#define TEST_BLOCK	10
#define DOT_BLOCK	SMALL_BLOCK //TEST_BLOCK
class ApDot : public Apply {
public:
	Flux *dotPrefix, *dotSuffix, *dotOut;
	sytet transOrd;
	bool cublasCheck;
	intt *preFirstAxis, *sufFirstAxis;
	DotVar fwDotv, bwPreDotv, bwSufDotv;
	intt jo_ax_pre[MX_DIM], out_ax_pre[MX_DIM], jo_ax_suf[MX_DIM], out_ax_suf[MX_DIM], preout_do[MX_DIM], sufout_do[MX_DIM];
	intt first_ax_ret[MX_DIM], second_ax_ret[MX_DIM];//cublas dot������ ���.
	intt *devideAxis(Flux *fx, vector<intt> *axis_jo, intt jo[], intt out[], bool &interv)
	{
		vector<intt>::iterator iter, end;
		intt i = 0, j = 0, k = 0, iax;
		intt *first = nullx;

		interv = 0;
		for(;i < fx->fdim; i++) {
			for(iter = axis_jo->begin(), end = axis_jo->end(), iax = 0;iter != end && i != *iter; iter++, iax++);
			if(iter == end) {
				if(k != 0 && (i - out[k-1] > 1 || i - out[k-1] < 1)) interv = 1;
				out[k++] = i;
				if(i == 0) first = out;
			} else {
				if(iax != 0 && (i - jo[iax - 1] > 1 || i - jo[iax - 1] < 1)) interv = 1;
				jo[iax] = i;
				j++;
				if(i == 0) first = jo;
			}
		}
		out[k] = -1;
		jo[j] = -1;
		return first;
	}
	void jo_rank_arange(intt out_ax[], intt njo_ret, intt jo_ax_ret[])
	{//�������� pre�� suf���� [1,2] [2,1]�� ���� ���������� ��� ����� �������� ���� ��Ʈ������ �������� ������ �࿡ ���� ������Ų��.
		struct ax_rink {
			intt axis, ith;
			struct ax_rink *ptrLeft, *ptrRight;
		} *outl = nullx, *jol = nullx, *axr, *axr2;
		intt i;

		for(i = 0;out_ax[i] >= 0; i++) {//���� ������(������)�ϼ��� �ռ��� ������
			axr = (struct ax_rink *)apTcr->bxalloc(sizeof(struct ax_rink));
			axr->axis = out_ax[i];
			for(axr2 = outl;axr2; axr2 = axr2->ptrRight) {
				if(axr2->axis > axr->axis) break;
			}
			if(axr2) {
				INSERT_LIST(outl, axr2, axr);
			} else APPEND_LIST(outl, axr);
		}
		if(i != njo_ret) throwFault(-1, "join dims inconsistant\n");
		for(i = 0;i < njo_ret; i++) {//���� ������(������)�ϼ��� �ռ��� ������
			axr = (struct ax_rink *)apTcr->bxalloc(sizeof(struct ax_rink));
			axr->axis = jo_ax_ret[i];
			axr->ith = i;
			for(axr2 = jol;axr2; axr2 = axr2->ptrRight) {
				if(axr2->axis > axr->axis) break;
			}
			if(axr2) {
				INSERT_LIST(jol, axr2, axr);
			} else APPEND_LIST(jol, axr);
		}
		for(axr = jol, axr2 = outl;axr;) {//���� ��Ʈ������ �� ��ġ�� ������ ��ġ��Ų��.
			out_ax[axr->ith] = axr2->axis;
			axr = axr->ptrRight;
			axr2 = axr2->ptrRight;
		}
	}
	intt max_common(intt u, intt v)
	{
		for(intt t;v;) {
			t = u % v;
			u = v;
			v = t;
		}
		return u;
	}
	void setCudot(DotVar *dotv, sytet bw_get_ori)
	{
		Flux *pre, *suf;
		intt *po, *pj, *so, *sj;
		intt i, k;

		dotv->useCublas = apTcr->tcrCublas;
		if(bw_get_ori == 0) {
			pre = dotPrefix; suf = dotSuffix;
			po = out_ax_pre; pj = jo_ax_pre; so = out_ax_suf; sj = jo_ax_suf;
			for(i = k = 0;out_ax_pre[i] >= 0; i++) first_ax_ret[i] = k++;//�����Ķ��� ���ret��Ʈ������ pre�� suf���� ����
			first_ax_ret[i] = -1;
			for(i= 0;out_ax_suf[i] >= 0; i++) second_ax_ret[i] = k++;
			second_ax_ret[i] = -1;
		} else if(bw_get_ori == BWDIV_PREF) {
			if(fwDotv.transOrder & TOA) {
				pre = dotSuffix;
				suf = dotOut;
				pj = out_ax_suf;//�����Ķ� ��� ������ �����Ķ� ���� ������ �ȴ�.
				sj = second_ax_ret;//ret��Ʈ������ pref �����Ŀ��� �׻� ret��Ʈ������ suf������ ������������ �´�.
				po = jo_ax_suf;//���� ������ ��� �������� ����.
				so = first_ax_ret;//���� ������ ��� �������� ����.
			} else {
				pre = dotOut;
				suf = dotSuffix;
				pj = second_ax_ret;//ret��Ʈ������ pref �����Ŀ��� �׻� ret��Ʈ������ suf������ ������������ �´�.
				sj = out_ax_suf;
				po = first_ax_ret;
				so = jo_ax_suf;
			}
		} else {
			if(fwDotv.transOrder & TOB) {
				pre = dotOut;
				suf = dotPrefix;
				pj = first_ax_ret;//ret��Ʈ������ suf �����Ŀ��� �׻� ret��Ʈ������ pre������ ������������ �´�.
				sj = out_ax_pre;
				po = second_ax_ret;
				so = jo_ax_pre;
			} else {
				pre = dotPrefix;
				suf = dotOut;
				pj = out_ax_pre;
				sj = first_ax_ret;//ret��Ʈ������ suf �����Ŀ��� �׻� ret��Ʈ������ pre������ ������������ �´�.
				po = jo_ax_pre;
				so = second_ax_ret;
			}
		}
		dotv->transOrder = 0;
		if(po[0] == 0) {//dot������ pre�� ���� ������ suf�� ���� ���� ������ �����̰� �������� �̿� �ݴ��̸� ��ġ�� �����Ѵ�.
			for(i = 0, k = 1;pj[i] >= 0; i++) k *= pre->fshape[pj[i]];
		} else {//pre�� �ֻ��� ������ 0 �� �ƴϸ� �ֻ����� pj(��������)�� �����Ƿ� ��ġ�� ���̴�.
			dotv->transOrder = 1;
			for(i = 0, k = 1;po[i] >= 0; i++) k *= pre->fshape[po[i]];
		}
		dotv->prem = pre->fxSize / k;
		dotv->joik = k;
		dotv->lda = k;
		if(sj[0] == 0) {
			for(i = 0, k = 1;sj[i] >= 0; i++) k *= suf->fshape[sj[i]];
		} else {//suf�� �ֻ��� ������ 0 �� �ƴϸ� �ֻ����� so(�������)�� �����Ƿ� ��ġ�� ���̴�.
			dotv->transOrder = (dotv->transOrder == 1 ? 3 : 2);
			for(i = 0, k = 1;so[i] >= 0; i++) k *= suf->fshape[so[i]];
		}
		dotv->sufn = suf->fxSize / k;
		dotv->ldb = dotv->sufn;
		for(i = 0, k = 1;so[i] >= 0; i++) k *= suf->fshape[so[i]];
		dotv->ldc = k;
	}
	void setDotv(DotVar *dotv, Flux *fxp, Flux *fxs, intt out_pre_ax[], intt jo_pre_ax[], intt jo_suf_ax[], intt out_suf_ax[], sytet bw_get_ori)
	{
		intt jo_sz_pre, axid[MX_DIM], i, j, jpre_dims[MX_DIM], jsuf_dims[MX_DIM], k;

		jo_sz_pre = 1;
		for(i = 0;jo_pre_ax[i] >= 0; i++) {
			jpre_dims[i] = fxp->fshape[jo_pre_ax[i]];//���� ������ ��������� pre�� suf�� ���ƾ� �Ϲǰ� ��ǥ�� pre���� ����.
			dotv->joAxisPre[i] = jo_pre_ax[i];//����Ʈ ���� �ε��� ����.
			dotv->sprPreJo[i].rkdim = fxp->fshape[jo_pre_ax[i]];
			dotv->sprPreJo[i].rksz = FX_SIZE_RK2(fxp, jo_pre_ax[i]) < 0 ? 0 : (fxp->fdim -1 == jo_pre_ax[i] ? 1 : FX_SIZE_RK(fxp, jo_pre_ax[i] + 1));
			dotv->sprPreJo[i].rktsz = dotv->sprPreJo[i].rksz * (dotv->sprPreJo[i].rkdim -1);
			jo_sz_pre *= fxp->fshape[jo_pre_ax[i]];
		}
		dotv->njoPre = i; 
		for(i = 0;i < dotv->njoPre; i++) axid[i] = fxp->fshape[*(dotv->joAxisPre + i)];//����Ʈ ���� ���������.
		Matrixr::make_rank_sz(dotv->njoPre, axid, dotv->joRankPre);//����Ʈ ���� ��ũ ������ ���.gpu ĳ�������� ����

		dotv->nJointAxis = 1;
		dotv->jdimEqual = 1;
		for(i = 0;jo_suf_ax[i] >= 0; i++) {
			jsuf_dims[i] = fxs->fshape[jo_suf_ax[i]];
			if(jpre_dims[i] != jsuf_dims[i]) dotv->jdimEqual = 0;
			dotv->joAxisSuf[i] = jo_suf_ax[i];
			dotv->sprSufJo[i].rkdim = fxs->fshape[jo_suf_ax[i]];
			dotv->sprSufJo[i].rksz = FX_SIZE_RK2(fxs, jo_suf_ax[i]) < 0 ? 0 : (fxs->fdim - 1 == jo_suf_ax[i] ? 1 : FX_SIZE_RK(fxs, jo_suf_ax[i] + 1));
			dotv->sprSufJo[i].rktsz = dotv->sprSufJo[i].rksz * (dotv->sprSufJo[i].rkdim -1);
			dotv->nJointAxis *= fxs->fshape[jo_suf_ax[i]];//����Ʈ �� ���� ���.
		}
		dotv->njoSuf = i;
		for(i = 0;i < dotv->njoSuf; i++) axid[i] = fxs->fshape[*(dotv->joAxisSuf + i)];//����Ʈ ���� ���������.
		Matrixr::make_rank_sz(dotv->njoSuf, axid, dotv->joRankSuf);//����Ʈ ���� ��ũ ������ ���.gpu ĳ�������� ����

		if(jo_sz_pre != dotv->nJointAxis) throwFault(-1, "dot inconsistant shape");
		dotv->bwGetOri = bw_get_ori;
		if(bw_get_ori) {
			dotv->bwMxp = (fxp == dotOut ? TENSOR(fxp->quantum)->mxGrad : TENSOR(fxp->quantum)->mxData);
			dotv->bwMxs = (fxs == dotOut ? TENSOR(fxs->quantum)->mxGrad : TENSOR(fxs->quantum)->mxData);
			if(fwDotv.jdimEqual == 0) dotv->intervOut = 1;//�������� ���� ������� ��ġ���� ������ �����Ŀ��� ��ǥ ��ȯ�ϰ� �����Ѵ�.
		}// else fxp->nJoint = fxs->nJoint = dotv->nJointAxis;
		if(bw_get_ori == BWDIV_PREF) {//preffix
			dotv->njoRet = fwDotv.njoPre;
			dotv->noutRet = fwDotv.noutPre;
			memcpy(dotv->joAxisRet, fwDotv.joAxisPre, dotv->njoRet * sizeof(intt));
			memcpy(dotv->outAxisRet, fwDotv.outAxisPre, dotv->noutRet * sizeof(intt));
		} else if(bw_get_ori == BWDIV_SUFF) {//suffix, ��� ������ ��� ���� ���󰡰� szShrinkSuf ������ ������ �����࿡ ������ ������ �ö󰣴�.
			dotv->njoRet = fwDotv.njoSuf;
			dotv->noutRet = fwDotv.noutSuf;
			memcpy(dotv->joAxisRet, fwDotv.joAxisSuf, dotv->njoRet * sizeof(intt));
			memcpy(dotv->outAxisRet, fwDotv.outAxisSuf, dotv->noutRet * sizeof(intt));
		} else dotv->intervOut = 0;
		if(dotv->intervOut && dotv->retFirst == 0) jo_rank_arange(out_pre_ax, dotv->njoRet, dotv->joAxisRet);
		for(i = j = 0;out_pre_ax[i] >= 0; i++) {//����Ʈ�� �ƴ� ��� ���� �ε��� ����
			dotv->outAxisPre[i] = out_pre_ax[i];//[2,3,4]*[3,2]�϶� (1,0)������ dot�ϸ� out�� [2,4,2]���ǰ� �������϶� 
			if(dotv->intervOut) {//pre�� [out*suf]�μ� [2,4,2]*[3,2]�� (2,1)������ dot�Ͽ� [2,4,3]���� ȹ��ǰ� ret(�����Ķ� pre)��
				if(dotv->retFirst) k = dotv->outAxisRet[i];//����� ��ġ(0,2)�� �̹� ������ dot�� predout��[2,4]�� �ְ� ret�� ������
				else k = dotv->joAxisRet[i];//(1)�� ��)���� [3]�� ��ġ���� [2,3,4]�� ����� �����Ķ��� pre�� ȹ���Ѵ�.
			} else k = i;
			dotv->sprPreOut[k].rkPref = 1;
			dotv->sprPreOut[k].rkdim = fxp->fshape[out_pre_ax[i]];
			dotv->sprPreOut[k].rksz = FX_SIZE_RK2(fxp, out_pre_ax[i]) < 0 ? 0 : (fxp->fdim - 1 == out_pre_ax[i] ? 1 : FX_SIZE_RK(fxp, out_pre_ax[i] + 1));
			dotv->sprPreOut[k].rktsz = dotv->sprPreOut[k].rksz * (dotv->sprPreOut[k].rkdim - 1);
		}
		dotv->noutPre = i;
		//dotv->szSuf = 0;
		if(dotv->intervOut && dotv->retFirst) jo_rank_arange(out_suf_ax, dotv->njoRet, dotv->joAxisRet);
		for(i = 0;out_suf_ax[i] >= 0; i++) {
			dotv->outAxisSuf[i] = out_suf_ax[i];
			if(dotv->intervOut) {
				if(dotv->retFirst) k = dotv->joAxisRet[i];//��.
				else k = dotv->outAxisRet[i];
			} else k = dotv->noutPre + i;
			dotv->sprPreOut[k].rkPref = 0;
			dotv->sprPreOut[k].rkdim = fxs->fshape[out_suf_ax[i]];
			dotv->sprPreOut[k].rksz = FX_SIZE_RK2(fxs, out_suf_ax[i]) < 0 ? 0 : (fxs->fdim - 1 == out_suf_ax[i] ? 1 : FX_SIZE_RK(fxs, out_suf_ax[i] + 1));
			dotv->sprPreOut[k].rktsz = dotv->sprPreOut[k].rksz * (dotv->sprPreOut[k].rkdim - 1);
			//if(dotv->szSuf == 0) dotv->szSuf = dotv->sprPreOut[k].rktsz;
		}
		dotv->noutSuf = i;
		dotv->noutRank = dotv->noutPre + dotv->noutSuf;
		for(i = 0;i < dotv->noutRank; i++) axid[i] = dotv->sprPreOut[i].rkdim;
		Matrixr::make_rank_sz(dotv->noutRank, axid, dotv->outRank);
		dotv->intervOut = 0;//���Ŀ��� �� ���� true�� ������ ȣ��Ʈ ����̽��� �޸� ���� ������ ������ ��ġ�Ƿ� ����.
		i = dotv->noutRank - 1;//���� gpu ĳ�� ������ ����
		bool pref = dotv->sprPreOut[i].rkPref;
		for(;;i--) {
			if(dotv->sprPreOut[i].rkPref != pref) break;
		}
		dotv->axisCache = i;
		intt sz_smit = MRANK_SIZE(dotv->outRank, dotv->axisCache + 1);//�����ϴ� �� ���� ���� ���� ��
		if(sz_smit > DOT_BLOCK) {
			intt d = sz_smit / DOT_BLOCK + (sz_smit % DOT_BLOCK ? 1 : 0);//�������� ���� �յ� ���� ���� �ʱⰪ�� ����.
			for(;sz_smit % d && sz_smit / d > DOT_BLOCK / 2; d++);//�ʱ� ���� �������� �յ��� �ƴϸ� �������� ������Ű�� �յ� �������� find
			if(sz_smit % d || sz_smit / d < DOT_BLOCK / 2) {//������ �յ���� ����� �ƴϰų� ���������� ���� ����� Ŀ�� �������� �� �����̸� �յ��� �����Ѵ�.
				dotv->szOutKernel = DOT_BLOCK;
				dotv->fitOutKernel = 0;
			} else {
				dotv->szOutKernel = sz_smit / d;
				dotv->fitOutKernel = 1;
			}
			dotv->szJoKernel = dotv->szOutKernel;//Ŀ�λ������ �ָ��� ������ �ϳ���(smit�� ��ũ�� �ѹ�ü ���ϰ� �����) ó���Ҽ������Ƿ� ���� Ŀ���� ��� Ŀ�� ��ü ������
			dotv->nrecycCache = sz_smit / dotv->szOutKernel + (dotv->fitOutKernel == 0 ? 1 : 0) -1;//�ѹ��� �ݺ��Ǹ� 1����.
			dotv->shareUnit = sz_smit;// * NP_BLOCK;//gpu ĳ�� ���� �� ������ ���. �� �������� �� ���� ó���ҷ���(�׸������� ���Ϸ��� NP_BLOCK�� �����Ѵ�.)
		} else {
			intt n_smit = DOT_BLOCK / sz_smit;//�� ������ȿ� �� ������ �ָ����� ���� ���� ���� ����
			if(n_smit == 0) n_smit = 1;
			dotv->szOutKernel = n_smit * sz_smit;//���� �ָ��� ���� �������� �ѹ��� ó���ϴ� Ŀ�� ������
			dotv->szJoKernel = sz_smit;
			dotv->fitOutKernel = 1;
			dotv->nrecycCache = 0;
			dotv->shareUnit = dotv->szOutKernel;// * NP_BLOCK;//gpu ĳ�� ���� �� ������ ���. �� �������� �� ���� ó���ҷ���(�׸������� ���Ϸ��� NP_BLOCK�� �����Ѵ�.)
		}
		dotv->fitJoKernel = (dotv->nJointAxis % dotv->szJoKernel ? 0 : 1);
		dotv->ncycJo = dotv->nJointAxis / dotv->szJoKernel + (dotv->fitJoKernel == 0 ? 1 : 0) -1;//�ѹ��� �ݺ��Ǹ� 1����.
	}
	ApDot(Trace *tcr, Flux *fxp, Flux *fxs, vector<intt> *axis_p, vector<intt> *axis_s, sytet trans_order, intt &fxo_ndim, intt fxo_axid[]) : Apply(tcr)
	{
		intt i;

		dotPrefix = fxp;
		dotSuffix = fxs;
		transOrd = trans_order;
		apCode = APC_DOT;
		nfanIn = 2;
		nfanOut = 1;

		preFirstAxis = devideAxis(fxp, axis_p, jo_ax_pre, out_ax_pre, bwPreDotv.intervOut);
		sufFirstAxis = devideAxis(fxs, axis_s, jo_ax_suf, out_ax_suf, bwSufDotv.intervOut);
		if(bwPreDotv.intervOut || bwSufDotv.intervOut) cublasCheck = 0;
		else cublasCheck = 1;

		for(i = fxo_ndim = 0;out_ax_pre[i] >= 0; i++) {
			preout_do[i] = fxo_ndim;
			fxo_axid[fxo_ndim++] = fxp->fshape[out_ax_pre[i]];
		}
		preout_do[i] = -1;
		for(i = 0;out_ax_suf[i] >= 0; i++) {
			sufout_do[i] = fxo_ndim;
			fxo_axid[fxo_ndim++] = fxs->fshape[out_ax_suf[i]];
		}
		sufout_do[i] = -1;

		registPhoto(fxp, fxs);
		listingApin(fxp);
		listingApin(fxs);
	}
	//A: [a0, a1] B : [b0, b1]	[3,4] [3,2]		A: [a0, a1, a2] B: [b0, b1, b2]
	//J : [a1, a0] [b0, b1]		[4,3] [3,2]		J: [a0, a1, a2] * [b1, b0, b2]
	//C : [a1, b1]				[4,2]			C: [a0, b2]
	//A^ : [b0, b1] * [c1, c0]	[3,2] [2,4]		A^: [c0, c1] * [b2, b1, b0]
	//B^ : [a0, a1] * [c0, c1]	[3,4] [4,2]		B^: [a1, a2, a0] * [c0, c1]
	//A^: a0�� A�� first, a0�� b0�� ���� ���� b0�� first�� �ϸ� b1�� �� ������ �Ǽ� [b0,b1]�� pre���ǰ� 
	//		b1�� C matrix�� c1�̹Ƿ� [c1, c0]�� suf�� �ȴ�.
	intt *p_that(intt *axis, intt *axis_pair[])
	{
		if(axis_pair[0] == axis) return axis_pair[1];
		else if(axis_pair[1] == axis) return axis_pair[0];
		else return nullx;
	}
	intt i_this(intt *axis, intt *axis_pair[])
	{
		if(axis_pair[0] == axis) return 0;
		else if(axis_pair[1] == axis) return 1;
		else exit(1);
	}
	bool bw_check_dot(Flux *fx_out, Flux *fx_opposit, intt *ax_first, intt *ax_opposit[], intt *ax_join[], intt *ax_out[], 
		intt *ax_do[], intt *&ax_out_pre, intt *&ax_jo_pre, intt *&ax_jo_suf, intt *&ax_out_suf, Flux *&fx_pre, Flux *&fx_suf)
	{
		intt *joint_opposit, i_do, i_opposit;
		//ax_out - �����Ķ� pre�� suf�� axis(�ε���)�� �����Ǵ� ��� ��Ʈ���� axis, ax_do - ��� ��Ʈ������ �ε����� ���� axis
		joint_opposit = p_that(ax_first, ax_join);
		if(joint_opposit) {//�����Ķ� first�� ���εǾ����� 
			ax_out_pre = joint_opposit;//first�� ���� matrix�� ������(���ε�) ��� matirx�� ��� axis�� pre out�� ����.
			ax_jo_pre = p_that(joint_opposit, ax_opposit);//��� matrix�� ���� axis�� ȹ��.
			i_do = i_this(ax_jo_pre, ax_out);//���� axis�� ��¸�Ʈ������ pre�� suf�� ������ΰ� ��å�Ͽ� �� �ε����� ȹ��.
			ax_jo_suf = ax_do[i_do];//ȹ��� ������ �ε����� ��� ��Ʈ���� ���� ���� axis�� suf join���� ����.
			ax_out_suf = ax_do[!i_do];//�� �ݴ븦 suf out���� ����.
			fx_pre = fx_opposit;//first�� ���� ��Ʈ������ �̹� ������ ���� preffix�� �ȴ�.
			fx_suf = fx_out;//�����Ķ��� ��� ��Ʈ������ �̹� ������ ���� suffix�� �ȴ�.
			//printf("%d(%d) %d(%d) %d(%d) %d(%d)\n", fx_pre->fshape[*ax_out_pre], *ax_out_pre,
			//	fx_pre->fshape[*ax_jo_pre], *ax_jo_pre, fx_suf->fshape[*ax_jo_suf], *ax_jo_suf,
			//	fx_suf->fshape[*ax_out_suf], *ax_out_suf);
			return 0;
		} else {//�����Ķ� first�� ���ε��� �ʾ����� ��¸�Ʈ������ ��µ����Ƿ� 
			i_do = i_this(ax_first, ax_out);//first�� ��¸�Ʈ������ pre�� suf�� ������ΰ� ��å�Ͽ� �� �ε����� ȹ��.
			ax_out_pre = ax_do[i_do];//�Ͽ� ��¸�Ʈ�������� �� �ε����� �ش��ϴ� axis�� pre out�� �����ϰ� 
			ax_jo_pre = ax_do[!i_do];//��¸�Ʈ�������� �� �ݴ븦 pre join���� ����.
			i_opposit = i_this(ax_out[!i_do], ax_opposit);//��� ��Ʈ�������� ���εǴ� axis �ε��� ȸ���Ͽ� 
			ax_jo_suf = ax_opposit[i_opposit];//��� ��Ʈ������ ���� axis ����
			ax_out_suf = ax_opposit[!i_opposit];//�� �ݴ븦 ��� axis�� ����.
			fx_pre = fx_out;
			fx_suf = fx_opposit;
			return 1;
		}
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		Flux *fxr;
		sytet iv = invariance(dotPrefix, dotSuffix, fxr);
		if(iv == 0) dotOut->resizing2(fxr, "dotOut");
		if(iv <= 0) {
			if(cublasCheck && (dotOut->qType == tfloat || dotOut->qType == tdouble)) {
				setCudot(&fwDotv, 0);
				setCudot(&bwPreDotv, BWDIV_PREF);
				setCudot(&bwSufDotv, BWDIV_SUFF);
			} else {
				fwDotv.useCublas = bwPreDotv.useCublas = bwSufDotv.useCublas = 0;
			}
			setDotv(&fwDotv, dotPrefix, dotSuffix, out_ax_pre, jo_ax_pre, jo_ax_suf, out_ax_suf, 0);//[fxp(batch seq, feat)][fxs(feat, lattent)]=>[fxo(batch seq, lattent)]
			intt *ax_join[] = { jo_ax_pre, jo_ax_suf };
			intt *ax_out[] = { out_ax_pre, out_ax_suf };
			intt *ax_pre[] = { out_ax_pre, jo_ax_pre };
			intt *ax_suf[] = { jo_ax_suf, out_ax_suf };
			intt *ax_do[] = { preout_do, sufout_do };
			intt *po, *pj, *sj, *so;
			Flux *pre_fx, *suf_fx;
			//printf("%p %p %p %p\n", out_ax_pre, jo_ax_pre, jo_ax_suf, out_ax_suf);
			bwPreDotv.retFirst = bw_check_dot(dotOut, dotSuffix, preFirstAxis, ax_suf, ax_join, ax_out, ax_do, po, pj, sj, so, pre_fx, suf_fx);
			//printf("%p %p %p %p\n", po, pj, sj, so);
			setDotv(&bwPreDotv, pre_fx, suf_fx, po, pj, sj, so, BWDIV_PREF);//[fxo(batch seq, lattent)][fxs^(lattent, feat)]->[fxp(batch seq, feat)]
			bwSufDotv.retFirst = bw_check_dot(dotOut, dotPrefix, sufFirstAxis, ax_pre, ax_join, ax_out, ax_do, po, pj, sj, so, pre_fx, suf_fx);
			//printf("%p %p %p %p\n", po, pj, sj, so);
			setDotv(&bwSufDotv, pre_fx, suf_fx, po, pj, sj, so, BWDIV_SUFF);//[fxp^(feat, batch seq)][fxo(batch seq, lattent)]->[fxs(feat, lattent)]
		}
		if(apTcr->pathPrint) printf("dot fw\n");
		
		TENSOR(dotPrefix->quantum)->mxData->mdot(tcxt, TENSOR(dotSuffix->quantum)->mxData, TENSOR(dotOut->quantum)->mxData,
			&fwDotv, transOrd, 1, nullx);//[fxp(batch seq, feat)][fxs(feat, lattent)]=>[fxo(batch seq, lattent)]
		multiArrangeUnlock();
		return dotOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);//prefix, suffix�� ������ �Ҷ� ������ ���� �ǹǷ� �ΰ� ���ÿ� ���� �����ؾ� �Ѵ�.
		if(apTcr->pathPrint) printf("dot bw\n");
		//arrangeLock(dotPrefix, bw_mutmx, lapInput, 0);
		/*if(apTcr->dbgStep2) {
			((Matrixr *)bwPreDotv.bwMxp)->printo();
			printf("\n===================================\n");
			((Matrixr *)bwPreDotv.bwMxs)->printo();
		}*/
		((Matrixr *)bwPreDotv.bwMxp)->mdot(tcxt, (Matrixr *)bwPreDotv.bwMxs, TENSOR(dotPrefix->quantum)->mxGrad,
			&bwPreDotv, TOB, 1, (void *)1);//[fxo(batch seq, lattent)][fxs^(lattent, feat)]->[fxp(batch seq, feat)]
		//arrangeUnlock(dotPrefix, lapInput, 0);
		//arrangeLock(dotSuffix, bw_mutmx, lapInput->ptrRight, 0);
		((Matrixr *)bwSufDotv.bwMxp)->mdot(tcxt, (Matrixr *)bwSufDotv.bwMxs, TENSOR(dotSuffix->quantum)->mxGrad,
			&bwSufDotv, TOA, 1, (void *)1);//[fxp^(feat, batch seq)][fxo(batch seq, lattent)]->[fxs(feat, lattent)]
		//arrangeUnlock(dotSuffix, lapInput->ptrRight, 0);
		multiArrangeUnlock(0);
		return lapInput;
	}
};
class ApSplit : public Apply {
public:
	Flux *spIn;
	Capsule *spOuts;
	bytet *spLoad;
	intt spAxis;
	bool spStack;

	ApSplit(Trace *tcr, Flux *ip, intt nout, intt axis, bool stacking) : Apply(tcr)
	{
		apCode = APC_SPLIT;
		spIn = ip;
		nfanOut = nout;
		nfanIn = 1;
		spAxis = axis;
		spStack = stacking;
		spOuts = nullx;
		registPhoto(ip, nullx);
		listingApin(ip);
	}
	void outputs(Flux *fxo)
	{
		Capsule *cap = new(apTcr)Capsule;
		cap->vcaps = fxo;
		APPEND_LIST(spOuts, cap);
		
	}
	void opEnding(void)
	{
		spLoad = (bytet *)apTcr->xalloc(nfanOut * sizeof(bytet *));
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		Capsule *cap;
		intt i;
		if(apTcr->pathPrint) printf("split fw\n");
		if(invariance2(spIn) == 0) {
			for(cap = spOuts, i = 0;cap; cap = cap->ptrRight, i++) {
				cap->vcaps->resizing2(spIn, "split");
			}
		}
		for(cap = spOuts, i = 0;cap; cap = cap->ptrRight, i++) {
			*(Matrixr **)((Matrixr **)spLoad + i) = TENSOR(cap->vcaps->quantum)->mxData;
		}
		
		TENSOR(spIn->quantum)->mxData->msplit(tcxt, spLoad, i, spAxis, spStack, false, 1);
		multiArrangeUnlock();
		return spOuts;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		intt i = 0;
		for(Capsule *cap = spOuts;cap; cap = cap->ptrRight, i++) {
			*(Matrixr **)((Matrixr **)spLoad + i) = TENSOR(cap->vcaps->quantum)->mxGrad;
		}
		if(apTcr->pathPrint) printf("split bw\n");
		//arrangeLock(spIn, bw_mutmx, lapInput, 0);
		TENSOR(spIn->quantum)->mxGrad->mconcat(tcxt, spLoad, i, spAxis, spStack, true, 1);
		//arrangeUnlock(spIn, lapInput, 0);
		multiArrangeUnlock(0);
		return lapInput;
	}
};
class ApConcat : public Apply {
public:
	Flux *ccOut;
	bytet *ccLoad;
	intt ccAxis, nccIns;
	bool spStack, parityParty;

	ApConcat(Trace *tcr, Flux *op, intt nin, intt axis, bool stacking) : Apply(tcr)
	{
		apCode = APC_CONCAT;
		ccOut = op;
		nfanIn = nin;
		nfanOut = 1;
		ccAxis = axis;
		spStack = stacking;
		nccIns = 0;
		parityParty = 1;
	}
	void inputs(Flux *fxi)
	{
		nccIns++;
		listingApin(fxi);
	}
	void opEnding(void)
	{
		ccLoad = (bytet *)apTcr->xalloc(nccIns * sizeof(bytet *));
		registPhoto(lapInput->vcaps, nullx);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		if(invariance2(lapInput->vcaps) == 0) {
			//������ ��ġ ������ ��ġ�Ǿ� �ڷΰ��� �ٸ� ������ 0�� �����εǾ� ���յǴ� ��찡 ������ ���� �����Ѵ�.
			//if(ccAxis == 0) throwFault(-1, "variable rank and cat axis error\n");
			ccOut->resizing2(lapInput->vcaps, "concat");
		}
		intt i = 0;
		if(apTcr->pathPrint) printf("concat fw\n");
		for(Capsule *cap = lapInput;cap; cap = cap->ptrRight, i++) *(Matrixr **)((Matrixr **)ccLoad + i) = TENSOR(cap->vcaps->quantum)->mxData;
		
		TENSOR(ccOut->quantum)->mxData->mconcat(tcxt, ccLoad, nccIns, ccAxis, spStack, false, parityParty);
		multiArrangeUnlock();
		return ccOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		intt i = 0;
		if(apTcr->pathPrint) printf("concat bw\n");
		for(Capsule *cap = lapInput;cap; cap = cap->ptrRight, i++) *(Matrixr **)((Matrixr **)ccLoad + i) = TENSOR(cap->vcaps->quantum)->mxGrad;
		
		TENSOR(ccOut->quantum)->mxGrad->msplit(tcxt, ccLoad, nccIns, ccAxis, spStack, true, parityParty);
		multiArrangeUnlock(0);
		return lapInput;
	}
};
class ApReshape : public Apply {
public:
	Flux *rsOut, *rsIn;

	ApReshape(Trace *tcr, Flux *ip, Flux *op) : Apply(tcr)
	{
		apCode = APC_RESHAPE;
		rsIn = ip;
		rsOut = op;
		nfanIn = 1;
		nfanOut = 1;
		registPhoto(ip, nullx);
		listingApin(ip);
		//printf("111 %p %p %d %d\n", ip, op, ip->fxSize, op->fxSize);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		if(invariance2(rsIn) == 0) {
			//if(rsOut->ptrMastfx) printf("111 %p %p %p %p\n", rsIn, rsOut->ptrMastfx, rsIn->begin_p(), rsOut->ptrMastfx->begin_p());
			rsOut->resizing3(rsIn);
			//if(rsOut->ptrMastfx) printf("222 %p %p %p %p\n", rsIn, rsOut->ptrMastfx, rsIn->begin_p(), rsOut->ptrMastfx->begin_p());
		}
		//printf("222 %p %p %d %d\n", rsIn, rsOut, rsIn->fxSize, rsOut->fxSize);
		if(apTcr->pathPrint) printf("reshape fw\n");//�޸𸮰� �����Ǵ� ������ �ٲ����Ƿ� ������Ѵ�.
		if(apTcr->fastshape) TENSOR(rsOut->quantum)->mxData->cpmhot = TENSOR(rsIn->quantum)->mxData->cpmhot;
		else TENSOR(rsOut->quantum)->mxData->inCopy(TENSOR(rsIn->quantum)->mxData, 0);
		multiArrangeUnlock();
		return rsOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		if(apTcr->pathPrint) printf("reshape bw\n");
		if(apTcr->fastshape) TENSOR(rsIn->quantum)->mxGrad->cpmhot = TENSOR(rsOut->quantum)->mxGrad->cpmhot;
		else TENSOR(rsOut->quantum)->mxGrad->mone(tcxt, nullx, TENSOR(rsIn->quantum)->mxGrad, 2, nullx, AOP_ACTF, DJUST_COPY, PDC, 1);
		multiArrangeUnlock(0);
		return lapInput;
	}
};
class ApFill : public Apply {
public:
	Flux *rsOut, *rsIn;
	unit cvFill;
	sytet tpFill;
	ApFill(Trace *tcr, Flux *ip, Flux *op, void *cv, sytet stp) : Apply(tcr)
	{
		apCode = APC_FILL;
		rsIn = ip;
		rsOut = op;
		nfanIn = 1;
		nfanOut = 1;
		tpFill = stp;
		copy_val_type(&cvFill, cv, stp);
		registPhoto(ip, nullx);
		listingApin(ip);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		if(invariance2(rsIn) <= 0) {
			rsOut->resizing2(rsIn, "fill");
			((Tensor *)rsOut->quantum)->mxData->fill(&cvFill, tpFill);
		}
		multiArrangeUnlock();
		if(apTcr->pathPrint) printf("fill fw\n");
		return rsOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		if(apTcr->pathPrint) printf("fill bw\n");
		return lapInput;
	}
};
/*class ApSinPos : public Apply {
public:
	Flux *rsOut, *rsIn;
	intt nseqSinpos;
	ApSinPos(Trace *tcr, Flux *ip, Flux *op, intt nseq) : Apply(tcr)
	{
		apCode = APC_SINPOS;
		rsIn = ip;
		rsOut = op;
		nfanIn = 1;
		nfanOut = 1;
		nseqSinpos = nseq;

		registPhoto(ip, nullx);
		listingApin(ip);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		if(invariance2(rsIn) <= 0) {
			rsOut->resizing2(rsIn);
			Univ uv(SIN_POSITIONAL, 0);
			uv.cvuni = nseqSinpos;
			((Tensor *)rsOut->quantum)->mxData->uniform(nullx, &uv);
		}
		if(apTcr->pathPrint) printf("sinpos fw\n");
		return rsOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		if(apTcr->pathPrint) printf("sinpos bw\n");
		return lapInput;
	}
};*/
class ApBypass : public Apply {
public:
	Flux *rsOut, *rsIn;
	bytet msgpass[64];
	ApBypass(Trace *tcr, Flux *ip, Flux *op, const bytet *msg) : Apply(tcr)
	{
		apCode = APC_BYPASS;
		rsIn = ip;
		rsOut = op;
		nfanIn = 1;
		nfanOut = 1;
		registPhoto(ip, nullx);
		listingApin(ip);
		if(msg) strcpy(msgpass, msg);
		else msgpass[0] = '\0';
		//printf("1111 in: %p out %p\n", rsIn, rsOut);
		//rsIn->shape();
		//rsOut->shape();
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		//printf("222 in: %p out %p\n", rsIn, rsOut);
		//rsIn->shape();
		//rsOut->shape();
		if(apTcr->pathPrint) printf("bypass fw: %s\n", msgpass);
		if(apTcr->bpPrint == 1 || apTcr->bpPrint == 3) TENSOR(rsIn->quantum)->mxData->printo(2);
		if(apTcr->bpPrint > -2 && msgpass[0] != '\0') printf("\nbypass fw: %s\n", msgpass);
		if(apTcr->bpPrint >= 3) rsIn->shape();
		if(invariance2(rsIn) <= 0) rsOut->resizing2(rsIn, "bypass");
		
		TENSOR(rsOut->quantum)->mxData->inCopy(TENSOR(rsIn->quantum)->mxData, 0);
		multiArrangeUnlock();

		return rsOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		//if(!strcmp(msgpass, "cc33\n")) apTcr->dbgStep2 = 1;
		if(apTcr->pathPrint) printf("bypass bw: %s\n", msgpass);
		if(apTcr->bpPrint == 2 || apTcr->bpPrint == 3) TENSOR(rsOut->quantum)->mxGrad->printo();
		if(apTcr->bpPrint > -2 && msgpass[0] != '\0') printf("\nbypass bw: %s\n", msgpass);
		if(apTcr->bpPrint >= 3) rsOut->shape();
		//if(!strcmp(msgpass, "aa11\n")) apTcr->dbgStep2 = 0;
		//arrangeLock(rsIn, bw_mutmx, lapInput, 0);
		TENSOR(rsIn->quantum)->mxGrad->inCopy(TENSOR(rsOut->quantum)->mxGrad, 0);
		//arrangeUnlock(rsIn, lapInput, 0);
		multiArrangeUnlock(0);
		return lapInput;
	}
};
class ApSwitchout : public Apply {
public:
	Flux *apPre, *apSuf, *apOut;
	ApSwitchout(Trace *tcr, Flux *fxp, Flux *fxs, Flux *fxo) : Apply(tcr)
	{
		apCode = APC_SWITCH_OUT;
		apPre = fxp;
		apSuf = fxs;
		apOut = fxo;
		nfanIn = 2;
		nfanOut = 1;

		registPhoto(fxp, fxs);
		listingApin(fxp);
		listingApin(fxs);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		if(apPre->bwbreak == 0) {
			if(invariance2(apPre) == 0) apOut->resizing2(apPre, "ApSwitchout");
			TENSOR(apOut->quantum)->mxData->inCopy(TENSOR(apPre->quantum)->mxData, 0);
		} else if(apSuf->bwbreak == 0) {
			if(invariance2(apSuf) == 0) apOut->resizing2(apSuf, "ApSwitchout");
			TENSOR(apOut->quantum)->mxData->inCopy(TENSOR(apSuf->quantum)->mxData, 0);
		}
		multiArrangeUnlock();
		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		if(apPre->bwbreak == 0) {
			//arrangeLock(apPre, bw_mutmx, lapInput, 0);
			TENSOR(apPre->quantum)->mxGrad->inCopy(TENSOR(apOut->quantum)->mxGrad, 0);
			//arrangeUnlock(apPre, lapInput, 0);
		} else if(apSuf->bwbreak == 0) {
			//arrangeLock(apSuf, bw_mutmx, lapInput, 0);
			TENSOR(apSuf->quantum)->mxGrad->inCopy(TENSOR(apOut->quantum)->mxGrad, 0);
			//arrangeUnlock(apSuf, lapInput, 0);
		}
		multiArrangeUnlock(0);
		return lapInput;
	}
};
class ApPartition : public Apply {
public:
	Flux *rsOut, *rsIn;
	ApPartition(Trace *tcr, Flux *ip, Flux *op) : Apply(tcr)
	{
		apCode = APC_PART;
		rsIn = ip;
		rsOut = op;
		nfanIn = 1;
		nfanOut = 1;
		registPhoto(ip, nullx);
		//listingApin(op);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		if(apTcr->pathPrint) printf("partition fw\n");
		multiArrangeLock();
		if(invariance2(rsIn) <= 0) rsOut->resizing2(rsIn, "partition");
		
		TENSOR(rsOut->quantum)->mxData->inCopy(TENSOR(rsIn->quantum)->mxData, 0);
		multiArrangeUnlock();
		//rsIn->printo();
		return nullx;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		if(apTcr->pathPrint) printf("partition bw\n");
		return nullx;
		//TENSOR(rsIn->quantum)->mxGrad->inCopy(TENSOR(rsOut->quantum)->mxGrad, 0);
		//return lapInput;
	}
};
class ApAdjust : public Apply {
public:
	Flux *adjOut, *adjIn;
	ApAdjust(Trace *tcr, Flux *ip, Flux *op) : Apply(tcr)
	{
		apCode = APC_ADJUST;
		adjIn = ip;
		adjOut = op;
		nfanIn = 1;
		nfanOut = 1;
		registPhoto(ip, nullx);
		listingApin(ip);
		//printf("1111 in: %p out %p\n", adjIn, adjOut);
		//adjIn->shape();
		//adjOut->shape();
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		//printf("222 in: %p out %p\n", adjIn, adjOut);
		//adjIn->shape();
		//adjOut->shape();
		if(apTcr->pathPrint) printf("adjust fw\n");
		if(invariance2(adjIn) == 0) adjOut->resizing2(adjIn, "adjust");
		return adjOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		if(apTcr->pathPrint) printf("adjust bw\n");
		return lapInput;
	}
};
class ApTranspose : public Apply {
public:
	Flux *tsOut, *tsIn;
	TransVar trsVar, bwTrsv;

	ApTranspose(Trace *tcr, Flux *ip, Flux *op, intt txid[]) : Apply(tcr)
	{
		apCode = APC_TRANSPOSE;
		tsIn = ip;
		tsOut = op;
		nfanIn = 1;
		nfanOut = 1;
		trsVar.ntrDims = op->fdim;
		memcpy(trsVar.trTxid, txid, sizeof(intt) * trsVar.ntrDims);
		bwTrsv.ntrDims = op->fdim;
		for(intt i = 0;i < bwTrsv.ntrDims; i++) {
			intt j = i;
			for(;txid[j] != i; j = txid[j]);//��ġ�ε����� �ڱ������Ѱ��� �� �ڸ��� �ƴ� 
			bwTrsv.trTxid[i] = j;//������ �ڱ������ε����� j�� ������ ��ġ �ε���
		}
		registPhoto(ip, nullx);
		listingApin(ip);
	}
	void mktrs_map(Flux *tsin, intt txid[], TransRank *trsrk)
	{
		intt n = tsin->fdim, i = 0;

		for(;i < n; i++) {
			trsrk[i].trsdim = tsin->fshape[txid[i]];
			trsrk[i].trssz = FX_SIZE_RK2(tsin, txid[i]) < 0 ? 0 : (tsin->fdim - 1 == txid[i] ? 1 : FX_SIZE_RK(tsin, txid[i] + 1));
			trsrk[i].trstsz = trsrk[i].trssz * (trsrk[i].trsdim - 1);
		}
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		sytet iv = invariance2(tsIn);
		if(iv == 0) {//ù��° ��ũ�� ��ȭ�ƴµ�
			//for(intt i = 1;i < trsVar.ntrDims; i++) {//Ÿ�ٿ� �ٸ� ��ũ�� ��ġ�Ǹ� �ȵȴ�.
			//	if(trsVar.trTxid[i] == 0) throwFault(-1, "first variable rank transposed error\n");
			//}
			tsOut->resizing2(tsIn, "resize_fhs");
		}
		if(apTcr->pathPrint) printf("transpose fw\n");
		if(iv <= 0) {
			mktrs_map(tsIn, trsVar.trTxid, trsVar.tspmap);
			mktrs_map(tsOut, bwTrsv.trTxid, bwTrsv.tspmap);
			memcpy(trsVar.trRankRet, TENSOR(tsOut->quantum)->mxData->mxranksz, trsVar.ntrDims * sizeof(intt));
			memcpy(bwTrsv.trRankRet, TENSOR(tsIn->quantum)->mxData->mxranksz, bwTrsv.ntrDims * sizeof(intt));
		}
		
		TENSOR(tsIn->quantum)->mxData->mtranspose(tcxt, TENSOR(tsOut->quantum)->mxData, &trsVar, 0);
		multiArrangeUnlock();
		return tsOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		if(apTcr->pathPrint) printf("transpose bw\n");
		//arrangeLock(tsIn, bw_mutmx, lapInput, 0);
		TENSOR(tsOut->quantum)->mxGrad->mtranspose(tcxt, TENSOR(tsIn->quantum)->mxGrad, &bwTrsv, 1);
		//arrangeUnlock(tsIn, lapInput, 0);
		multiArrangeUnlock(0);
		return lapInput;
	}
};
class ApSoftmax : public Apply {
public:
	Flux *sfOut, *sfIn;
	Matrixr *sfmSum, *sfmMax, *sfmBuf;
	OprVar1 sfOprv;
	Flux *smcross, *smcross2, *a_dx, *a_dx2;
	Apply *cross_ap, *cross_ap2, *diag_ap, *diag_ap2;

	ApSoftmax(Trace *tcr, Flux *ip, Flux *op) : Apply(tcr)
	{
		apCode = APC_SOFTMAX;
		sfIn = ip;
		sfOut = op;
		nfanIn = 1;
		nfanOut = 1;
		sfmSum = sfmMax = sfmBuf = nullx;

		registPhoto(ip, nullx);
		listingApin(ip);
		
		intt dim[MX_DIM], i = 0;
		for(;i < sfOut->fdim; i++) dim[i] = sfOut->fshape[i];
		dim[i] = dim[i - 1];
		tcr->directx(1);//�׶��� �������� �ʰ� �Ѵ�.
		smcross = flux(tcr, sfOut->fdim +1, dim, sfIn->qType, variable);
		smcross->scaleout = ip->scaleout;//smcross�� ip�� ��ġ �����Ǿ� �ϹǷ� �����ϰ� ����
		smcross2 = *smcross * -1.0;//-aa
		a_dx = 1.0 - *sfOut;//(1-a)
		a_dx->scaleout = ip->scaleout;//sfOut�� �����Ͼƿ��� ���ϵǾ� �����ǹǷ� ��������
		a_dx2 = *sfOut * *a_dx;//a(1-a)
		tcr->directx(0);//����

		sfmSum = apTcr->instMatrix(sfmSum, sfOut->qType, sfIn->fdim - 1, sfIn->fshape, false, JUST_SIZEDEF);
		sfmMax = apTcr->instMatrix(sfmMax, sfOut->qType, sfIn->fdim - 1, sfIn->fshape, false, JUST_SIZEDEF);
		sfmBuf = apTcr->instMatrix(sfmBuf, sfOut->qType, 1, &sfIn->fshape[sfIn->fdim - 1], false, JUST_SIZEDEF);
	}
	intt mcalcapp(intt n) //��ġ�� n���� ������ ���ö��� ���� ��� �޸� ���, �̸����� �̵� ��Ʈ������
	{						//������ �޸� �Ҵ� ���� ������ ��길 �Ҽ��ְ� ����
		intt sz = 0;
		sz += smcross->sizefx2(n);
		sz += sfmSum->sizem2(n);
		sz += sfmMax->sizem2(n);
		sz += sfmBuf->sizem();
		return sz;
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		sytet iv = invariance2(sfIn);
		if(iv == 0) {
			sfOut->resizing2(sfIn, "softmax");
			//���ö����� ����� �ƴϾ �׷����� ���� ��� �׶��忡 �ڵ� ���Ҵ���� �����Ƿ� ���⼭ ���� ���Ҵ�
			//������ ���游�̸� smcross2, a_dx, a_dx2�� ����������� ������ üũ�Ǿ� �ڵ����� ���Ҵ�Ǿ� 
			//���⼭ ���� ���Ҵ� �� �ʿ������ �׶��� ���̵� ����� ��� ������ gid�� �Ҵ�Ǿ� �ϹǷ� ���� ���Ҵ�.
			smcross->resizing5(sfIn->scaleout, DMATFX(sfOut)->didground);
			smcross2->resizing5(sfIn->scaleout, DMATFX(sfOut)->didground);
			a_dx->resizing5(sfIn->scaleout, DMATFX(sfOut)->didground);
			a_dx2->resizing5(sfIn->scaleout, DMATFX(sfOut)->didground);
		}
		if(iv <= 0) {
			intt gid = DMATFX(sfOut)->didground;//gid �׶��忡���� �Ҵ� ������ �� mcalc app���� ���Ǿ� Ȯ���Ǿ���.
			sfmSum = apTcr->instMatrix(sfmSum, sfOut->qType, sfIn->fdim -1, sfIn->fshape, false, gid);
			sfmMax = apTcr->instMatrix(sfmMax, sfOut->qType, sfIn->fdim - 1, sfIn->fshape, false, gid);
			sfmBuf = apTcr->instMatrix(sfmBuf, sfOut->qType, 1 , &sfIn->fshape[sfIn->fdim -1], false, gid);
			sfOprv.noprDims1 = sfOut->fdim;
			memcpy(sfOprv.oprRank1, TENSOR(sfOut->quantum)->mxData->mxranksz, sizeof(intt) * sfOprv.noprDims1);
		}
		if(apTcr->pathPrint) printf("softmax fw\n");
		
		TENSOR(sfIn->quantum)->mxData->msoftmax(tcxt, TENSOR(sfOut->quantum)->mxData, sfmSum, sfmMax, sfmBuf, &sfOprv);
		multiArrangeUnlock();
		return sfOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		OneVar onevar;
		intt idf = sfOut->fdim - 1;
		if(apTcr->pathPrint) printf("softmax bw\n");
		//sfOut�� ���� shape�� [batch, seq, feat]�̳� dimension 1�� ������ ũ�Ⱑ �����Ƿ�  
		onevar.idxOne[0] = sfOut->fshape[idf];//M, //[batch, seq, feat, 1], mat mul�ϱ����� K������ 1�� �ش�.
		onevar.idxOne[1] = 1;//K
		onevar.idxOne[2] = sfOut->fshape[idf];//N //[batch, seq, 1, feat]
		onevar.idxOne[3] = 0;
		TENSOR(sfOut->quantum)->mxData->mone(tcxt, TENSOR(sfOut->quantum)->mxData,
			TENSOR(smcross->quantum)->mxData, 2, &onevar, AOP_MATMUL, 0, PDG, 1);//[batch, seq, feat, feat]
		smcross2->exec(tcxt);//-aa [batch, seq, feat, feat]
		a_dx->exec(tcxt);//(1-a) [batch, seq, feat]
		a_dx2->exec(tcxt);//a(1-a) [batch, seq, feat]
		TENSOR(a_dx2->quantum)->mxData->mdiag_fill(nullx, TENSOR(smcross2->quantum)->mxData);//[[(1-a)a -aa]
																	//[batch, seq, feat, feat] [[-aa (1-a)a]]
		//delta-out([batch, seq, feat]) * delta-a([batch, seq, feat, feat])
		onevar.idxOne[0] = 1;//M, //[batch, seq, 1, feat], delta-out�� ���� shape�� [batch, seq, feat]�̳� dimension 1�� ������ ũ�Ⱑ �����Ƿ� mat mul�ϱ����� M(�ο�)������ 1�� �ش�.
		onevar.idxOne[1] = sfOut->fshape[idf];//K(feat)
		onevar.idxOne[2] = sfIn->fshape[idf];//N(feat) //[batch, seq, feat, feat]
		onevar.idxOne[3] = 0;//���� ��� ���(delta-in)�� [batch, seq, 1, feat]�� �Ǿ��ϳ� sfIn�� ����� [batch, seq, feat]�� ����ǵ� �ȴ�. 1�� ũ�Ⱑ �����Ƿ�
		//arrangeLock(sfIn, bw_mutmx, lapInput, 0);
		TENSOR(sfOut->quantum)->mxGrad->mone(tcxt, TENSOR(smcross2->quantum)->mxData,
			TENSOR(sfIn->quantum)->mxGrad, 2, &onevar, AOP_MATMUL, 0, PDG, 1);//delta-in([batch, seq, feat])
		//arrangeUnlock(sfIn, lapInput, 0);
		multiArrangeUnlock(0);
		return lapInput;
	}
};
class ApSoftmaxCrossE : public Apply {
public:
	Flux *sfcOut, *sfcIn, *sfcTar;
	Matrixr *sfmSum, *sfmMax, *sfomtx, *sfoBuf;
	ArithVar minusArv, devideArv;
	//unit withBatch;
	OneVar smcVar;

	ApSoftmaxCrossE(Trace *tcr, Flux *ip, Flux *tp, Flux *op) : Apply(tcr)
	{
		apCode = APC_SOFTCROSS;
		sfcIn = ip;
		sfcOut = op;
		sfcTar = tp;
		nfanIn = 2;
		nfanOut = 1;
		sfmSum = sfmMax = sfomtx = sfoBuf = nullx;
		minusArv.paintVar = devideArv.paintVar = 0;

		registPhoto(ip, tp);
		listingApin(ip);
		listingApin(tp);//tpŸ���� ���������� ���� ���� �� ���ΰ� asce), fsce)���� fxt reference���� �ʰ� ����

		sfmSum = apTcr->instMatrix(sfmSum, sfcOut->qType, sfcIn->fdim - 1, sfcIn->fshape, false, JUST_SIZEDEF);
		sfmMax = apTcr->instMatrix(sfmMax, sfcOut->qType, sfcIn->fdim - 1, sfcIn->fshape, false, JUST_SIZEDEF);
		sfomtx = apTcr->instMatrix(sfomtx, sfcOut->qType, sfcIn->fdim, sfcIn->fshape, false, JUST_SIZEDEF);
		sfoBuf = apTcr->instMatrix(sfoBuf, sfcOut->qType, 1, &sfcIn->fshape[sfcIn->fdim - 1], false, JUST_SIZEDEF);
	}
	intt mcalcapp(intt n) //��ġ�� n���� ������ ���ö��� ���� ��� �޸� ���, �̸����� �̵� ��Ʈ������
	{						//������ �޸� �Ҵ� ���� ������ ��길 �Ҽ��ְ� ����
		intt sz = 0;
		sz += sfmSum->sizem2(n);
		sz += sfmMax->sizem2(n);
		sz += sfomtx->sizem2(n);
		sz += sfoBuf->sizem();
		return sz;
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		Flux *fxr;
		sytet iv = invariance(sfcIn, sfcTar, fxr);
		if(iv == 0) sfcOut->resizing2(fxr, "softmax cross entroy");
		if(iv <= 0) {
			intt gid = DMATFX(sfcOut)->didground;//gid �׶��忡���� �Ҵ� ������ �� mcalc app���� ���Ǿ� Ȯ���Ǿ���.
			sfmSum = apTcr->instMatrix(sfmSum, sfcOut->qType, sfcIn->fdim - 1, sfcIn->fshape, false, gid);
			sfmMax = apTcr->instMatrix(sfmMax, sfcOut->qType, sfcIn->fdim - 1, sfcIn->fshape, false, gid);
			sfomtx = apTcr->instMatrix(sfomtx, sfcOut->qType, sfcIn->fdim, sfcIn->fshape, false, gid);
			sfoBuf = apTcr->instMatrix(sfoBuf, sfcOut->qType, 1, &sfcIn->fshape[sfcIn->fdim - 1], false, gid);
			minusArv.paintVar = devideArv.paintVar = 0;
			intt bn = (apTcr->ibatchSize ? apTcr->ibatchSize : sfcIn->fshape[0]);
			if(bn >= apTcr->gradby) bn /= apTcr->gradby;
			//intt bn = (apTcr->ibatchSize ? apTcr->ibatchSize : MTX_SIZE(TENSOR(sfcIn->quantum)->mxData));
			copy_cval_type(smcVar.idxOne, bn, sfomtx->mxType);//��ġ ������
			//copy_cval_type(smcVar.idxOne, MTX_SIZE(sfmSum), sfomtx->mxType);//��ġ ������
			//copy_cval_type(&withBatch, MTX_SIZE(sfmSum), sfomtx->mxType);//��ġ ������
		}
		if(apTcr->pathPrint) printf("softmax cross entroy err fw\n");
		
		TENSOR(sfcIn->quantum)->mxData->msoftmax(tcxt, sfomtx, sfmSum, sfmMax, sfoBuf, nullx);
		sfomtx->msoftx_cross_e(tcxt, TENSOR(sfcOut->quantum)->mxData, TENSOR(sfcTar->quantum)->mxData);//Ÿ�ٰ��� 0�̸�
		//TENSOR(sfcOut->quantum)->mxData->printo();//0���� �������Ƿ� �ȵȴ�. ũ�ν���Ʈ���Ǵ� �̻갪�� ���̰� ���� 0���� �ΰ����� �־��Ѵ�.
		multiArrangeUnlock();
		return sfcOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		if(apTcr->pathPrint) printf("softmax cross entroy err bw\n");
		multiArrangeLock(0);
		//arrangeLock(sfcIn, bw_mutmx, lapInput, 0);
		//sfomtx->marith(tcxt, TENSOR(sfcTar->quantum)->mxData, TENSOR(sfcIn->quantum)->mxGrad,
		//				&minusArv, 0, nullx, nullx, AOP_MINUS);
		//TENSOR(sfcIn->quantum)->mxGrad->marith(tcxt, nullx, TENSOR(sfcIn->quantum)->mxGrad,
		//	&devideArv, sfomtx->mxType, &withBatch, nullx, AOP_DIV);//��ġ ������� ����.
		sfomtx->mone(tcxt, TENSOR(sfcTar->quantum)->mxData, TENSOR(sfcIn->quantum)->mxGrad,
			2, &smcVar, AOP_ACTF, DLOSS_FUNC, PDC, 1);
		//arrangeUnlock(sfcIn, lapInput, 0);
		/*arrangeLock(sfcTar, bw_mutmx, lapInput, 0);
		//asce.����. Ÿ���� ���� ���Ĵ� sgd�� ��Ȯ�ϳ� cross e�� ������ �߻�, ���� �̺н��� Ʋ���� ����. ���߿� ����.
		//TENSOR(sfcTar->quantum)->mxData->marith(tcxt, sfomtx, TENSOR(sfcTar->quantum)->mxGrad,
		//	&minusArv, 0, nullx, nullx, AOP_MINUS);
		//TENSOR(sfcTar->quantum)->mxGrad->marith(tcxt, nullx, TENSOR(sfcTar->quantum)->mxGrad,
		//	&devideArv, sfomtx->mxType, &withBatch, nullx, AOP_DIV);//��ġ ������� ����.
		TENSOR(sfcTar->quantum)->mxData->mone(tcxt, sfomtx, TENSOR(sfcTar->quantum)->mxGrad,
			2, &smcVar, AOP_ACTF, DLOSS_FUNC, PDC, 1);
		arrangeUnlock(sfcTar, lapInput, 0);*/
		multiArrangeUnlock(0);
		return lapInput;
	}
};
class ApSum : public Apply {
public:
	Flux *apOut, *apIn;
	bool withMean;
	intt mb_axid[1];
	Matrixr *meanby;

	ApSum(Trace *tcr, Flux *ip, Flux *op, bool mean) : Apply(tcr)
	{
		apCode = APC_SUM;
		apIn = ip;
		apOut = op;
		withMean = mean;
		nfanIn = 1;
		nfanOut = 1;
		meanby = nullx;
		if(mean) {
			meanApply = 1;
			mb_axid[0] = 1;
			meanby = apTcr->instMatrix(meanby, apIn->qType, 1, mb_axid, false);//axid�� ��Ʈ���� �������ο��� �����ͷ� �����ȴ�.
			floatt cv = 1.0 / apIn->fxSize;
			meanby->fill(&cv, tfloat);
		}
		registPhoto(ip, nullx);
		listingApin(ip);
	}
	intt mcalcapp(intt n) //��ġ�� n���� ������ ���ö��� ���� ��� �޸� ���
	{
		if(meanby) return meanby->sizem();
		else return 0;
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		if(withMean && invariance2(apIn) == 0) {
			apTcr->instMatrix(meanby, apIn->qType, 1, mb_axid, false, DMATFX(apOut)->didground);
			floatt cv = 1.0 / apIn->fxSize;
			meanby->fill(&cv, tfloat);
		}
		if(apTcr->pathPrint) printf("sum fw: %d\n", withMean);
		
		TENSOR(apIn->quantum)->mxData->msum(tcxt, TENSOR(apOut->quantum)->mxData, nullx, withMean);
		multiArrangeUnlock();
		//apIn->printo();
		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		if(apTcr->pathPrint) printf("sum bw: %d\n", withMean);
		//arrangeLock(apIn, bw_mutmx, lapInput, 0);
		TENSOR(apOut->quantum)->mxGrad->mone(tcxt, meanby, TENSOR(apIn->quantum)->mxGrad, 2, nullx, AOP_ACTF, DJUST_COPY2, PDN, 1);
		//arrangeUnlock(apIn, lapInput, 0);
		multiArrangeUnlock(0);
		return lapInput;
	}
};
class ApBatchSum : public Apply {
public:
	Flux *apOut, *apIn;
	OneVar bsumvar;

	ApBatchSum(Trace *tcr, Flux *ip, Flux *op, bool mean) : Apply(tcr)
	{
		apCode = APC_BSUM;
		apIn = ip;
		apOut = op;
		nfanIn = 1;
		nfanOut = 1;
		bsumvar.idxOne[0] = mean;
		bsumvar.idxOne[1] = FX_SIZE_RK(ip, 1);
		*(floatt *)&bsumvar.idxOne[2] = 1 / bsumvar.idxOne[1];
		if(mean) meanApply = 1;
		/*mb_axid[0] = 1;
		sumsz = apTcr->instMatrix(sumsz, apIn->qType, 1, mb_axid, 0);//axid�� ��Ʈ���� �������ο��� �����ͷ� �����ȴ�.
		sumSize = FX_SIZE_RK3(ip, 0);
		floatt cv = (floatt)sumSize;
		sumsz->fill(&cv, tfloat);
		if(mean) {
			meanApply = 1;
			withMean = 2;
		} else withMean = 1;*/
		registPhoto(ip, nullx);
		listingApin(ip);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		if(invariance2(apIn) == 0) {
			apOut->resizing2(apIn, "sum");
		}
		if(apTcr->pathPrint) printf("sum fw: %d\n", bsumvar.idxOne[0]);
		
		TENSOR(apIn->quantum)->mxData->mone(tcxt, nullx, TENSOR(apOut->quantum)->mxData, 0, &bsumvar, AOP_BSUM, 0, PDN, 2, bsumvar.idxOne[1]);
		multiArrangeUnlock();
		//apIn->printo();
		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		if(apTcr->pathPrint) printf("sum bw: %d\n", bsumvar.idxOne[0]);
		//arrangeLock(apIn, bw_mutmx, lapInput, 0);
		TENSOR(apOut->quantum)->mxGrad->mone(tcxt, nullx, TENSOR(apIn->quantum)->mxGrad, 2, &bsumvar, AOP_BSUM, 1, PDN, 1, bsumvar.idxOne[1]);
		//arrangeUnlock(apIn, lapInput, 0);
		multiArrangeUnlock(0);
		return lapInput;
	}
};
/*class ApReductSum : public Apply {
public:
	Flux *apOut, *apIn;
	bool withMean;
	intt mb_axid[1];
	OneVar redsum;

	ApReductSum(Trace *tcr, Flux *ip, Flux *op, intt sum_xid[], bool mean) : Apply(tcr)
	{
		apCode = APC_RSUM;
		apIn = ip;
		apOut = op;
		withMean = mean;
		nfanIn = 1;
		nfanOut = 1;
		
		redsum.idxOne[0] = nsum;
		redsum.idxOne[1] = nnot;

		intt *p = &redsum.idxOne[MX_DIM];
		for(intt i = 0;i < nsum; i++, ip++) *p = FX_SIZE_RK(ip, sum_xid[i]);
		p = &redsum.idxOne[MX_DIM * 2];
		for(intt i = 0;i < nnot; i++, ip++) *p = FX_SIZE_RK(ip, not_xid[i]);

		memcpy(&redsum.idxOne[MX_DIM], sum_xid, nsum * sizeof(intt));
		memcpy(&redsum.idxOne[MX_DIM * 2], not_xid, nnot * sizeof(intt));
		
		intt i = 0, j = 0;
		redsum.idxOne[0] = ip->fdim;
		intt *p = &redsum.idxOne[MX_DIM];
		for(;i < ip->fdim; i++) *(p + i) = FX_SIZE_RK3(ip, i);
		p = &redsum.idxOne[MX_DIM * 2];
		for(i = 0;i < ip->fdim; i++) {
			if(sum_xid[i]) *(p + i) = 0;
			else {
				*(p + i) = FX_SIZE_RK(op, j);
				j++;
			}
		}
		memcpy(&redsum.idxOne[MX_DIM * 3], ip->fshape, sizeof(intt) * ip->fdim);
		if(mean) {
			floatt div = 0;
			for(i = 0;i < ip->fdim; i++) {
				if(sum_xid[i] == 0) div += ip->fshape[i];
			}
			*(floatt *)&redsum.idxOne[1] = 1 / div;
		} else *(floatt *)&redsum.idxOne[1] = 0;
		registPhoto(ip, nullx);
		listingApin(ip);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		if(invariance2(apIn) == 0) {
			redsum.idxOne[MX_DIM * 3] = apIn->fshape[0];
			if(withMean) {
				floatt div = 0;
				intt *p = &redsum.idxOne[MX_DIM * 2];
				for(intt i = 0;i < apIn->fdim; i++) {
					if(*(p + i) == 0) div += apIn->fshape[i];
				}
				*(floatt *)&redsum.idxOne[1] = 1 / div;
			}
		}
		if(apTcr->pathPrint) printf("sum fw: %d\n", withMean);
		
		TENSOR(apIn->quantum)->mxData->mone(tcxt, nullx, TENSOR(apOut->quantum)->mxData, 2, &redsum, AOP_ACTF, RSUM, PDN, 2, apOut->sizef());
		multiArrangeUnlock();
		//apIn->printo();
		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		if(apTcr->pathPrint) printf("sum bw: %d\n", withMean);
		//arrangeLock(apIn, bw_mutmx, lapInput, 0);
		TENSOR(apOut->quantum)->mxGrad->mone(tcxt, nullx, TENSOR(apIn->quantum)->mxGrad, 2, &redsum, AOP_ACTF, DRSUM, PDN, 1);
		//arrangeUnlock(apIn, lapInput, 0);
		multiArrangeUnlock(0);
		return lapInput;
	}
};*/
class ApMeanSquareE : public Apply {
public:
	Flux *apOut, *apIn, *apTar;
	//Matrixr *squareMat;
	//ArithVar minusArv, divArv, mulArv, dminusArv;
	//unit withBatch;//, withHalf;
	OneVar mseVar;
	bool withMean;
	ApMeanSquareE(Trace *tcr, Flux *ip, Flux *tp, Flux *op, bool mean) : Apply(tcr)
	{
		apCode = APC_MEANSQ;
		apIn = ip;
		apOut = op;
		apTar = tp;
		nfanIn = 2;
		nfanOut = 1;
		//squareMat = nullx;
		withMean = mean;//��ձ��� ���ϴ� ���� �ƴϸ� ��ġ ������(��ġ�� ������ ��������) 
						//���������� ����, �����Ĵ� ������ �ҽ��� Ÿ���� ���̰����� ���ϹǷ� �����ϰ� ó��
		//minusArv.paintVar = divArv.paintVar = mulArv.paintVar = dminusArv.paintVar = 0;
		//copy_cval_type(&withHalf, 0.5, TENSOR(apIn->quantum)->mxData->mxType);//���� ��� ������ 1/2�� ����
		registPhoto(ip, tp);
		listingApin(ip);
		listingApin(tp);//tpŸ���� ���������� ���� ���� �� ���ΰ� amse), fmse)���� fxt reference���� �ʰ� ����
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		Flux *fxr;
		sytet iv = invariance(apIn, apTar, fxr);
		if(iv <= 0) {
			//minusArv.paintVar = divArv.paintVar = mulArv.paintVar = dminusArv.paintVar = 0;
			intt bn = (apTcr->ibatchSize ? apTcr->ibatchSize : apIn->fshape[0]);
			if(bn >= apTcr->gradby) bn /= apTcr->gradby;
			//intt bn = MTX_SIZE(TENSOR(apIn->quantum)->mxData) / apTcr->gradby;
			//intt bn = (apTcr->ibatchSize ? apTcr->ibatchSize : MTX_SIZE(TENSOR(apIn->quantum)->mxData));
			copy_cval_type(mseVar.idxOne, bn, TENSOR(apIn->quantum)->mxData->mxType);//��ġ ������
			//copy_cval_type(&withBatch, bn, TENSOR(apIn->quantum)->mxData->mxType);//��ġ ������
			//squareMat = apTcr->instMatrix(squareMat, apIn->qType, apIn->fdim, apIn->fshape, false);
			if(withMean == 0 && iv == 0) apOut->resizing2(apIn, "mean square");
		}
		if(apTcr->pathPrint) printf("mean square fw\n");
		//TENSOR(apIn->quantum)->mxData->marith(tcxt, TENSOR(apTar->quantum)->mxData, squareMat,
		//	&minusArv, 0, nullx, nullx, AOP_MINUS);
		//squareMat->marith(tcxt, squareMat, squareMat, &mulArv, 0, nullx, nullx, AOP_MUL);
		//squareMat->msum(tcxt, TENSOR(apOut->quantum)->mxData, nullx, true);//&withHalf, true);
		//apIn->printo(2);
		//printf("------------------------------------\n");
		//apTar->printo(2);
		
		TENSOR(apIn->quantum)->mxData->mmean_square_e(tcxt, TENSOR(apTar->quantum)->mxData, TENSOR(apOut->quantum)->mxData, withMean);
		multiArrangeUnlock();
		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		if(apTcr->pathPrint) printf("mean square bw\n");
		//arrangeLock(apIn, bw_mutmx, lapInput, 0);
		//TENSOR(apIn->quantum)->mxData->marith(tcxt, TENSOR(apTar->quantum)->mxData, TENSOR(apIn->quantum)->mxGrad,
		//	&dminusArv, 0, nullx, nullx, AOP_MINUS);
		//TENSOR(apIn->quantum)->mxGrad->marith(tcxt, nullx, TENSOR(apIn->quantum)->mxGrad,
		//	&divArv, TENSOR(apIn->quantum)->mxData->mxType, &withBatch, nullx, AOP_DIV);//��ġ ������� ����.
		TENSOR(apIn->quantum)->mxData->mone(tcxt, TENSOR(apTar->quantum)->mxData, TENSOR(apIn->quantum)->mxGrad,
			2, &mseVar, AOP_ACTF, DLOSS_FUNC, PDC, 1);
		//arrangeUnlock(apIn, lapInput, 0);
		/*
		//arrangeLock(apTar, bw_mutmx, lapInput, 0);
		//amse.����
		//TENSOR(apTar->quantum)->mxData->marith(tcxt, TENSOR(apIn->quantum)->mxData, TENSOR(apTar->quantum)->mxGrad,
		//	&minusArv, 0, nullx, nullx, AOP_MINUS);
		//TENSOR(apTar->quantum)->mxGrad->marith(tcxt, nullx, TENSOR(apTar->quantum)->mxGrad,
		//	&divArv, TENSOR(apTar->quantum)->mxData->mxType, &withBatch, nullx, AOP_DIV);//��ġ ������� ����.
		TENSOR(apTar->quantum)->mxData->mone(tcxt, TENSOR(apIn->quantum)->mxData, TENSOR(apTar->quantum)->mxGrad,
			2, &mseVar, AOP_ACTF, DLOSS_FUNC, PDC, 1);
		arrangeUnlock(apTar, lapInput, 0);*/
		multiArrangeUnlock(0);
		return lapInput;
	}
};

class ApActf : public Apply {//���� ��꿡 ���ȴ�.
public:
	Flux *apOut, *apIn;
	intt opActf;
	OneVar ovar;
	//Matrixr *dactfMat;
	//ArithVar dactfArv;

	ApActf(Trace *tcr, Flux *ip, Flux *op, intt aop2, floatt alpha) : Apply(tcr)
	{
		apCode = APC_ACTF;
		apIn = ip;
		apOut = op;
		nfanIn = 1;
		nfanOut = 1;
		opActf = aop2;
		//dactfMat = nullx;
		//dactfArv.paintVar = 0;
		*(floatt *)ovar.idxOne = alpha;
		registPhoto(ip, nullx);
		listingApin(ip);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		sytet iv = invariance2(apIn);
		if(iv == 0) apOut->resizing2(apIn, "activation");
		//if(iv <= 0) {
		//	dactfArv.paintVar = 0;
		//	dactfMat = apTcr->instMatrix(dactfMat, apIn->qType, apIn->fdim, apIn->fshape, false);//���� �̺� ��� ���� ��Ʈ����
		//}
		if(apTcr->pathPrint) printf("activation fw: %d\n", opActf);
		
		TENSOR(apIn->quantum)->mxData->mone(tcxt, nullx, TENSOR(apOut->quantum)->mxData, 2, &ovar, AOP_ACTF, opActf, PDG, 0);
		multiArrangeUnlock();
		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		if(apTcr->pathPrint) printf("activation bw: %d\n", opActf);
		//TENSOR(apIn->quantum)->mxData->mone(tcxt, nullx, dactfMat, 2, nullx, AOP_ACTF, opActf +1, PDG, 1);
		//arrangeLock(apIn, bw_mutmx, lapInput, 0);
		//TENSOR(apOut->quantum)->mxGrad->marith(tcxt, dactfMat, TENSOR(apIn->quantum)->mxGrad, &dactfArv, 0, nullx, (void *)1, AOP_MUL);
		TENSOR(apIn->quantum)->mxData->mone(tcxt, TENSOR(apOut->quantum)->mxGrad, TENSOR(apIn->quantum)->mxGrad, 2, &ovar, AOP_ACTF, opActf + 1, PDG, 1);
		//arrangeUnlock(apIn, lapInput, 0);
		multiArrangeUnlock(0);
		return lapInput;
	}
};
class ApActf2 : public Apply {
public:
	Flux *apOut, *apPref, *apSuf;
	intt opActf;
	//Matrixr *dprefMat, *dsufMat;
	//ArithVar dprefArv, dsufArv;
	OneVar onev;

	ApActf2(Trace *tcr, Flux *fxp, Flux *fxs, Flux *op, intt aop2) : Apply(tcr)
	{
		apCode = APC_ACTF;
		apPref = fxp;
		apSuf = fxs;//a��
		apOut = op;
		nfanIn = 2;
		nfanOut = 1;
		opActf = aop2;
		//dprefMat = nullx;
		//dprefArv.paintVar = 0;

		registPhoto(fxp, fxs);
		listingApin(fxp);
		listingApin(fxs);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		sytet iv;
		if(opActf == ACTF_PRELU) {//����ġ(apSuf)�� ��ġ�� �����ϹǷ� ��ġ�� ���õǸ� �н��� ����� ��
			iv = invariance2(apPref);
			if(iv == 0) {
				apOut->resizing2(apPref, "activation2");
				floatt pv = apSuf->at_d(0);
				//���ö����� ����� �ƴϾ �׷����� ���� ��� �׶��忡 �ڵ� ���Ҵ���� �����Ƿ� ���⼭ ���� ���Ҵ�
				lapInput->ptrRight->freeShadow();//apSuf//�׶����� ���� ������ �Ҵ�ƴٸ� ��� �׶��忡 ���Ҵ��ϹǷ� ����
				CudaDevSet(DMATFX(apOut)->didground);
				apSuf->resizing5(apPref->scaleout, DMATFX(apOut)->didground);
				apSuf->fill(pv);
			}
		} else {
			Flux *fxr;
			iv = invariance(apPref, apSuf, fxr);
			if(iv == 0) apOut->resizing2(fxr, "activation2");
		}
		if(iv <= 0) {
			*(void **)onev.idxOne = TENSOR(apSuf->quantum)->mxData;
			*(void **)&onev.idxOne[2] = TENSOR(apSuf->quantum)->mxGrad;
			//dprefArv.paintVar = dsufArv.paintVar = 0;
			//dprefMat = apTcr->instMatrix(dprefMat, apPref->qType, apPref->fdim, apPref->fshape, false);
			//dsufMat = apTcr->instMatrix(dsufMat, apSuf->qType, apSuf->fdim, apSuf->fshape, false);
			//*(void **)onev.idxOne = dsufMat;
		}
		if(apTcr->pathPrint) printf("activation2 fw: %d\n", opActf);
		
		TENSOR(apPref->quantum)->mxData->mone(tcxt, TENSOR(apSuf->quantum)->mxData, TENSOR(apOut->quantum)->mxData, 2, &onev, AOP_ACTF2, opActf, PDG, 0);
		multiArrangeUnlock();
		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		if(apTcr->pathPrint) printf("activation2 bw: %d\n", opActf);
		//TENSOR(apPref->quantum)->mxData->mone(tcxt, TENSOR(apSuf->quantum)->mxData, dprefMat, 2, &onev, AOP_ACTF2, opActf + 1, PDG, 1);
		//arrangeLock(apPref, bw_mutmx, lapInput, 0);
		//TENSOR(apOut->quantum)->mxGrad->marith(tcxt, dprefMat, TENSOR(apPref->quantum)->mxGrad, &dprefArv, 0, nullx, (void *)1, AOP_MUL);
		//arrangeUnlock(apPref);
		//arrangeLock(apSuf, bw_mutmx, lapInput, 0);
		//TENSOR(apOut->quantum)->mxGrad->marith(tcxt, dsufMat, TENSOR(apSuf->quantum)->mxGrad, &dsufArv, 0, nullx, (void *)1, AOP_MUL);
		//arrangeUnlock(apSuf, lapInput, 0);
		//arrangeLock(apPref, bw_mutmx, lapInput, 0);//apSuf�� prelu������ ���Ǵ� prelu����ġ�̹Ƿ� �� �ʿ����.adar)��������
		TENSOR(apOut->quantum)->mxGrad->mone(tcxt, TENSOR(apPref->quantum)->mxData, TENSOR(apPref->quantum)->mxGrad, 2, &onev, AOP_ACTF2, opActf + 1, PDG, 1);
		//arrangeUnlock(apPref, lapInput, 0);
		multiArrangeUnlock(0);
		return lapInput;
	}
};
class ApOptimizer : public Apply {
public:
	Flux *apOut, *apIn;
	floatt rLning = 0.001;
	intt nepoch = 1;
	sytet optLoadStep;

	ApOptimizer(Trace *tcr, Flux *ip, Flux *op, floatt lr) : Apply(tcr)
	{
		apIn = ip;
		apOut = op;
		nfanIn = 1;
		nfanOut = 1;
		optLoadStep = 1;
		if(lr > 0) rLning = lr;

		registPhoto(ip, nullx);
		listingApin(ip);
	}
	Capsule *minimize(void)
	{
		apTcr->initVsync();
		apTcr->vSync(apOut);
		apTcr->initArrange();//�ʱ� �ѹ� �ʱ�ȭ �Ѵ�.feed�ʰ� ������ ��쿡�� arrange����ɼ��ֵ���
		apTcr->listWeight();//�� v sync�Ŀ� �ؾ���.apTcr->elistw(false, memput::mp::trainable);
		apTcr->bwVersion++;//���� run���࿡�� v sync�ǰ� �ϱ�����
		return apTcr->trainWeight;
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		if(apTcr->pathPrint) printf("optimizer fw\n");
		return nullx;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		if(apTcr->pathPrint) printf("optimizer bw\n");
		doublet cv = 1;
		((Tensor *)apIn->quantum)->mxGrad->fill(&cv, tdouble);
		return lapInput;
	}
	virtual void update(TCxtArray *tcxt_arr, Trace *trc) = 0;
};
typedef struct OptAdamParam_ {
	Flux *wgUpdate;
	Matrixr *movavg, *movavgv;
	struct OptAdamParam_ *ptrLeft, *ptrRight;
} OptAdamParam;
class ApAdamOpt : public ApOptimizer {
public:
	floatt beta1 = 0.9;
	floatt beta2 = 0.999;
	OptAdamParam *otparmt;
	ApAdamOpt(Trace *tcr, Flux *ip, Flux *op, floatt lr) : ApOptimizer(tcr, ip, op, lr)
	{
		apCode = APC_ADMOPT;
		otparmt = nullx;
	}
	void adamParam(Flux *wfx, intt gid)
	{
		OptAdamParam *optp = (OptAdamParam *)apTcr->xalloc(sizeof(OptAdamParam));
		optp->wgUpdate = wfx;//gid �׶��忡 �Ʒ� �ΰ� ��Ʈ���� �Ҵ� ������ adcg)���� �̸� Ȯ���Ǿ���.
		//������ �ǰ� �������� �ʴ� ����ġ�� adcg)�� ���⼭�� ���ܵǹǷ� ����Ȯ���� �������� ����.
		optp->movavg = apTcr->instMatrix(nullx, wfx->qType, wfx->fdim, wfx->fshape, false, gid);
		optp->movavgv = apTcr->instMatrix(nullx, wfx->qType, wfx->fdim, wfx->fshape, false, gid);
		optp->movavg->resetMemory(0);
		optp->movavgv->resetMemory(0);
		APPEND_LIST(otparmt, optp);
	}
	void minimize(vector<Flux *> *weight_list = nullptr)
	{
		Capsule *wlist = ((ApOptimizer *)this)->minimize();
		intt gid = apTcr->didTrace;

		if(weight_list) {
			for(auto wfx : *weight_list) {
				if(gid != GMATFX_GID(wfx)) {//����ġ�� ���� ��迡 �Ҵ�.
					gid = GMATFX_GID(wfx);//backwardDevArrange������ ����ġ�� ��� �׶���� ����ȴ�.
					CudaDevSet(gid);//��������� Ʈ���̽� ���� �����忡�� ��� arrange�ȴ�.
				}
				adamParam(wfx, gid);
			}
		} else {
			for(; wlist; wlist = wlist->ptrRight) {
				if(gid != GMATFX_GID(wlist->vcaps)) {//����ġ�� ���� ��迡 �Ҵ�.
					gid = GMATFX_GID(wlist->vcaps);
					CudaDevSet(gid);//��������� Ʈ���̽� ���� ������� ��� �����ȴ�.
				}
				adamParam(wlist->vcaps, gid);
			}
		}
		//if(gid != apTcr->didTrace) {//didTrace�� ���� trc�� ����Ʈ ��� ������ �����μ�
		//	CudaDevSet(apTcr->didTrace);//����Ǹ� �ȵǰ� �� ���̵� ���� ��� ��Ŀ���� �����.
		//}
		apTcr->didTrace = gid;
	}
	floatt lr_f(ApAdamOpt *opt, floatt alpha, intt epoch) {
		floatt fix1 = 1.0 - std::pow(opt->beta1, epoch);
		floatt fix2 = 1.0 - std::pow(opt->beta2, epoch);
		return alpha * std::sqrt(fix2) / fix1;
	}
	void update(TCxtArray *tcxt_arr, Trace *trc)
	{
		intt gid = trc->didTrace;
		OptAdamParam *optp;
		//printf("!!!!! ADAM OPT \n");
		for(optp = otparmt;optp; optp = optp->ptrRight) {
			if(gid != GMATFX_GID(optp->wgUpdate)) {//����ġ�� ���� ��迡�� �۾�.
				gid = GMATFX_GID(optp->wgUpdate);
				CudaDevSet(gid);//Ʈ���̽��� run�Լ� ���� ������� ��� �����ȴ�.
			}
			DMATFX_SET_GROUND(optp->wgUpdate, gid);//�����İ������� �������� �����찡 ��Ŀ�� �������������Ƿ� �׶��� ��Ŀ���Ѵ�.
			if(optp->wgUpdate->fxSize != optp->movavg->maxmSize) {//prelu�� ��� ����ġ���� ��ġ ����� �����ϹǷ� ������ ����Ǹ� �ݿ��Ѵ�.
				optp->movavg = apTcr->instMatrix(optp->movavg, optp->wgUpdate->qType, optp->wgUpdate->fdim, optp->wgUpdate->fshape, false, gid);
				optp->movavgv = apTcr->instMatrix(optp->movavgv, optp->wgUpdate->qType, optp->wgUpdate->fdim, optp->wgUpdate->fshape, false, gid);
				optp->movavg->resetMemory(0);//gid �׶��忡 �� �ΰ� ��Ʈ���� �Ҵ� ������ adcg)���� �̸� Ȯ���Ǿ���.
				optp->movavgv->resetMemory(0);//������ �ǰ� �������� �ʴ� ����ġ�� adcg)�� ���⼭�� ���ܵǹǷ� ����Ȯ���� �������� ����.
			}
			//if(!strcmp("kkk_0-ln_gamma", optp->wgUpdate->fxName))
			//	optp->wgUpdate->printo();
			optp->movavg->moptadm(tcxt_arr->getgputxt(gid, trc), optp->movavgv,
				GMATFX(optp->wgUpdate),	DMATFX(optp->wgUpdate), beta1, beta2,
				lr_f(this, rLning, nepoch), 1e-8, -1);
			//printf("sss: %p %s\n", optp->wgUpdate, optp->wgUpdate->fxName);
			//TENSOR(optp->wgUpdate->quantum)->mxData->printo();
		}
		trc->didTrace = gid;
		nepoch++;
		//printf("!!!!! ADAM OPT end \n");
	}
};
class ApSgdOpt : public ApOptimizer {
public:
	Capsule *otpsrmt;
	ApSgdOpt(Trace *tcr, Flux *ip, Flux *op, floatt lr) : ApOptimizer(tcr, ip, op, lr)
	{
		apCode = APC_SGDOPT;
		otpsrmt = nullx;
	}
	void minimize(vector<Flux *> *weight_list = nullptr)
	{
		Capsule *wlist = ((ApOptimizer *)this)->minimize(), *cap;

		if(weight_list) {
			for(auto wfx : *weight_list) {
				cap = (Capsule *)apTcr->xalloc(sizeof(Capsule));
				cap->vcaps = wfx;
				APPEND_LIST(otpsrmt, cap);
			}
		} else {
			for(;wlist; wlist = wlist->ptrRight) {
				cap = (Capsule *)apTcr->xalloc(sizeof(Capsule));
				cap->vcaps = wlist->vcaps;
				APPEND_LIST(otpsrmt, cap);
			}
		}
	}
	void update(TCxtArray *tcxt_arr, Trace *trc)
	{
		intt gid = trc->didTrace;
		Capsule *optp;
		unit lap = (apTcr->lapType == 1 ? xucurrenttime() : 0);
		//printf("!!!!! SGD OPT \n");
		for(optp = otpsrmt;optp; optp = optp->ptrRight) {
			//printf("%s\n", optp->vcaps->fxName);
			//optp->vcaps->printo();
			if(gid != GMATFX_GID(optp->vcaps)) {//����ġ�� ���� ��迡�� �۾�.
				gid = GMATFX_GID(optp->vcaps);
				CudaDevSet(gid);//Ʈ���̽��� run�Լ� ���� ������� ��� �����ȴ�.
			}
			DMATFX_SET_GROUND(optp->vcaps, gid);//�����İ������� �������� �����찡 ��Ŀ�� �������������Ƿ� �׶��� ��Ŀ���Ѵ�.
			GMATFX(optp->vcaps)->moptsgd(tcxt_arr->getgputxt(gid, trc),
				DMATFX(optp->vcaps), rLning, -1);
		}
		trc->didTrace = gid;
		if(apTcr->lapType == 2) printf("sgd lap: %lld\n", xucurrenttime() - lap);
	}
};
class ApEmbedding : public Apply {
public:
	Flux *apPrefix, *apSuffix, *apOut;

	ApEmbedding(Trace *tcr, Flux *fxp, Flux *fxs, Flux *fxo) : Apply(tcr)
	{
		apCode = APC_EMBEDDING;
		apPrefix = fxp;//�ӹ�� ���̺�
		apSuffix = fxs;
		apOut = fxo;
		nfanIn = 2;
		nfanOut = 1;
		loadOnExec = 1;
		registPhoto(apPrefix, apSuffix);
		listingApin(apPrefix);
		listingApin(apSuffix);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		Flux *fxr;
		invariance3(apPrefix, true);
		if(invariance(nullx, apSuffix, fxr) == 0) apOut->resizing2(fxr, "embedding");
		if(apTcr->pathPrint) printf("embedding fw\n");
		
		TENSOR(apPrefix->quantum)->mxData->mone(tcxt, TENSOR(apSuffix->quantum)->mxData, TENSOR(apOut->quantum)->mxData, 2, nullx, AOP_EMBEDDING, 0, PDC, 0);
		multiArrangeUnlock();
		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx) //suffix�� �����Ŀ� ���� ����ġ ������ ���� �ʴ´�. suffix�� Ʈ���̴� ����� �ƴ�.
	{
		if(apTcr->pathPrint) printf("embedding bw\n");
		multiArrangeLock(0);
		TENSOR(apOut->quantum)->mxGrad->mone(tcxt, TENSOR(apSuffix->quantum)->mxData, TENSOR(apPrefix->quantum)->mxGrad, 0, nullx, AOP_EMBEDDING, 1, PDC, -1);
		multiArrangeUnlock(0);
		return lapInput;
	}
};
class ApOneHot : public Apply {
public:
	Flux *apPrefix, *apOut;
	sytet verginF;
	intt axisoh, depthoh;
	doublet onvoh, offvoh;
	OneVar onev;

	void setohv(doublet on_value, doublet off_value, intt axis, intt depth)
	{
		*(doublet *)onev.idxOne = on_value;
		*(doublet *)&onev.idxOne[2] = off_value;
		if(axis == 0) onev.idxOne[4] = apPrefix->fxSize;
		else onev.idxOne[4] = FX_SIZE_RK4(apPrefix, (axis - 1));
		onev.idxOne[5] = depth;
	}
	ApOneHot(Trace *tcr, Flux *fxp, Flux *fxo, doublet on_value, doublet off_value, intt axis, intt depth) : Apply(tcr)
	{
		apCode = APC_ONEHOT;
		apPrefix = fxp;
		apOut = fxo;
		nfanIn = 1;
		nfanOut = 1;
		onvoh = on_value;
		offvoh = off_value;
		axisoh = axis;
		depthoh = depth;
		verginF = 1;

		registPhoto(apPrefix, nullx);
		listingApin(apPrefix);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		sytet iv = invariance2(apPrefix);
		if(iv == 0) {
			if(axisoh == 0) throwFault(-1, "first variable rank encoding error\n");
			apOut->resizing2(apPrefix, "onehot");
		}
		if(iv <= 0 || verginF) {
			verginF = 0;
			setohv(onvoh, offvoh, axisoh, depthoh);
		}
		if(apTcr->pathPrint) printf("onehot fw\n");
		apOut->fill(*(doublet *)&onev.idxOne[2]);//off value���� �ϰ� ����, ���⼭ off value�� ä���� ��� ��Ʈ������
		//�Ʒ� �Լ����� gpu����ɶ� ȣ��Ʈ���� ����̽� �޸𸮷� �����ϱ����� ���� rplus�� -1�� ȣ���Ѵ�.(cpu�����̸� ����ȵ�)
		
		TENSOR(apPrefix->quantum)->mxData->mone(tcxt, nullx, TENSOR(apOut->quantum)->mxData, 0, &onev, AOP_ONEHOT, 0, PDC, -1);
		multiArrangeUnlock();
		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		if(apTcr->pathPrint) printf("onehot bw\n");
		return nullx;
	}
};
class ApVMax : public Apply {
public:
	Flux *apPrefix, *apOut;
	sytet verginF;
	intt axisamax;
	OneVar onev;
	Matrixr *margmax;

	void setvmaxv(void)
	{
		onev.nrkPre = apPrefix->fshape[axisamax];

		if(axisamax == 0) onev.nrkOut = apPrefix->fxSize;
		else onev.nrkOut = FX_SIZE_RK(apPrefix, axisamax);//outer size, axis super size

		onev.nrkSuf = FX_SIZE_RK3(apPrefix, axisamax);//inner size, axis rank size
		if(onev.nrkSuf == 0) onev.nrkSuf = onev.nrkOut;
	}
	ApVMax(Trace *tcr, Flux *fxp, Flux *fxo, intt axis) : Apply(tcr)
	{
		apCode = APC_VMAX;
		apPrefix = fxp;
		apOut = fxo;
		nfanIn = 1;
		nfanOut = 1;
		axisamax = axis;
		verginF = 1;
		margmax = nullx;

		registPhoto(apPrefix, nullx);
		listingApin(apPrefix);

		margmax = apTcr->instMatrix(margmax, apOut->qType, apOut->fdim, apOut->fshape, false, JUST_SIZEDEF);
	}
	intt mcalcapp(intt n) //��ġ�� n���� ������ ���ö��� ���� ��� �޸� ���, �̸����� �̵� ��Ʈ������
	{						//������ �޸� �Ҵ� ���� ������ ��길 �Ҽ��ְ� ����
		return margmax->sizem2(n);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		sytet iv = invariance2(apPrefix);
		if(iv == 0) {
			apOut->resizing2(apPrefix, "resize_fhs");
		}
		if(iv <= 0 || verginF) {
			verginF = 0;
			setvmaxv();
			if(iv <= 0) {
				intt gid = DMATFX(apOut)->didground;//gid �׶��忡���� �Ҵ� ������ �� mcalc app���� ���Ǿ� Ȯ���Ǿ���.
				margmax = apTcr->instMatrix(margmax, apOut->qType, apOut->fdim, apOut->fshape, false, gid);
			}
		}
		if(apTcr->pathPrint) printf("vmax fw\n");

		TENSOR(apPrefix->quantum)->mxData->mone(tcxt, margmax, TENSOR(apOut->quantum)->mxData, 2, &onev, AOP_VMAX, 0, PDC, 0);
		multiArrangeUnlock();
		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		//margmax->printo();
		//printf("\n-------------------\n");
		//TENSOR(apOut->quantum)->mxGrad->printo();
		//printf("\n-------------------\n");
		margmax->mone(tcxt, TENSOR(apOut->quantum)->mxGrad, TENSOR(apPrefix->quantum)->mxGrad, 2, &onev, AOP_VMAX, 1, PDG, 1);
		//arrangeUnlock(apIn, lapInput, 0);
		//printf("\n-------------------\n");
		//TENSOR(apPrefix->quantum)->mxGrad->printo();
		multiArrangeUnlock(0);
		return lapInput;
	}
};
class ApArgmax : public Apply {
public:
	Flux *apPrefix, *apOut;
	sytet verginF;
	intt axisamax;
	OneVar onev;

	void setamaxv(void)
	{
		onev.nrkPre = apPrefix->fshape[axisamax];
		
		//if(axisamax == 0) onev.nrkOut = apPrefix->fxSize;
		//else onev.nrkOut = FX_SIZE_RK3(apPrefix, (axisamax -1));//outer size, axis super size

		if(axisamax == 0) onev.nrkOut = apPrefix->fxSize;
		else onev.nrkOut = FX_SIZE_RK(apPrefix, axisamax);//outer size, axis super size

		onev.nrkSuf = FX_SIZE_RK3(apPrefix, axisamax);//inner size, axis rank size
		if(onev.nrkSuf == 0) onev.nrkSuf = onev.nrkOut;
	}
	ApArgmax(Trace *tcr, Flux *fxp, Flux *fxo, intt axis) : Apply(tcr)
	{
		apCode = APC_ARGMAX;
		apPrefix = fxp;
		apOut = fxo;
		nfanIn = 1;
		nfanOut = 1;
		axisamax = axis;
		verginF = 1;

		registPhoto(apPrefix, nullx);
		listingApin(apPrefix);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		sytet iv = invariance2(apPrefix);
		if(iv == 0) {
			apOut->resizing2(apPrefix, "resize_fhs");
		}
		if(iv <= 0 || verginF) {
			verginF = 0;
			setamaxv();
		}
		if(apTcr->pathPrint) printf("argmax fw\n");
		
		TENSOR(apPrefix->quantum)->mxData->mone(tcxt, nullx, TENSOR(apOut->quantum)->mxData, 2, &onev, AOP_ARGMAX, 0, PDC, 0);
		multiArrangeUnlock();
		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		if(apTcr->pathPrint) printf("argmax bw\n");
		return nullx;
	}
};
class ApSlice : public Apply {
public:
	Flux *apPrefix, *apOut;
	intt codeSlice[MX_DIM * 3];
	sytet verginF;
	OneVar onev;

	void setSlicev(void)
	{
		intt bslice[MX_DIM * 3];

		onev.nrkPre = apPrefix->fdim;
		memcpy(onev.rankPre, TENSOR(apPrefix->quantum)->mxData->mxranksz, onev.nrkPre * sizeof(intt));
		onev.nrkOut = apOut->fdim;
		memcpy(onev.rankOut, TENSOR(apOut->quantum)->mxData->mxranksz, onev.nrkOut * sizeof(intt));
		apPrefix->boundSlice(apPrefix->fdim * 3, codeSlice, bslice, true);

		intt n = apOut->fdim, i = 0, j = 0, v;
		SliceRank *slicer = (SliceRank *)onev.idxOne;
		for(;i < n; i++, j += 3) {
			slicer[i].sldim = apOut->fshape[i];
			v = FX_SIZE_RK3(apPrefix, i);
			slicer[i].slbase = v * bslice[j];
			slicer[i].slsz = v * bslice[j + 2];
			slicer[i].sltsz = slicer[i].slsz * (slicer[i].sldim - 1);
		}
	}
	ApSlice(Trace *tcr, Flux *fxp, Flux *fxo, intt code[]) : Apply(tcr)
	{
		apCode = APC_SLICE;
		apPrefix = fxp;
		apOut = fxo;
		nfanIn = 1;
		nfanOut = 1;
		memcpy(codeSlice, code, fxp->fdim * 3 * sizeof(intt));

		verginF = 1;

		registPhoto(apPrefix, nullx);
		listingApin(apPrefix);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		sytet iv = invariance2(apPrefix);
		if(iv == 0) {
			intt slicer_idx[3];
			apPrefix->boundSlice(3, codeSlice, slicer_idx, false);
			intt nfirst = 1 + (slicer_idx[1] - slicer_idx[0]) / slicer_idx[2];
			//apOut->resizing4(nfirst);//ù��° ��ũ ����� ����Ͽ� ������¡
			apOut->resizing2(apPrefix, "slice");
		}
		if(iv <= 0 || verginF) {
			verginF = 0;
			setSlicev();
		}
		if(apTcr->pathPrint) printf("slice fw\n");
		
		TENSOR(apPrefix->quantum)->mxData->mone(tcxt, nullx, TENSOR(apOut->quantum)->mxData, 2, &onev, AOP_SLICE, 0, PDC, 0);
		multiArrangeUnlock();
		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		//printf("zzz %p\n", apPrefix);
		if(apTcr->pathPrint) printf("slice bw\n");
		//arrangeLock(apPrefix, bw_mutmx, lapInput, 0);
		TENSOR(apOut->quantum)->mxGrad->mone(tcxt, nullx, TENSOR(apPrefix->quantum)->mxGrad, 0, &onev, AOP_SLICE, 1, PDC, 1);
		//arrangeUnlock(apPrefix, lapInput, 0);
		multiArrangeUnlock(0);
		return lapInput;
	}
};
class ApOne : public Apply {
public:
	Flux *apPrefix, *apSuffix, *apOut;
	intt aopPrime, aopSecondary, axisResize;
	sytet verginF, iszFw, iszBw, prsDiv, rpusApo;
	bool oneIndex;
	OneVar onevar;

	void setone(void)
	{
		if(oneIndex == 0) return;
		onevar.nrkPre = apPrefix->fdim;
		memcpy(onevar.rankPre, TENSOR(apPrefix->quantum)->mxData->mxranksz, onevar.nrkPre * sizeof(intt));
		memcpy(onevar.dimPre, apPrefix->fshape, onevar.nrkPre * sizeof(intt));
		if(apSuffix) {
			onevar.nrkSuf = apSuffix->fdim;
			memcpy(onevar.rankSuf, TENSOR(apSuffix->quantum)->mxData->mxranksz, onevar.nrkSuf * sizeof(intt));
			memcpy(onevar.dimSuf, apSuffix->fshape, onevar.nrkSuf * sizeof(intt));
		}
		onevar.nrkOut = apOut->fdim;
		memcpy(onevar.rankOut, TENSOR(apOut->quantum)->mxData->mxranksz, onevar.nrkOut * sizeof(intt));
		memcpy(onevar.dimOut, apOut->fshape, onevar.nrkOut * sizeof(intt));
	}
	ApOne(Trace *tcr, Flux *fxp, Flux *fxs, Flux *fxo, OneVar *onev, bool variance_check, sytet isz_fw, sytet isz_bw,
		intt aop, intt aop2, intt axis, bool indexing, sytet pdiv, sytet rplus) : Apply(tcr)
	{
		apCode = APC_ONE;
		apPrefix = fxp;
		apSuffix = fxs;
		apOut = fxo;
		if(fxs) nfanIn = 2;
		else nfanIn = 1;
		nfanOut = 1;
		memcpy(&onevar, onev, sizeof(OneVar));
		aopPrime = aop;
		aopSecondary = aop2;
		iszFw = isz_fw;
		iszBw = isz_bw;
		axisResize = axis;
		oneIndex = indexing;
		prsDiv = pdiv;
		rpusApo = rplus;
		if(variance_check == false) verginF = -1;
		else verginF = 1;

		registPhoto(apPrefix, apSuffix);
		listingApin(apPrefix);
		listingApin(apSuffix);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		Flux *fxr;
		sytet iv = invariance(apPrefix, apSuffix, fxr);
		if(iv == 0) {
			if(axisResize == 0) throwFault(-1, "first variable rank resize error\n");
			apOut->resizing2(fxr, "one");
		}
		if(verginF >= 0 && (iv <= 0 || verginF)) {
			verginF = 0;
			setone();
		}
		if(apTcr->pathPrint) printf("one fw: %d %d\n", aopPrime, aopSecondary);
		
		TENSOR(apPrefix->quantum)->mxData->mone(tcxt, apSuffix ? TENSOR(apSuffix->quantum)->mxData : nullx,
			TENSOR(apOut->quantum)->mxData, iszFw, &onevar, aopPrime, aopSecondary, prsDiv, 0);
		multiArrangeUnlock();
		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		if(iszBw < 0) return nullx;
		if(apTcr->pathPrint) printf("one bw: %d %d\n", aopPrime, aopSecondary);
		multiArrangeLock(0);
		TENSOR(apOut->quantum)->mxGrad->mone(tcxt, apSuffix ? TENSOR(apSuffix->quantum)->mxData : nullx,
			TENSOR(apPrefix->quantum)->mxGrad, iszBw, &onevar, aopPrime, aopSecondary, prsDiv, rpusApo);
		multiArrangeUnlock(0);
		return lapInput;
	}
};
class ApTwo : public Apply {
public:
	Flux *apPrefix, *apSuffix, *apOut;
	intt aopPrime, aopSub;

	ApTwo(Trace *tcr, Flux *fxp, Flux *fxs, Flux *fxo, intt aop, intt aop2) : Apply(tcr)
	{
		apCode = APC_TWO;
		apPrefix = fxp;
		apSuffix = fxs;
		apOut = fxo;
		if(fxs) nfanIn = 2;
		else nfanIn = 1;
		nfanOut = 1;
		aopPrime = aop;
		aopSub = aop2;

		registPhoto(apPrefix, apSuffix);
		listingApin(apPrefix);
		listingApin(apSuffix);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		if(apTcr->pathPrint) printf("two fw: %d %d\n", aopPrime, aopSub);
		//if(aopSub == TWOF_SQDIFF) {
		//	apPrefix->printo(2);
		//	apSuffix->printo(2);
		//}
		multiArrangeLock();
		TENSOR(apPrefix->quantum)->mxData->mtwo(tcxt, TENSOR(apSuffix->quantum)->mxData, TENSOR(apOut->quantum)->mxData,
			nullx, nullx, aopPrime, aopSub);
		multiArrangeUnlock();
		//if(aopSub == TWOF_SQDIFF) apOut->printo();
		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		if(apTcr->pathPrint) printf("two bw: %d %d\n", aopPrime, aopSub);
		multiArrangeLock(0);
		TENSOR(apPrefix->quantum)->mxGrad->mtwo(tcxt, TENSOR(apSuffix->quantum)->mxGrad, TENSOR(apOut->quantum)->mxGrad,
			TENSOR(apPrefix->quantum)->mxData, TENSOR(apSuffix->quantum)->mxData, aopPrime, aopSub + 1);
		multiArrangeUnlock(0);
		return lapInput;
	}
};
class ApOverwrite : public Apply {
public:
	Flux *apPrefix, *apSuffix, *apOut;
	intt aopPrime, aopSub;

	ApOverwrite(Trace *tcr, Flux *fxp, Flux *fxs, Flux *fxo) : Apply(tcr)
	{
		apCode = APC_OVERWRITE;
		apPrefix = fxp;
		apSuffix = fxs;
		apOut = fxo;
		nfanIn = 2;
		nfanOut = 1;

		registPhoto(apPrefix, nullx);
		listingApin(apPrefix);
		listingApin(apSuffix);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		if(apTcr->pathPrint) printf("ap overwrite\n");
		sytet iv = invariance2(apPrefix);
		
		if(iv == 0) apSuffix->resizing2(apPrefix, "overwrite");//apSuffix�� ��ġ ��������Ǿ� 
													//ȣ��Ǵ� ���� �⺻�̹Ƿ� �ʿ������ �׳� �Ѵ�.
		apSuffix->copyf(apPrefix);
		multiArrangeUnlock();

		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		return lapInput;//���� �����Ĵ� ���� �ʰ� ������ ī��Ʈ�� �ϵ��� ��ǲ ����Ʈ�� �����Ѵ�.
	}
};
class ApCombination : public Apply {
public:
	Flux *apOut, *apIn;
	intt szKernelCb, nStrideCb;
	doublet rContigusCb;
	sytet zapdCb;
	bool onebnCb;
	ApCombination(Trace *tcr, Flux *ip, intt width, intt stride, doublet exc_contig_r, sytet zero_pading, bool one_batch) : Apply(tcr)
	{
		apCode = APC_COMB;
		apIn = ip;
		nfanIn = 1;
		nfanOut = 1;
		szKernelCb = width;
		nStrideCb = stride;
		rContigusCb = exc_contig_r;
		zapdCb = zero_pading;
		onebnCb = one_batch;
		apOut = rsc::rsc_combination(ip, width, stride, exc_contig_r, zero_pading, -1, one_batch);
		registPhoto(ip, apOut);
		listingApin(ip);
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		if(invariance2(apIn) == 0) apOut->resizing2(apIn, "combination");
		if(apTcr->pathPrint) printf("combination fw\n");
		rsc::rsc_combination2(apIn, apOut, szKernelCb, nStrideCb, rContigusCb, zapdCb, 0, onebnCb);
		//apOut->printo(1, 2);
		//apOut->shape();
		Univ uv(COPY_H2D_OP, 0);
		TENSOR(apOut->quantum)->mxData->uniform(tcxt, &uv);
		multiArrangeUnlock();

		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		if(apTcr->pathPrint) printf("combination bw\n");
		return lapInput;
	}
};
class ApLayerNormal : public Apply {
public:
	Flux *apOut, *apIn, *apGamma, *apBeta;
	Matrixr *md, *mz, *mv, *mean, *var, *g_mz, *mdmean;

	ApLayerNormal(Trace *tcr, Flux *ip, Flux *op, Flux *gamma, Flux *beta) : Apply(tcr)
	{
		apCode = APC_LAYER_N;
		apIn = ip;
		apOut = op;
		nfanIn = 3;
		nfanOut = 1;
		apCode = APC_LAYN;
		apGamma = gamma;
		apBeta = beta;
		md = mz = mv = mean = g_mz = var = mdmean = nullx;
		registPhoto(ip, nullx);
		listingApin(ip);
		listingApin(gamma);//�� ap�������� ���Ǵ� ���� ����ġ�̹Ƿ� adar)���� Ÿ�� �׶��忡 ���Ҵ� �Ǿ� �����찡
		listingApin(beta);//�ʿ�����Ƿ� ip�� �����ϸ� �ȴ�. �������� ���� �����Ƿ� multiArrangeLock�� ���� ���࿡�� ���õ�.
		
		md = apTcr->instMatrix(md, apIn->qType, apIn->fdim, apIn->fshape, false, JUST_SIZEDEF);
		mz = apTcr->instMatrix(mz, apIn->qType, apIn->fdim, apIn->fshape, false, JUST_SIZEDEF);
		mv = apTcr->instMatrix(mv, apIn->qType, apIn->fdim - 1, apIn->fshape, false, JUST_SIZEDEF);
		mean = apTcr->instMatrix(mean, apIn->qType, apIn->fdim - 1, apIn->fshape, false, JUST_SIZEDEF);
		g_mz = apTcr->instMatrix(g_mz, apIn->qType, apIn->fdim, apIn->fshape, false, JUST_SIZEDEF);
		var = apTcr->instMatrix(var, apIn->qType, apIn->fdim - 1, apIn->fshape, false, JUST_SIZEDEF);
		mdmean = apTcr->instMatrix(mdmean, apIn->qType, apIn->fdim - 1, apIn->fshape, false, JUST_SIZEDEF);
	}
	intt mcalcapp(intt n) //��ġ�� n���� ������ ���ö��� ���� ��� �޸� ���, �̸����� �̵� ��Ʈ������
	{						//������ �޸� �Ҵ� ���� ������ ��길 �Ҽ��ְ� ����
		intt sz = 0;
		sz += md->sizem2(n);
		sz += mz->sizem2(n);
		sz += mv->sizem2(n);
		sz += mean->sizem2(n);
		sz += g_mz->sizem2(n);
		sz += var->sizem2(n);
		sz += mdmean->sizem2(n);
		return sz;
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		sytet iv = invariance2(apIn);
		if(iv == 0) apOut->resizing2(apIn, "nlayer normal");
		if(iv <= 0) {
			intt gid = DMATFX(apOut)->didground;//gid �׶��忡���� �Ҵ� ������ �� mcalc app���� ���Ǿ� Ȯ���Ǿ���.
			md = apTcr->instMatrix(md, apIn->qType, apIn->fdim, apIn->fshape, false, gid);
			mz = apTcr->instMatrix(mz, apIn->qType, apIn->fdim, apIn->fshape, false, gid);
			mv = apTcr->instMatrix(mv, apIn->qType, apIn->fdim - 1, apIn->fshape, false, gid);
			mean = apTcr->instMatrix(mean, apIn->qType, apIn->fdim - 1, apIn->fshape, false, gid);
			g_mz = apTcr->instMatrix(g_mz, apIn->qType, apIn->fdim, apIn->fshape, false, gid);
			var = apTcr->instMatrix(var, apIn->qType, apIn->fdim - 1, apIn->fshape, false, gid);
			mdmean = apTcr->instMatrix(mdmean, apIn->qType, apIn->fdim - 1, apIn->fshape, false, gid);
		}
		//LOCK_MUT_(rsc::srutra->mutrt_dbg);
		if(apTcr->pathPrint) printf("\nlayer normal fw\n");
		//TENSOR(apIn->quantum)->mxData->printo();
		
		TENSOR(apIn->quantum)->mxData->mlayer_norm(tcxt, TENSOR(apOut->quantum)->mxData, md, mz, mv, mean, nullx,
			nullx, nullx, TENSOR(apGamma->quantum)->mxData, TENSOR(apBeta->quantum)->mxData, nullx, nullx, 0);
		multiArrangeUnlock();
		//printf("\n V=========================================V\n");
		//TENSOR(apOut->quantum)->mxData->printo();
		//printf("\n V=========================================V\n");
		//UNLOCK_MUT_(rsc::srutra->mutrt_dbg);

		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		if(apTcr->pathPrint) printf("layer normal bw\n");
		//TENSOR(apOut->quantum)->mxGrad->printo();
		//arrangeLock(apIn, bw_mutmx, lapInput, 0);
		TENSOR(apOut->quantum)->mxGrad->mlayer_norm(tcxt, TENSOR(apIn->quantum)->mxGrad, md, mz, mv, mean, g_mz,
			var, mdmean, TENSOR(apGamma->quantum)->mxData, nullx,
			TENSOR(apGamma->quantum)->mxGrad, TENSOR(apBeta->quantum)->mxGrad, 1);
		//arrangeUnlock(apIn, lapInput, 0);
		//printf("\n V=========================================V\n");
		//TENSOR(apIn->quantum)->mxGrad->printo();
		//TENSOR(apGamma->quantum)->mxGrad->printo();
		multiArrangeUnlock(0);
		return lapInput;
	}
};

class ApMatmul : public Apply {
public:
	Flux *apPre, *apSuf, *apOut;
	intt conjuDim;
	sytet transOrd;

	static intt matmul_out_shape(Flux *fxp, Flux *fxs, intt axid[MX_DIM], intt trans_order, intt &ja)
	{
		intt ndim, i, to_dim[2], jb;

		if(fxp->fdim != fxs->fdim) throwFault(-1, "batch dimention inconsistant\n");
		ndim = fxp->fdim - 2;
		for(i = 0;i < ndim; i++) {
			if(fxp->fshape[i] != fxs->fshape[i]) throwFault(-1, "batch dimention inconsistant %d %d\n", fxp->fshape[i], fxs->fshape[i]);
			axid[i] = fxs->fshape[i];
		}
		switch(trans_order) {
		case TON:
			to_dim[0] = fxp->fshape[i];
			to_dim[1] = fxs->fshape[i + 1];
			ja = fxp->fshape[i + 1];
			jb = fxs->fshape[i];
			break;
		case TOA:
			to_dim[0] = fxp->fshape[i + 1];
			to_dim[1] = fxs->fshape[i + 1];
			ja = fxp->fshape[i];
			jb = fxs->fshape[i];
			break;
		case TOB:
			to_dim[0] = fxp->fshape[i];
			to_dim[1] = fxs->fshape[i];
			ja = fxp->fshape[i + 1];
			jb = fxs->fshape[i + 1];
			break;
		case TOT:
			to_dim[0] = fxp->fshape[i + 1];
			to_dim[1] = fxs->fshape[i];
			ja = fxp->fshape[i];
			jb = fxs->fshape[i + 1];
		}
		axid[i++] = to_dim[0];
		axid[i] = to_dim[1];
		if(ja != jb) throwFault(-1, "matrix mul axis not equal\n");
		return fxp->fdim;
	}
	ApMatmul(Trace *tcr, Flux *fxp, Flux *fxs, Flux *fxo, intt jo, sytet trans_order) : Apply(tcr)
	{
		apPre = fxp;
		apSuf = fxs;
		apOut = fxo;
		transOrd = trans_order;
		apCode = APC_MATM;
		nfanIn = 2;
		nfanOut = 1;
		registPhoto(fxp, fxs);
		listingApin(fxp);
		listingApin(fxs);
		conjuDim = jo;
	}
#define LOW_DIM(fx) fx->fshape[fx->fdim - 2]
#define COL_DIM(fx) fx->fshape[fx->fdim - 1]
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		Flux *fxr;
		sytet iv = invariance(apPre, apSuf, fxr);
		if(iv == 0) {
			if(fxr->fdim <= 2) throwFault(-1, "only batch dimension changable\n");
			apOut->resizing2(fxr, "matmul");//�Ʒ� �Լ��� ��ġ ������ �ƴ� ������� ������¡�Ҽ������� ���⼭ ��ġ���� ������
			//���°� �ٲ�� �Ŀ� dot�� ���� ������ ����� ��� ���⼭ ������ �÷����� �ԷƵǸ� �����߻��Ѵ�. ���߿� ��� 
			//������ ���� ���� �����ϵ��� �����Ѵٸ� �Ʒ� �Լ��� �츰��.
			//apOut->fdim = ApMatmul::matmul_out_shape(apPre, apSuf, apOut->fshape, transOrd, conjuDim);
			//apOut->instTens(true);//resize
		}
		if(apTcr->pathPrint) printf("matmul fw\n");
		OneVar onevar;
		onevar.idxOne[0] = LOW_DIM(apOut);//M
		onevar.idxOne[1] = conjuDim;//K
		onevar.idxOne[2] = COL_DIM(apOut);//N
		onevar.idxOne[3] = transOrd;
		
		TENSOR(apPre->quantum)->mxData->mone(tcxt, TENSOR(apSuf->quantum)->mxData, 
			TENSOR(apOut->quantum)->mxData, 2, &onevar, AOP_MATMUL, 0, PDG, 0);
		multiArrangeUnlock();
		return apOut;
	}
	//[4, 3] * [3, 2] => [4, 2] ������ ����
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		OneVar onevar;
		multiArrangeLock(0);
		if(apTcr->pathPrint) printf("matmul bw\n");
		switch(transOrd) {
		case TON://[4, 3] * [3, 2] => [4, 2]
			onevar.idxOne[0] = LOW_DIM(apOut);//M
			onevar.idxOne[1] = COL_DIM(apOut);//K
			onevar.idxOne[2] = LOW_DIM(apSuf);//N
			onevar.idxOne[3] = TOB;//[4, 2] * [2, 3]^ => [4, 3]
			//arrangeLock(apPre, bw_mutmx, lapInput, 0);
			TENSOR(apOut->quantum)->mxGrad->mone(tcxt, TENSOR(apSuf->quantum)->mxData,
				TENSOR(apPre->quantum)->mxGrad, 2, &onevar, AOP_MATMUL, 0, PDG, 1);
			//arrangeUnlock(apPre, lapInput, 0);
			onevar.idxOne[0] = COL_DIM(apPre);
			onevar.idxOne[1] = LOW_DIM(apOut);
			onevar.idxOne[2] = COL_DIM(apOut);
			onevar.idxOne[3] = TOA;//[3, 4]^ * [4, 2] => [3, 2]
			//arrangeLock(apSuf, bw_mutmx, lapInput->ptrRight, 0);
			TENSOR(apPre->quantum)->mxData->mone(tcxt, TENSOR(apOut->quantum)->mxGrad,
				TENSOR(apSuf->quantum)->mxGrad, 2, &onevar, AOP_MATMUL, 0, PDG, 1);
			//arrangeUnlock(apSuf, lapInput, 0);
			break;
		case TOA://[3, 4]^ * [3, 2] => [4, 2]
			onevar.idxOne[0] = LOW_DIM(apSuf);//M
			onevar.idxOne[1] = COL_DIM(apSuf);//K
			onevar.idxOne[2] = LOW_DIM(apOut);//N
			onevar.idxOne[3] = TOB;//[3, 2] * [4, 2]^ => [3, 4]
			//arrangeLock(apPre, bw_mutmx, lapInput, 0);
			TENSOR(apSuf->quantum)->mxData->mone(tcxt, TENSOR(apOut->quantum)->mxGrad,
				TENSOR(apPre->quantum)->mxGrad, 2, &onevar, AOP_MATMUL, 0, PDG, 1);
			//arrangeUnlock(apPre, lapInput, 0);
			onevar.idxOne[0] = LOW_DIM(apPre);
			onevar.idxOne[1] = COL_DIM(apPre);
			onevar.idxOne[2] = COL_DIM(apOut);
			onevar.idxOne[3] = TON;//[3, 4] * [4, 2] => [3, 2]
			//arrangeLock(apSuf, bw_mutmx, lapInput->ptrRight, 0);
			TENSOR(apPre->quantum)->mxData->mone(tcxt, TENSOR(apOut->quantum)->mxGrad,
				TENSOR(apSuf->quantum)->mxGrad, 2, &onevar, AOP_MATMUL, 0, PDG, 1);
			//arrangeUnlock(apSuf, lapInput, 0);
			break;
		case TOB://[4, 3] * [2, 3]^ => [4, 2]
			onevar.idxOne[0] = LOW_DIM(apOut);//M
			onevar.idxOne[1] = COL_DIM(apOut);//K
			onevar.idxOne[2] = COL_DIM(apSuf);//N
			onevar.idxOne[3] = TON;//[4, 2] * [2, 3] => [4, 3]
			//arrangeLock(apPre, bw_mutmx, lapInput, 0);
			TENSOR(apOut->quantum)->mxGrad->mone(tcxt, TENSOR(apSuf->quantum)->mxData,
				TENSOR(apPre->quantum)->mxGrad, 2, &onevar, AOP_MATMUL, 0, PDG, 1);
			//arrangeUnlock(apPre, lapInput, 0);
			onevar.idxOne[0] = COL_DIM(apOut);
			onevar.idxOne[1] = LOW_DIM(apOut);
			onevar.idxOne[2] = COL_DIM(apPre);
			onevar.idxOne[3] = TOA;//[4, 2]^ * [4, 3] => [2, 3]
			//arrangeLock(apSuf, bw_mutmx, lapInput->ptrRight, 0);
			TENSOR(apOut->quantum)->mxGrad->mone(tcxt, TENSOR(apPre->quantum)->mxData,
				TENSOR(apSuf->quantum)->mxGrad, 2, &onevar, AOP_MATMUL, 0, PDG, 1);
			//arrangeUnlock(apSuf, lapInput, 0);
			break;
		case TOT://[3, 4]^ * [2, 3]^ => [4, 2]
			onevar.idxOne[0] = COL_DIM(apSuf);//M
			onevar.idxOne[1] = LOW_DIM(apSuf);//K
			onevar.idxOne[2] = LOW_DIM(apOut);//N
			onevar.idxOne[3] = TOT;//[2, 3]^ * [4, 2]^ => [3, 4]
			//arrangeLock(apPre, bw_mutmx, lapInput, 0);
			TENSOR(apSuf->quantum)->mxData->mone(tcxt, TENSOR(apOut->quantum)->mxGrad,
				TENSOR(apPre->quantum)->mxGrad, 2, &onevar, AOP_MATMUL, 0, PDG, 1);
			//arrangeUnlock(apPre, lapInput, 0);
			onevar.idxOne[0] = COL_DIM(apOut);
			onevar.idxOne[1] = LOW_DIM(apOut);
			onevar.idxOne[2] = LOW_DIM(apPre);
			onevar.idxOne[3] = TOT;//[4, 2]^ * [3, 4]^ => [2, 3]
			//arrangeLock(apSuf, bw_mutmx, lapInput->ptrRight, 0);
			TENSOR(apOut->quantum)->mxGrad->mone(tcxt, TENSOR(apPre->quantum)->mxData,
				TENSOR(apSuf->quantum)->mxGrad, 2, &onevar, AOP_MATMUL, 0, PDG, 1);
			//arrangeUnlock(apSuf, lapInput, 0);
			break;
		}
		multiArrangeUnlock(0);
		return lapInput;
	}
};
extern intt scoopeout_size(bool scoop_inner, intt seqy, intt seqx, intt slidey, intt slidex, intt stridey, intt stridex, intt &outy, intt &outx);
extern intt scoopeout_shape(bool scoop_inner, intt slidey, intt slidex, intt stridey, intt stridex, intt &outy, intt &outx, Flux *fxp, intt &ndim, intt axid[]);
class ApScoop : public Apply {//�׶��� �������� �ʰ� �׶��� �������� �ʰ� ������ ����.
public:
	Flux *apPre, *apOut;
	OneVar onev;
	intt scoopFeat;

	ApScoop(Trace *tcr, Flux *fxp, intt slidey, intt slidex, intt stridey, intt stridex, intt &ndim, intt axid[]) : Apply(tcr)
	{
		apPre = fxp;
		apCode = APC_SCOOP;
		nfanIn = 1;
		nfanOut = 1;
		registPhoto(fxp, nullx);
		//listingApin(fxp);

		scoopeout_shape(tcr->scoopinner, slidey, slidex, stridey, stridex, onev.idxOne[4], onev.idxOne[5], apPre, ndim, axid);

		onev.idxOne[0] = onev.idxOne[4] == 1 ? 1 : slidey;
		onev.idxOne[1] = slidex;
		onev.idxOne[2] = onev.idxOne[4] == 1 ? 1 : stridey;
		onev.idxOne[3] = stridex;

		if(onev.idxOne[4] == 1) {//1d
			onev.idxOne[6] = 1;				//pre y
			onev.idxOne[7] = fxp->fshape[1];//pre x
			onev.idxOne[8] = fxp->fshape[2];//feature
		} else {//2d
			onev.idxOne[6] = fxp->fshape[1];//pre y
			onev.idxOne[7] = fxp->fshape[2];//pre x
			onev.idxOne[8] = fxp->fshape[3];//feature
		}
		scoopFeat = onev.idxOne[0] * onev.idxOne[1] * onev.idxOne[8];
	}
	void *forward(TContext *tcxt, intt stridey, intt stridex)
	{
		Flux *fxr;
		sytet iv = invariance2(apPre);
		bool d2 = onev.idxOne[4] != 1 ? 1 : 0;

		multiArrangeLock();
		if(iv == 0 || (d2 && onev.idxOne[2] != stridey) || onev.idxOne[3] != stridex) {
			scoopeout_size(apTcr->scoopinner, d2 ? apPre->fshape[1] : 0, d2 ? apPre->fshape[2] : apPre->fshape[1],
				onev.idxOne[0], onev.idxOne[1], stridey, stridex, onev.idxOne[4], onev.idxOne[5]);
			apOut->resizing4(onev.idxOne[4] * onev.idxOne[5] * apPre->fshape[0]);
			onev.idxOne[2] = d2 ? stridey : 1;
			onev.idxOne[3] = stridex;
			scoopFeat = onev.idxOne[0] * onev.idxOne[1] * onev.idxOne[8];
		}
		TENSOR(apPre->quantum)->mxData->mone(tcxt, nullx, TENSOR(apOut->quantum)->mxData,
			2, &onev, AOP_ACTF, SCOOP_UP, PDC3, 0, scoopFeat);
		multiArrangeUnlock();

		return apOut;
	}
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx) { return apOut; }
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx) { return lapInput; }
};

class ApFork : public Apply {
public:
	Flux *apIn;
	Capsule *apOuts;

	ApFork(Trace *tcr, Flux *ip) : Apply(tcr)
	{
		apCode = APC_FORK;
		apIn = ip;
		nfanIn = 1;
		nfanOut = 0;
		apOuts = nullx;
		registPhoto(ip, nullx);
		listingApin(ip);
	}
	Flux *forkout(void);
	void *forward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock();
		Capsule *cap;
		intt i;
		if(apTcr->pathPrint) printf("fork fw\n");
		if(invariance2(apIn) == 0) {
			for(cap = apOuts, i = 0; cap; cap = cap->ptrRight, i++) {
				cap->vcaps->resizing2(apIn, "fork");
			}
		}
		
		for(cap = apOuts; cap; cap = cap->ptrRight) {
			TENSOR(cap->vcaps->quantum)->mxData->inCopy(TENSOR(apIn->quantum)->mxData, 0);
		}
		multiArrangeUnlock();
		return apOuts;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		multiArrangeLock(0);
		//arrangeLock(apIn, bw_mutmx, lapInput, 0);
		for(Capsule *cap = apOuts; cap; cap = cap->ptrRight) {
			TENSOR(cap->vcaps->quantum)->mxGrad->mone(tcxt, nullx, TENSOR(apIn->quantum)->mxGrad, 2, nullx, AOP_ACTF, DJUST_COPY, PDC, 1);
		}
		//arrangeUnlock(apIn, lapInput, 0);
		if(apTcr->pathPrint) printf("fork bw\n");
		multiArrangeUnlock(0);
		return lapInput;
	}
};
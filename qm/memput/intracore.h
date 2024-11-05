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
		//else 빌드단계에서만 장비메모리가 모잘라 할당실패될수있고 이때는 그래프실행 단계에서 재할당된다.
		//이경우 할당과정에서 즉시실행 off되어 오퍼레이션은 실행되지 않고 빌드만 된다.
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
	intt oidground(void)//대표로 출력의 첫번째의 그라운드 아이디 리턴
	{
		return DMATFX_GID(lapOuput->fxPoint);
	}
	intt iidground(void)//대표로 입력의 첫번째의 그라운드 아이디 리턴
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
		if(rv == 0) return 0;//불일치
		if(dirty) return 1;//일치
		else {//첫번째
			dirty = true;
			return -1;
		}
	}
	sytet invariance2(Flux *fxp) //첫번째 랭크만 불일치를 허용한다.
	{
		if(nprefPhoto != fxp->fdim) throwFault(-1, "inconsistancy dimension\n");
		if(memcmp(&prefPhoto[1], &fxp->fshape[1], (nprefPhoto - 1) * sizeof(intt))) throwFault(-1, "not first dim inconsistancy\n");
		if(prefPhoto[0] != fxp->fshape[0]) {
			if(fxp->fxType == trainable) throwFault(-1, "train not variable\n");
			prefPhoto[0] = fxp->fshape[0];
			dirty = true;
			return 0;//불일치
		}
		if(dirty) return 1;//일치
		else {//첫번째 
			dirty = true;
			return -1;
		}
	}
	void invariance3(Flux *ip, bool pref) //첫번째 랭크만 불일치를 허용한다.
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

		if(infx->ptrMastfx) {//infx가 ptrMastfx에서 reshape으로 생성된 fx이면 메모리는 공유되는데
			//플럭스는 각기 따로 생성되므로 nbRefer로 마스터의 메모리가 여러곳에서 참조는지 알수없으므로
			bw_mutmx_r = TENSOR(infx->ptrMastfx->quantum)->mxGrad;//무조건 블럭한다.
			LOCK_MUT_(bw_mutmx_r->mutmtx);
		} else {//if(infx->nbRefer > 1) {//infx가 다중참조되지 않아도 split의 출력 혹은 메모리 수급에의해 출력과 gid가 다를수있으므로 arrange수행.
			bw_mutmx_r = TENSOR(infx->quantum)->mxGrad;
			LOCK_MUT_(bw_mutmx_r->mutmtx);
			if(dev_arrange) {
				if(infx != cap->vcaps) throwFault(-1, "bw lock inconsistant fx\n");
				if(fw) {
					//입력과 출력 그라운드가 다르면 입력 그라운드 메모리를 쉐도우 메모리로 복사하고 쉐도우 메모리를 포커싱
					if(DMATFX_GID(infx) != arrangeGround) {// || infx->ds) {
						//if(cap->dbgIdShadow != cap->dataShadow->didshadow || infx->sizefx() != cap->dataShadow->shadowsz) exit(1);
						//printf("%p arange device %p %p %p %p %d %d %d %d %p\n", this, oper, cap, infx, oper->lapOuput->fxPoint, DMATFX_GID(bwfx), iGround, bwfx->scaleout, oper->apCode, cap->dataShadow);
						DMATFX(infx)->arrangeDevice(1, cap->dataShadow->didshadow, cap->dataShadow->devShadow, 1);//ground to shadow copy
						//printf("%p arange device2 %p %p %p\n", this, oper, cap, infx);
					} else DMATFX_SET_GROUND(infx, arrangeGround);
				} else {
					if(GMATFX_GID(infx) != arrangeGround) {// || infx->ds) {
						//printf("%p backward dev set %p %p %p %d %d %d %d %p\n", this, oper, infx, oper->lapOuput->fxPoint, DMATFX_GID(infx), iGround, infx->scaleout, oper->apCode, cap->gradShadow);
						GMATFX(infx)->arrangeDevice(1, cap->gradShadow->didshadow, cap->gradShadow->devShadow, 0);//ㄱ.그라운드 -> 쉐도우 카피(순전파에서 분기 됐을 경우 역전파 과정에서 분기점으로부터 기울기 더해저야하므로)
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
				//입력과 출력 그라운드가 다르면 입력 쉐도우 메모리를 그라운드 메모리로 복사하고 그라운드 메모리를 포커싱
				if(GMATFX_GID(infx) != arrangeGround) {// || infx->ds) {
					//printf("backward dev arrange %p %p %p %d %d %d %d\n", oper, infx, oper->lapOuput->fxPoint, DMATFX_GID(infx), iGround, infx->scaleout, oper->apCode);
					GMATFX(infx)->arrangeDevice(0, cap->gradShadow->didshadow, cap->gradShadow->devShadow, 0);//shadow to ground copy
				}
			}
			UNLOCK_MUT_(TENSOR(infx->quantum)->mxGrad->mutmtx);
		}
	}
	void multiArrangeLock(bool fw = true) //입력 여려개를 한꺼번에 블럭처리하고 정렬할 경우 호출
	{
		if(arrangeGround < 0) return;//빌드 과정에서 호출될때는 수행 않는다. arrangeLock는 백워드에서만 호출되므로 이것 고려안는다.
		Trace *trc = TRACER(lapInput->vcaps->fxTcr);
		Matrixr *mx = nullptr;
		//포워드는 정렬이 아니면 수행하지 않는다. 백워드는 정렬이 아니면 블럭만 수행.
		if(fw && (trc->devArrange == 0 || trc->cpmMode > 0)) return;
		//입력이 복수일때만 개별 블럭 수행을 블럭 처리(데드락 방지)
		if(lapInput->ptrRight) LOCK_MUT_(trc->mutArrange);
		if(fw) {
			for(FxAnchor *fxa = lapOuput; fxa; fxa = fxa->ptrRight) {//포커스가 그라운드 메모리가 
				//아니면 이 출력을 입력으로 하는 후행 어플라이 실행에서 포커스가 쉐도우로 설정한 것이기때문에
				//어플라이 실행전에 다시 포커스를 그라운드 메모리로 설정 한다.
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
		//포워드는 정렬이 아니면 수행하지 않는다. 백워드는 정렬이 아니면 블럭만 수행.
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
#define APC_ADMOPT		11 //optimizer는 이 사이에 위치 시켜야 함.
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
	bytet arith_msg[16];//디버깅 목적

	//cdim - 브로드캐스트에 의해 구성된 결과 디멘젼, rdim - 리턴 디멘젼 역젼파에서 소스 디멘전
	intt set_bro_dim(intt cdim, intt crank[], intt rdim, intt rrank[], intt bro_dim[], intt bro_idx[])
	{
		intt i, j, k, n = cdim - rdim, ndim;

		for(i = j = 0;i < n; i++) {//앞쪽을 cdim의 디멘젼들로 채운다.
			ndim = MRANK_SIZE(crank, i) / MRANK_SIZE(crank, i + 1);
			if(ndim == 1) continue;//디멘젼이 1이면 스킵
			bro_idx[j] = i;
			bro_dim[j++] = ndim;
		}
		for(k = 0;k < rdim - 1; k++, i++) {//rdim의 1인 디멘젼을 cdim으로 채운다.
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
			bwPreArv.bwGetOri = BWDIV_PREF;//pre표시
			bwPreArv.zarPre = arOut->fxSize;
			bwPreArv.zarOut = arPrefix->fxSize;
			bwPreArv.narMast = fwArv.narMast;
			memcpy(bwPreArv.arRankMast, fwArv.arRankMast, fwArv.narMast * sizeof(intt));
			bwPreArv.narPre = fwArv.narMast;//역전파에서는 master가 pre
			memcpy(bwPreArv.arRankPre, fwArv.arRankMast, fwArv.narMast * sizeof(intt));
			if(fwArv.narSuf) {
				bwPreArv.zarSuf = arSuffix->fxSize;
				bwPreArv.narSuf = fwArv.narSuf;
				memcpy(bwPreArv.arRankSuf, fwArv.arRankSuf, fwArv.narSuf * sizeof(intt));
			}//else suff는 스칼라
			bwPreArv.narRet = fwArv.narPre;
			memcpy(bwPreArv.arRankRet, fwArv.arRankPre, fwArv.narPre * sizeof(intt));
			bwPreArv.narBro = set_bro_dim(bwPreArv.narMast, bwPreArv.arRankMast, bwPreArv.narRet, bwPreArv.arRankRet, bwPreArv.broDimRet, bwPreArv.broIdxRet);
			if(bwPreArv.narBro) fwArv.narBro = 1;
			if(opArith == AOP_DIV) bwPreArv.bopAtrith = ABP_DIV_PREF;
			else if(opArith == AOP_MINUS) bwPreArv.bopAtrith = ABP_MINUS_PREF;
			else bwPreArv.bopAtrith = opArith;
		}//else pre는 스칼라 역전파 없음
		if(fwArv.narSuf) {
			bwSufArv.paintVar = 1;
			bwSufArv.bwGetOri = BWDIV_SUFF;//suff표시
			bwSufArv.zarPre = arOut->fxSize;
			bwSufArv.zarOut = arSuffix->fxSize;
			bwSufArv.narMast = fwArv.narMast;
			memcpy(bwSufArv.arRankMast, fwArv.arRankMast, fwArv.narMast * sizeof(intt));
			bwSufArv.narPre = fwArv.narMast;//역전파에서는 master가 pre
			memcpy(bwSufArv.arRankPre, fwArv.arRankMast, fwArv.narMast * sizeof(intt));
			if(fwArv.narPre) {
				bwSufArv.zarSuf = arPrefix->fxSize;
				bwSufArv.narSuf = fwArv.narPre;
				memcpy(bwSufArv.arRankSuf, fwArv.arRankPre, fwArv.narPre * sizeof(intt));
			}//else pre는 스칼라
			if(opArith == AOP_DIV) {//div연산 역전파는 분모 역수 제곱이므로 suff를 suff로
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
			if(opArith == AOP_DIV) bwSufArv.bopAtrith = ABP_DIV_SUFF;//원래 위아래가 바뀌어 ABP_MINUS_SUFF가 설정됐음, 나중에 검토
			else if(opArith == AOP_MINUS) bwSufArv.bopAtrith = ABP_MINUS_SUFF;
			else bwSufArv.bopAtrith = opArith;
		}//else suff는 스칼라 역전파 없음
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
		multiArrangeLock(0);//prefix, suffix각 역전파 할때 상대방이 참조 되므로 두개 동시에 블럭을 설정해야 한다.
		if(apTcr->pathPrint) printf("arith bw: %d\n", opArith);
		if(fwArv.narPre) {
			//arrangeLock(arPrefix, bw_mutmx, lapInput, 0);
			TENSOR(arOut->quantum)->mxGrad->marith(tcxt, (fwArv.narSuf ? TENSOR(arSuffix->quantum)->mxData : nullx), TENSOR(arPrefix->quantum)->mxGrad,
				&bwPreArv, tScalarv, svbuf, (void *)1, bwPreArv.bopAtrith);//ABP_BWTEST);//
			//arrangeUnlock(arPrefix, lapInput, 0);
		}
		if(fwArv.narSuf) {//pre가 플럭스이면 리스트 두번째 것
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
	intt first_ax_ret[MX_DIM], second_ax_ret[MX_DIM];//cublas dot에서만 사용.
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
	{//조인축이 pre와 suf간에 [1,2] [2,1]과 같이 역전되있을 경우 백워드 과정에서 리턴 매트릭스를 기준으로 상대방을 축에 따라 역전시킨다.
		struct ax_rink {
			intt axis, ith;
			struct ax_rink *ptrLeft, *ptrRight;
		} *outl = nullx, *jol = nullx, *axr, *axr2;
		intt i;

		for(i = 0;out_ax[i] >= 0; i++) {//축이 상위값(낮은값)일수록 앞서게 오러딩
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
		for(i = 0;i < njo_ret; i++) {//축이 상위값(낮은값)일수록 앞서게 오러딩
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
		for(axr = jol, axr2 = outl;axr;) {//리턴 매트릭스의 축 위치에 상대방을 위치시킨다.
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
			for(i = k = 0;out_ax_pre[i] >= 0; i++) first_ax_ret[i] = k++;//순전파때의 결과ret매트릭스의 pre와 suf차원 설정
			first_ax_ret[i] = -1;
			for(i= 0;out_ax_suf[i] >= 0; i++) second_ax_ret[i] = k++;
			second_ax_ret[i] = -1;
		} else if(bw_get_ori == BWDIV_PREF) {
			if(fwDotv.transOrder & TOA) {
				pre = dotSuffix;
				suf = dotOut;
				pj = out_ax_suf;//순전파때 출력 차원은 역전파때 조인 차원이 된다.
				sj = second_ax_ret;//ret매트릭스는 pref 역전파에는 항상 ret매트릭스의 suf차원이 조인차원으로 온다.
				po = jo_ax_suf;//남은 차원을 출력 차원으로 설정.
				so = first_ax_ret;//남은 차원을 출력 차원으로 설정.
			} else {
				pre = dotOut;
				suf = dotSuffix;
				pj = second_ax_ret;//ret매트릭스는 pref 역전파에는 항상 ret매트릭스의 suf차원이 조인차원으로 온다.
				sj = out_ax_suf;
				po = first_ax_ret;
				so = jo_ax_suf;
			}
		} else {
			if(fwDotv.transOrder & TOB) {
				pre = dotOut;
				suf = dotPrefix;
				pj = first_ax_ret;//ret매트릭스는 suf 역전파에는 항상 ret매트릭스의 pre차원이 조인차원으로 온다.
				sj = out_ax_pre;
				po = second_ax_ret;
				so = jo_ax_pre;
			} else {
				pre = dotPrefix;
				suf = dotOut;
				pj = out_ax_pre;
				sj = first_ax_ret;//ret매트릭스는 suf 역전파에는 항상 ret매트릭스의 pre차원이 조인차원으로 온다.
				po = jo_ax_pre;
				so = second_ax_ret;
			}
		}
		dotv->transOrder = 0;
		if(po[0] == 0) {//dot연산은 pre의 하위 차원과 suf의 상위 차원 연결이 정상이고 양측에서 이와 반대이면 전치로 설정한다.
			for(i = 0, k = 1;pj[i] >= 0; i++) k *= pre->fshape[pj[i]];
		} else {//pre의 최상위 차원이 0 이 아니면 최상위가 pj(조인차원)에 있으므로 전치된 것이다.
			dotv->transOrder = 1;
			for(i = 0, k = 1;po[i] >= 0; i++) k *= pre->fshape[po[i]];
		}
		dotv->prem = pre->fxSize / k;
		dotv->joik = k;
		dotv->lda = k;
		if(sj[0] == 0) {
			for(i = 0, k = 1;sj[i] >= 0; i++) k *= suf->fshape[sj[i]];
		} else {//suf의 최상위 차원이 0 이 아니면 최상위가 so(출력차원)에 있으므로 전치된 것이다.
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
			jpre_dims[i] = fxp->fshape[jo_pre_ax[i]];//조인 차원의 디멘젼들을 pre와 suf가 같아야 하므고 대표로 pre에서 설정.
			dotv->joAxisPre[i] = jo_pre_ax[i];//조인트 차원 인덱스 설정.
			dotv->sprPreJo[i].rkdim = fxp->fshape[jo_pre_ax[i]];
			dotv->sprPreJo[i].rksz = FX_SIZE_RK2(fxp, jo_pre_ax[i]) < 0 ? 0 : (fxp->fdim -1 == jo_pre_ax[i] ? 1 : FX_SIZE_RK(fxp, jo_pre_ax[i] + 1));
			dotv->sprPreJo[i].rktsz = dotv->sprPreJo[i].rksz * (dotv->sprPreJo[i].rkdim -1);
			jo_sz_pre *= fxp->fshape[jo_pre_ax[i]];
		}
		dotv->njoPre = i; 
		for(i = 0;i < dotv->njoPre; i++) axid[i] = fxp->fshape[*(dotv->joAxisPre + i)];//조인트 차원 디멘젼설정.
		Matrixr::make_rank_sz(dotv->njoPre, axid, dotv->joRankPre);//조인트 차원 랭크 사이즈 계산.gpu 캐쉬버전만 적용

		dotv->nJointAxis = 1;
		dotv->jdimEqual = 1;
		for(i = 0;jo_suf_ax[i] >= 0; i++) {
			jsuf_dims[i] = fxs->fshape[jo_suf_ax[i]];
			if(jpre_dims[i] != jsuf_dims[i]) dotv->jdimEqual = 0;
			dotv->joAxisSuf[i] = jo_suf_ax[i];
			dotv->sprSufJo[i].rkdim = fxs->fshape[jo_suf_ax[i]];
			dotv->sprSufJo[i].rksz = FX_SIZE_RK2(fxs, jo_suf_ax[i]) < 0 ? 0 : (fxs->fdim - 1 == jo_suf_ax[i] ? 1 : FX_SIZE_RK(fxs, jo_suf_ax[i] + 1));
			dotv->sprSufJo[i].rktsz = dotv->sprSufJo[i].rksz * (dotv->sprSufJo[i].rkdim -1);
			dotv->nJointAxis *= fxs->fshape[jo_suf_ax[i]];//조인트 총 갯수 계산.
		}
		dotv->njoSuf = i;
		for(i = 0;i < dotv->njoSuf; i++) axid[i] = fxs->fshape[*(dotv->joAxisSuf + i)];//조인트 차원 디멘젼설정.
		Matrixr::make_rank_sz(dotv->njoSuf, axid, dotv->joRankSuf);//조인트 차원 랭크 사이즈 계산.gpu 캐쉬버전만 적용

		if(jo_sz_pre != dotv->nJointAxis) throwFault(-1, "dot inconsistant shape");
		dotv->bwGetOri = bw_get_ori;
		if(bw_get_ori) {
			dotv->bwMxp = (fxp == dotOut ? TENSOR(fxp->quantum)->mxGrad : TENSOR(fxp->quantum)->mxData);
			dotv->bwMxs = (fxs == dotOut ? TENSOR(fxs->quantum)->mxGrad : TENSOR(fxs->quantum)->mxData);
			if(fwDotv.jdimEqual == 0) dotv->intervOut = 1;//순전파의 조인 디맨젼이 일치하지 않으면 역전파에서 좌표 변환하게 설정한다.
		}// else fxp->nJoint = fxs->nJoint = dotv->nJointAxis;
		if(bw_get_ori == BWDIV_PREF) {//preffix
			dotv->njoRet = fwDotv.njoPre;
			dotv->noutRet = fwDotv.noutPre;
			memcpy(dotv->joAxisRet, fwDotv.joAxisPre, dotv->njoRet * sizeof(intt));
			memcpy(dotv->outAxisRet, fwDotv.outAxisPre, dotv->noutRet * sizeof(intt));
		} else if(bw_get_ori == BWDIV_SUFF) {//suffix, 출력 순서가 출력 축을 따라가고 szShrinkSuf 갯수를 넘으면 조인축에 단위가 순차로 올라간다.
			dotv->njoRet = fwDotv.njoSuf;
			dotv->noutRet = fwDotv.noutSuf;
			memcpy(dotv->joAxisRet, fwDotv.joAxisSuf, dotv->njoRet * sizeof(intt));
			memcpy(dotv->outAxisRet, fwDotv.outAxisSuf, dotv->noutRet * sizeof(intt));
		} else dotv->intervOut = 0;
		if(dotv->intervOut && dotv->retFirst == 0) jo_rank_arange(out_pre_ax, dotv->njoRet, dotv->joAxisRet);
		for(i = j = 0;out_pre_ax[i] >= 0; i++) {//조인트가 아닌 출력 차원 인덱스 설정
			dotv->outAxisPre[i] = out_pre_ax[i];//[2,3,4]*[3,2]일때 (1,0)축으로 dot하면 out은 [2,4,2]가되고 역전파일때 
			if(dotv->intervOut) {//pre는 [out*suf]로서 [2,4,2]*[3,2]를 (2,1)축으로 dot하여 [2,4,3]으로 획득되고 ret(순전파때 pre)의
				if(dotv->retFirst) k = dotv->outAxisRet[i];//출력축 위치(0,2)에 이번 역전파 dot의 predout인[2,4]를 넣고 ret의 조인축
				else k = dotv->joAxisRet[i];//(1)에 ㄱ)에서 [3]을 위치시켜 [2,3,4]를 만들어 순전파때의 pre를 획득한다.
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
				if(dotv->retFirst) k = dotv->joAxisRet[i];//ㄱ.
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
		dotv->intervOut = 0;//이후에는 이 값이 true로 있으면 호스트 디바이스간 메모리 복사 시점에 영향을 미치므로 리셋.
		i = dotv->noutRank - 1;//이하 gpu 캐쉬 버전만 적용
		bool pref = dotv->sprPreOut[i].rkPref;
		for(;;i--) {
			if(dotv->sprPreOut[i].rkPref != pref) break;
		}
		dotv->axisCache = i;
		intt sz_smit = MRANK_SIZE(dotv->outRank, dotv->axisCache + 1);//연속하는 최 말단 차원 원소 수
		if(sz_smit > DOT_BLOCK) {
			intt d = sz_smit / DOT_BLOCK + (sz_smit % DOT_BLOCK ? 1 : 0);//블럭사이즈 단위 균등 분할 젯수 초기값을 설정.
			for(;sz_smit % d && sz_smit / d > DOT_BLOCK / 2; d++);//초기 분할 젯수값이 균등이 아니면 젯수값을 증가시키며 균등 젯수값을 find
			if(sz_smit % d || sz_smit / d < DOT_BLOCK / 2) {//젯수가 균등분할 계수가 아니거나 젯수에의한 분할 사이즈가 커널 사이즈의 반 이하이면 균등을 포기한다.
				dotv->szOutKernel = DOT_BLOCK;
				dotv->fitOutKernel = 0;
			} else {
				dotv->szOutKernel = sz_smit / d;
				dotv->fitOutKernel = 1;
			}
			dotv->szJoKernel = dotv->szOutKernel;//커널사이즈로 최말단 차원을 하나만(smit가 더크면 한번체 못하고 나누어서) 처리할수있으므로 조인 커널은 출력 커널 전체 사이즈
			dotv->nrecycCache = sz_smit / dotv->szOutKernel + (dotv->fitOutKernel == 0 ? 1 : 0) -1;//한번더 반복되면 1설정.
			dotv->shareUnit = sz_smit;// * NP_BLOCK;//gpu 캐쉬 다중 블럭 버전만 사용. 한 블럭내에서 더 많이 처리할려면(그리드사이즈를 줄일려면 NP_BLOCK를 설정한다.)
		} else {
			intt n_smit = DOT_BLOCK / sz_smit;//블럭 사이즈안에 들어갈 수있은 최말단의 상위 차원 원소 갯수
			if(n_smit == 0) n_smit = 1;
			dotv->szOutKernel = n_smit * sz_smit;//연속 최말단 차원 여러개를 한번에 처리하는 커널 사이즈
			dotv->szJoKernel = sz_smit;
			dotv->fitOutKernel = 1;
			dotv->nrecycCache = 0;
			dotv->shareUnit = dotv->szOutKernel;// * NP_BLOCK;//gpu 캐쉬 다중 블럭 버전만 사용. 한 블럭내에서 더 많이 처리할려면(그리드사이즈를 줄일려면 NP_BLOCK를 설정한다.)
		}
		dotv->fitJoKernel = (dotv->nJointAxis % dotv->szJoKernel ? 0 : 1);
		dotv->ncycJo = dotv->nJointAxis / dotv->szJoKernel + (dotv->fitJoKernel == 0 ? 1 : 0) -1;//한번더 반복되면 1설정.
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
	//A^: a0가 A의 first, a0는 b0와 같고 따라서 b0를 first로 하면 b1이 그 후행이 되서 [b0,b1]이 pre가되고 
	//		b1은 C matrix의 c1이므로 [c1, c0]가 suf가 된다.
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
		//ax_out - 순전파때 pre와 suf의 axis(인덱스)로 구성되는 출력 매트릭의 axis, ax_do - 출력 매트릭스의 인덱스를 갖는 axis
		joint_opposit = p_that(ax_first, ax_join);
		if(joint_opposit) {//순전파때 first로 조인되었으면 
			ax_out_pre = joint_opposit;//first가 속한 matrix와 곱해진(조인된) 상대 matirx의 출력 axis를 pre out로 설정.
			ax_jo_pre = p_that(joint_opposit, ax_opposit);//상대 matrix이 조인 axis을 획득.
			i_do = i_this(ax_jo_pre, ax_out);//조인 axis가 출력매트릭스의 pre와 suf중 어느쪽인가 검책하여 그 인덱스를 획득.
			ax_jo_suf = ax_do[i_do];//획득된 조인측 인덱스로 출력 매트릭스 상의 조인 axis를 suf join으로 설정.
			ax_out_suf = ax_do[!i_do];//그 반대를 suf out으로 설정.
			fx_pre = fx_opposit;//first의 상태 메트릭스가 이번 역전파 곲의 preffix가 된다.
			fx_suf = fx_out;//순전파때의 출력 매트릭스가 이번 역전파 곱의 suffix가 된다.
			//printf("%d(%d) %d(%d) %d(%d) %d(%d)\n", fx_pre->fshape[*ax_out_pre], *ax_out_pre,
			//	fx_pre->fshape[*ax_jo_pre], *ax_jo_pre, fx_suf->fshape[*ax_jo_suf], *ax_jo_suf,
			//	fx_suf->fshape[*ax_out_suf], *ax_out_suf);
			return 0;
		} else {//순전파때 first가 조인되지 않았으면 출력매트릭스에 출력됐으므로 
			i_do = i_this(ax_first, ax_out);//first가 출력매트릭스의 pre와 suf중 어느쪽인가 검책하여 그 인덱스를 획득.
			ax_out_pre = ax_do[i_do];//하여 출력매트릭스에서 그 인덱스에 해당하는 axis를 pre out로 설정하고 
			ax_jo_pre = ax_do[!i_do];//출력매트릭스에서 그 반대를 pre join으로 설정.
			i_opposit = i_this(ax_out[!i_do], ax_opposit);//상대 매트릭스에서 조인되는 axis 인덱스 회득하여 
			ax_jo_suf = ax_opposit[i_opposit];//상대 매트릭스의 조인 axis 설정
			ax_out_suf = ax_opposit[!i_opposit];//그 반대를 출력 axis로 설정.
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
		multiArrangeLock(0);//prefix, suffix각 역전파 할때 상대방이 참조 되므로 두개 동시에 블럭을 설정해야 한다.
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
			//원래의 배치 차원이 전치되어 뒤로가고 다른 차원이 0번 축으로되어 병합되는 경우가 있으니 사용상 주의한다.
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
		if(apTcr->pathPrint) printf("reshape fw\n");//메모리가 공유되는 것으로 바꿨으므로 복사안한다.
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
			for(;txid[j] != i; j = txid[j]);//전치인덱스를 자기참조한것이 현 자리가 아닌 
			bwTrsv.trTxid[i] = j;//마지막 자기참조인덱스가 j가 역전파 전치 인덱스
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
		if(iv == 0) {//첫번째 랭크가 변화됐는데
			//for(intt i = 1;i < trsVar.ntrDims; i++) {//타겟에 다른 랭크로 전치되면 안된다.
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
		tcr->directx(1);//그라프 연결하지 않게 한다.
		smcross = flux(tcr, sfOut->fdim +1, dim, sfIn->qType, variable);
		smcross->scaleout = ip->scaleout;//smcross는 ip와 배치 연동되야 하므로 스케일값 설정
		smcross2 = *smcross * -1.0;//-aa
		a_dx = 1.0 - *sfOut;//(1-a)
		a_dx->scaleout = ip->scaleout;//sfOut의 스케일아웃은 리턴되야 설정되므로 직접설정
		a_dx2 = *sfOut * *a_dx;//a(1-a)
		tcr->directx(0);//해제

		sfmSum = apTcr->instMatrix(sfmSum, sfOut->qType, sfIn->fdim - 1, sfIn->fshape, false, JUST_SIZEDEF);
		sfmMax = apTcr->instMatrix(sfmMax, sfOut->qType, sfIn->fdim - 1, sfIn->fshape, false, JUST_SIZEDEF);
		sfmBuf = apTcr->instMatrix(sfmBuf, sfOut->qType, 1, &sfIn->fshape[sfIn->fdim - 1], false, JUST_SIZEDEF);
	}
	intt mcalcapp(intt n) //배치를 n으로 했을때 어플라이 내부 사용 메모리 계산, 이를위해 이들 매트릭스를
	{						//위에서 메모리 할당 없이 사이즈 계산만 할수있게 생성
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
			//어플라이의 출력이 아니어서 그래프에 의해 출력 그라운드에 자동 재할당되지 않으므로 여기서 직접 재할당
			//사이즈 변경만이면 smcross2, a_dx, a_dx2는 수행과정에서 사이즈 체크되어 자동으로 재할당되어 
			//여기서 직접 재할당 할 필요없지만 그라운드 아이디가 변경된 경우 동일한 gid에 할당되야 하므로 직점 재할당.
			smcross->resizing5(sfIn->scaleout, DMATFX(sfOut)->didground);
			smcross2->resizing5(sfIn->scaleout, DMATFX(sfOut)->didground);
			a_dx->resizing5(sfIn->scaleout, DMATFX(sfOut)->didground);
			a_dx2->resizing5(sfIn->scaleout, DMATFX(sfOut)->didground);
		}
		if(iv <= 0) {
			intt gid = DMATFX(sfOut)->didground;//gid 그라운드에서의 할당 공간은 위 mcalc app에서 계산되어 확보되었다.
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
		//sfOut의 원래 shape은 [batch, seq, feat]이나 dimension 1은 어차피 크기가 없으므로  
		onevar.idxOne[0] = sfOut->fshape[idf];//M, //[batch, seq, feat, 1], mat mul하기위해 K값으로 1을 준다.
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
		onevar.idxOne[0] = 1;//M, //[batch, seq, 1, feat], delta-out의 원래 shape은 [batch, seq, feat]이나 dimension 1은 어차피 크기가 없으므로 mat mul하기위해 M(로우)값으로 1을 준다.
		onevar.idxOne[1] = sfOut->fshape[idf];//K(feat)
		onevar.idxOne[2] = sfIn->fshape[idf];//N(feat) //[batch, seq, feat, feat]
		onevar.idxOne[3] = 0;//원래 출력 모양(delta-in)은 [batch, seq, 1, feat]가 되야하나 sfIn의 모양인 [batch, seq, feat]에 적재되도 된다. 1은 크기가 없으므로
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
		listingApin(tp);//tp타겟은 역전파하지 않을 려면 현 라인과 asce), fsce)에서 fxt reference되지 않게 뺄것

		sfmSum = apTcr->instMatrix(sfmSum, sfcOut->qType, sfcIn->fdim - 1, sfcIn->fshape, false, JUST_SIZEDEF);
		sfmMax = apTcr->instMatrix(sfmMax, sfcOut->qType, sfcIn->fdim - 1, sfcIn->fshape, false, JUST_SIZEDEF);
		sfomtx = apTcr->instMatrix(sfomtx, sfcOut->qType, sfcIn->fdim, sfcIn->fshape, false, JUST_SIZEDEF);
		sfoBuf = apTcr->instMatrix(sfoBuf, sfcOut->qType, 1, &sfcIn->fshape[sfcIn->fdim - 1], false, JUST_SIZEDEF);
	}
	intt mcalcapp(intt n) //배치를 n으로 했을때 어플라이 내부 사용 메모리 계산, 이를위해 이들 매트릭스를
	{						//위에서 메모리 할당 없이 사이즈 계산만 할수있게 생성
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
			intt gid = DMATFX(sfcOut)->didground;//gid 그라운드에서의 할당 공간은 위 mcalc app에서 계산되어 확보되었다.
			sfmSum = apTcr->instMatrix(sfmSum, sfcOut->qType, sfcIn->fdim - 1, sfcIn->fshape, false, gid);
			sfmMax = apTcr->instMatrix(sfmMax, sfcOut->qType, sfcIn->fdim - 1, sfcIn->fshape, false, gid);
			sfomtx = apTcr->instMatrix(sfomtx, sfcOut->qType, sfcIn->fdim, sfcIn->fshape, false, gid);
			sfoBuf = apTcr->instMatrix(sfoBuf, sfcOut->qType, 1, &sfcIn->fshape[sfcIn->fdim - 1], false, gid);
			minusArv.paintVar = devideArv.paintVar = 0;
			intt bn = (apTcr->ibatchSize ? apTcr->ibatchSize : sfcIn->fshape[0]);
			if(bn >= apTcr->gradby) bn /= apTcr->gradby;
			//intt bn = (apTcr->ibatchSize ? apTcr->ibatchSize : MTX_SIZE(TENSOR(sfcIn->quantum)->mxData));
			copy_cval_type(smcVar.idxOne, bn, sfomtx->mxType);//배치 사이즈
			//copy_cval_type(smcVar.idxOne, MTX_SIZE(sfmSum), sfomtx->mxType);//배치 사이즈
			//copy_cval_type(&withBatch, MTX_SIZE(sfmSum), sfomtx->mxType);//배치 사이즈
		}
		if(apTcr->pathPrint) printf("softmax cross entroy err fw\n");
		
		TENSOR(sfcIn->quantum)->mxData->msoftmax(tcxt, sfomtx, sfmSum, sfmMax, sfoBuf, nullx);
		sfomtx->msoftx_cross_e(tcxt, TENSOR(sfcOut->quantum)->mxData, TENSOR(sfcTar->quantum)->mxData);//타겟값이 0이면
		//TENSOR(sfcOut->quantum)->mxData->printo();//0값이 곱해지므로 안된다. 크로스엔트로피는 이산값에 쓰이고 따라서 0값은 널값으로 둬야한다.
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
		//	&devideArv, sfomtx->mxType, &withBatch, nullx, AOP_DIV);//배치 사이즈로 나눔.
		sfomtx->mone(tcxt, TENSOR(sfcTar->quantum)->mxData, TENSOR(sfcIn->quantum)->mxGrad,
			2, &smcVar, AOP_ACTF, DLOSS_FUNC, PDC, 1);
		//arrangeUnlock(sfcIn, lapInput, 0);
		/*arrangeLock(sfcTar, bw_mutmx, lapInput, 0);
		//asce.이하. 타겟쪽 기울기 전파는 sgd는 정확하나 cross e는 오차가 발생, 뭔가 미분식이 틀린것 같음. 나중에 수정.
		//TENSOR(sfcTar->quantum)->mxData->marith(tcxt, sfomtx, TENSOR(sfcTar->quantum)->mxGrad,
		//	&minusArv, 0, nullx, nullx, AOP_MINUS);
		//TENSOR(sfcTar->quantum)->mxGrad->marith(tcxt, nullx, TENSOR(sfcTar->quantum)->mxGrad,
		//	&devideArv, sfomtx->mxType, &withBatch, nullx, AOP_DIV);//배치 사이즈로 나눔.
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
			meanby = apTcr->instMatrix(meanby, apIn->qType, 1, mb_axid, false);//axid는 매트릭스 생성내부에서 포인터로 설정된다.
			floatt cv = 1.0 / apIn->fxSize;
			meanby->fill(&cv, tfloat);
		}
		registPhoto(ip, nullx);
		listingApin(ip);
	}
	intt mcalcapp(intt n) //배치를 n으로 했을때 어플라이 내부 사용 메모리 계산
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
		sumsz = apTcr->instMatrix(sumsz, apIn->qType, 1, mb_axid, 0);//axid는 매트릭스 생성내부에서 포인터로 설정된다.
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
		withMean = mean;//평균까지 구하는 것이 아니면 배치 단위로(배치를 제외한 차원들의) 
						//차이제곱을 구함, 역전파는 어차피 소스와 타겟의 차이값으로 구하므로 동일하게 처리
		//minusArv.paintVar = divArv.paintVar = mulArv.paintVar = dminusArv.paintVar = 0;
		//copy_cval_type(&withHalf, 0.5, TENSOR(apIn->quantum)->mxData->mxType);//제곱 평균 에러는 1/2로 나움
		registPhoto(ip, tp);
		listingApin(ip);
		listingApin(tp);//tp타겟은 역전파하지 않을 려면 현 라인과 amse), fmse)에서 fxt reference되지 않게 뺄것
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
			copy_cval_type(mseVar.idxOne, bn, TENSOR(apIn->quantum)->mxData->mxType);//배치 사이즈
			//copy_cval_type(&withBatch, bn, TENSOR(apIn->quantum)->mxData->mxType);//배치 사이즈
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
		//	&divArv, TENSOR(apIn->quantum)->mxData->mxType, &withBatch, nullx, AOP_DIV);//배치 사이즈로 나눔.
		TENSOR(apIn->quantum)->mxData->mone(tcxt, TENSOR(apTar->quantum)->mxData, TENSOR(apIn->quantum)->mxGrad,
			2, &mseVar, AOP_ACTF, DLOSS_FUNC, PDC, 1);
		//arrangeUnlock(apIn, lapInput, 0);
		/*
		//arrangeLock(apTar, bw_mutmx, lapInput, 0);
		//amse.이하
		//TENSOR(apTar->quantum)->mxData->marith(tcxt, TENSOR(apIn->quantum)->mxData, TENSOR(apTar->quantum)->mxGrad,
		//	&minusArv, 0, nullx, nullx, AOP_MINUS);
		//TENSOR(apTar->quantum)->mxGrad->marith(tcxt, nullx, TENSOR(apTar->quantum)->mxGrad,
		//	&divArv, TENSOR(apTar->quantum)->mxData->mxType, &withBatch, nullx, AOP_DIV);//배치 사이즈로 나눔.
		TENSOR(apTar->quantum)->mxData->mone(tcxt, TENSOR(apIn->quantum)->mxData, TENSOR(apTar->quantum)->mxGrad,
			2, &mseVar, AOP_ACTF, DLOSS_FUNC, PDC, 1);
		arrangeUnlock(apTar, lapInput, 0);*/
		multiArrangeUnlock(0);
		return lapInput;
	}
};

class ApActf : public Apply {//수식 계산에 사용된다.
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
		//	dactfMat = apTcr->instMatrix(dactfMat, apIn->qType, apIn->fdim, apIn->fshape, false);//수식 미분 결과 적재 매트릭스
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
		apSuf = fxs;//a값
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
		if(opActf == ACTF_PRELU) {//가중치(apSuf)가 배치를 포함하므로 배치가 셔플되면 학습이 어려워 짐
			iv = invariance2(apPref);
			if(iv == 0) {
				apOut->resizing2(apPref, "activation2");
				floatt pv = apSuf->at_d(0);
				//어플라이의 출력이 아니어서 그래프에 의해 출력 그라운드에 자동 재할당되지 않으므로 여기서 직접 재할당
				lapInput->ptrRight->freeShadow();//apSuf//그라프에 의해 쉐도우 할당됐다면 출력 그라운드에 재할당하므로 해제
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
		//arrangeLock(apPref, bw_mutmx, lapInput, 0);//apSuf는 prelu에서만 사용되는 prelu가중치이므로 락 필요없다.adar)설명참조
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
		apTcr->initArrange();//초기 한번 초기화 한다.feed않고 실행할 경우에도 arrange수행될수있도록
		apTcr->listWeight();//위 v sync후에 해야함.apTcr->elistw(false, memput::mp::trainable);
		apTcr->bwVersion++;//최초 run실행에서 v sync되게 하기위해
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
		optp->wgUpdate = wfx;//gid 그라운드에 아래 두개 매트릭스 할당 공간은 adcg)에서 미리 확보되었다.
		//생성만 되고 참조되지 않는 가중치는 adcg)와 여기서도 제외되므로 공간확보에 누락분은 없다.
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
				if(gid != GMATFX_GID(wfx)) {//가중치와 같은 기계에 할당.
					gid = GMATFX_GID(wfx);//backwardDevArrange에의해 가중치는 모두 그라운드로 복사된다.
					CudaDevSet(gid);//빌드시점에 트레이스 메인 쓰레드에서 장비 arrange된다.
				}
				adamParam(wfx, gid);
			}
		} else {
			for(; wlist; wlist = wlist->ptrRight) {
				if(gid != GMATFX_GID(wlist->vcaps)) {//가중치와 같은 기계에 할당.
					gid = GMATFX_GID(wlist->vcaps);
					CudaDevSet(gid);//빌드시점에 트레이스 메인 쓰레드와 장비 공유된다.
				}
				adamParam(wlist->vcaps, gid);
			}
		}
		//if(gid != apTcr->didTrace) {//didTrace는 메인 trc의 디폴트 장비를 지정한 것으로서
		//	CudaDevSet(apTcr->didTrace);//변경되면 안되고 이 아이디에 현행 장비 포커스를 맟춘다.
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
			if(gid != GMATFX_GID(optp->wgUpdate)) {//가중치와 같은 기계에서 작업.
				gid = GMATFX_GID(optp->wgUpdate);
				CudaDevSet(gid);//트레이스의 run함수 수행 쓰레드와 장비 공유된다.
			}
			DMATFX_SET_GROUND(optp->wgUpdate, gid);//순전파과정에서 데이터의 쉐도우가 포커싱 됐을수도있으므로 그라운드 포커싱한다.
			if(optp->wgUpdate->fxSize != optp->movavg->maxmSize) {//prelu의 경우 가중치값이 배치 사이즈를 포함하므로 사이즈 변경되면 반영한다.
				optp->movavg = apTcr->instMatrix(optp->movavg, optp->wgUpdate->qType, optp->wgUpdate->fdim, optp->wgUpdate->fshape, false, gid);
				optp->movavgv = apTcr->instMatrix(optp->movavgv, optp->wgUpdate->qType, optp->wgUpdate->fdim, optp->wgUpdate->fshape, false, gid);
				optp->movavg->resetMemory(0);//gid 그라운드에 위 두개 매트릭스 할당 공간은 adcg)에서 미리 확보되었다.
				optp->movavgv->resetMemory(0);//생성만 되고 참조되지 않는 가중치는 adcg)와 여기서도 제외되므로 공간확보에 누락분은 없다.
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
			if(gid != GMATFX_GID(optp->vcaps)) {//가중치와 같은 기계에서 작업.
				gid = GMATFX_GID(optp->vcaps);
				CudaDevSet(gid);//트레이스의 run함수 수행 쓰레드와 장비 공유된다.
			}
			DMATFX_SET_GROUND(optp->vcaps, gid);//순전파과정에서 데이터의 쉐도우가 포커싱 됐을수도있으므로 그라운드 포커싱한다.
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
		apPrefix = fxp;//임배딩 테이블
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
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx) //suffix로 역전파에 의한 가중치 변경은 되지 않는다. suffix는 트래이닝 대상이 아님.
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
		apOut->fill(*(doublet *)&onev.idxOne[2]);//off value먼저 일괄 설정, 여기서 off value로 채워진 출력 매트릭스를
		//아래 함수에서 gpu실행될때 호스트에서 디바이스 메모리로 복사하기위해 끝에 rplus를 -1로 호출한다.(cpu실행이면 복사안됨)
		
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
	intt mcalcapp(intt n) //배치를 n으로 했을때 어플라이 내부 사용 메모리 계산, 이를위해 이들 매트릭스를
	{						//위에서 메모리 할당 없이 사이즈 계산만 할수있게 생성
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
				intt gid = DMATFX(apOut)->didground;//gid 그라운드에서의 할당 공간은 위 mcalc app에서 계산되어 확보되었다.
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
			//apOut->resizing4(nfirst);//첫번째 랭크 사이즈를 계산하여 리사이징
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
		
		if(iv == 0) apSuffix->resizing2(apPrefix, "overwrite");//apSuffix는 배치 리사이즈되어 
													//호출되는 것이 기본이므로 필요없으나 그냥 한다.
		apSuffix->copyf(apPrefix);
		multiArrangeUnlock();

		return apOut;
	}
	Capsule *backward(TContext *tcxt, Matrixr *&bw_mutmx)
	{
		return lapInput;//기울기 역전파는 하지 않고 역전파 카운트만 하도록 인풋 리스트를 리턴한다.
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
		listingApin(gamma);//현 ap내에서만 사용되는 고립된 가중치이므로 adar)에서 타겟 그라운드에 재할당 되어 쉐도우가
		listingApin(beta);//필요없으므로 ip만 정렬하면 된다. 다중참조 되지 앟으므로 multiArrangeLock도 내부 수행에서 무시됨.
		
		md = apTcr->instMatrix(md, apIn->qType, apIn->fdim, apIn->fshape, false, JUST_SIZEDEF);
		mz = apTcr->instMatrix(mz, apIn->qType, apIn->fdim, apIn->fshape, false, JUST_SIZEDEF);
		mv = apTcr->instMatrix(mv, apIn->qType, apIn->fdim - 1, apIn->fshape, false, JUST_SIZEDEF);
		mean = apTcr->instMatrix(mean, apIn->qType, apIn->fdim - 1, apIn->fshape, false, JUST_SIZEDEF);
		g_mz = apTcr->instMatrix(g_mz, apIn->qType, apIn->fdim, apIn->fshape, false, JUST_SIZEDEF);
		var = apTcr->instMatrix(var, apIn->qType, apIn->fdim - 1, apIn->fshape, false, JUST_SIZEDEF);
		mdmean = apTcr->instMatrix(mdmean, apIn->qType, apIn->fdim - 1, apIn->fshape, false, JUST_SIZEDEF);
	}
	intt mcalcapp(intt n) //배치를 n으로 했을때 어플라이 내부 사용 메모리 계산, 이를위해 이들 매트릭스를
	{						//위에서 메모리 할당 없이 사이즈 계산만 할수있게 생성
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
			intt gid = DMATFX(apOut)->didground;//gid 그라운드에서의 할당 공간은 위 mcalc app에서 계산되어 확보되었다.
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
			apOut->resizing2(fxr, "matmul");//아래 함수로 배치 디벤젼이 아닌 디멘젼도 리사이징할수있으나 여기서 배치차원 이하의
			//형태가 바뀌면 후에 dot와 같은 연산이 실행될 경우 여기서 변형된 플럭스가 입렵되면 에러발생한다. 나중에 모든 
			//연산이 완전 형태 변경하도록 수정한다면 아래 함수를 살린다.
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
	//[4, 3] * [3, 2] => [4, 2] 예제로 생각
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
class ApScoop : public Apply {//그라프 구성되지 않고 그라프 실행하지 않고 역전파 없다.
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